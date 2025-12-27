# app/services/aml_client.py
"""
Azure ML scoring client.

This module wraps calls to an AML online endpoint and normalizes responses
that might be returned as "double-JSON" (JSON as a string), or wrapped under
keys like "result"/"body"/"data".

Environment variables expected:
- AML_SCORING_URI: full URL to the /score endpoint
- AML_KEY: Bearer token

Optional environment variables:
- AML_TIMEOUT_S: request timeout seconds (default 60)
- AML_VERIFY_TLS: true/false (default true)

This client returns a Python dict/list (already normalized), raising errors
with useful diagnostics otherwise.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import requests

from app.utils.normalize_json import (
    JSONNormalizationError,
    ensure_json_object,
    normalize_maybe_double_json,
)

JsonLike = Union[dict, list]


class AMLClientError(RuntimeError):
    """Raised when the AML scoring call fails or returns an invalid payload."""


@dataclass(frozen=True)
class AMLClientConfig:
    """Configuration for the AML client."""
    scoring_uri: str
    api_key: str
    deployment_name: Optional[str] = None  # e.g. "blue" / "green"
    timeout_s: float = 60.0
    verify_tls: bool = True  # set False only if you *really* need to (not recommended)


def _get_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise AMLClientError(f"Missing required environment variable: {name}")
    return v


def load_config_from_env() -> AMLClientConfig:
    """
    Load AMLClientConfig from environment variables.
    """
    scoring_uri = _get_env("AML_SCORING_URI")
    api_key = _get_env("AML_KEY")
    timeout_s = float(os.environ.get("AML_TIMEOUT_S", "60").strip())
    verify_tls = os.environ.get("AML_VERIFY_TLS", "true").strip().lower() not in ("0", "false", "no")
    deployment_name = os.environ.get("AML_DEPLOYMENT_NAME", "").strip() or None

    return AMLClientConfig(
        scoring_uri=scoring_uri,
        api_key=api_key,
        deployment_name=deployment_name,
        timeout_s=timeout_s,
        verify_tls=verify_tls,
    )


def score(
    request_payload: Dict[str, Any],
    *,
    config: Optional[AMLClientConfig] = None,
    session: Optional[requests.Session] = None,
) -> JsonLike:
    """
    Send a scoring request to AML and return normalized JSON response.

    Args:
        request_payload: dict to send as JSON (e.g., {"image_b64": "..."}).
        config: optional AMLClientConfig; if None, loaded from env.
        session: optional requests.Session.

    Returns:
        Normalized JSON: dict or list.

    Raises:
        AMLClientError: network / http / invalid response problems.
    """
    cfg = config or load_config_from_env()
    s = session or requests.Session()

    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Force a specific deployment when doing blue/green tests.
    if cfg.deployment_name:
        headers["azureml-model-deployment"] = cfg.deployment_name

    # Serialize ourselves so we know exactly what goes over the wire (and can log safely if needed)
    try:
        body = json.dumps(request_payload).encode("utf-8")
    except Exception as e:
        raise AMLClientError(f"Request payload is not JSON-serializable: {e}") from e

    t0 = time.time()
    try:
        resp = s.post(
            cfg.scoring_uri,
            headers=headers,
            data=body,
            timeout=cfg.timeout_s,
            verify=cfg.verify_tls,
        )
    except requests.RequestException as e:
        raise AMLClientError(f"Failed to call AML endpoint: {e}") from e
    dt_ms = int((time.time() - t0) * 1000)

    # HTTP errors: include useful context, but avoid dumping huge bodies.
    if not resp.ok:
        snippet = (resp.text or "")[:800]
        raise AMLClientError(
            f"AML endpoint returned HTTP {resp.status_code} in {dt_ms}ms. "
            f"Body snippet: {snippet}"
        )

    # Prefer resp.text over resp.json() to survive cases where service returns JSON string.
    raw_text = resp.text

    # First normalize the raw text (which might be JSON or double-JSON).
    normalized = normalize_maybe_double_json(raw_text)

    # Sometimes the response is already JSON, but requests decoded it differently; try resp.json if text path failed
    # (we only do this if normalize returned the original string unchanged).
    if isinstance(normalized.value, str) and normalized.decode_steps == 0:
        try:
            parsed = resp.json()
            normalized = normalize_maybe_double_json(parsed)
        except Exception:
            # keep prior normalized; we'll validate below
            pass

    try:
        out = ensure_json_object(normalized, allow_list=True)
    except JSONNormalizationError as e:
        snippet = (raw_text or "")[:800]
        raise AMLClientError(
            f"AML response could not be normalized into JSON object/list. "
            f"decode_steps={normalized.decode_steps}, unwrapped_keys={normalized.unwrapped_keys}. "
            f"Raw snippet: {snippet}"
        ) from e

    return out

