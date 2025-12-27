# app/services/aml_infer.py
"""
Helpers to call the Azure ML endpoint from the FastAPI backend.

- Takes raw uploaded image bytes
- Base64 encodes to {"image_b64": "..."}
- Calls app.services.aml_client.score() which normalizes double-JSON
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, Optional

from app.services.aml_client import AMLClientConfig, score as aml_score


def _b64_bytes(blob: bytes) -> str:
    """Encode bytes -> base64 ASCII string (no newlines)."""
    return base64.b64encode(blob).decode("ascii")


def score_image_bytes(
    image_bytes: bytes,
    *,
    scoring_uri: Optional[str] = None,
    api_key: Optional[str] = None,
    deployment_name: Optional[str] = None,
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
    """
    Call AML endpoint and return normalized JSON as dict/list.
    """
    scoring_uri = scoring_uri or os.environ.get("AML_SCORING_URI", "").strip()
    api_key = api_key or os.environ.get("AML_KEY", "").strip()
    deployment_name = deployment_name or os.environ.get("AML_DEPLOYMENT_NAME", "").strip() or None

    if not scoring_uri:
        raise RuntimeError("Missing AML_SCORING_URI.")
    if not api_key:
        raise RuntimeError("Missing AML_KEY.")

    cfg = AMLClientConfig(
        scoring_uri=scoring_uri,
        api_key=api_key,
        deployment_name=deployment_name,
        timeout_s=timeout_s,
    )

    payload = {"image_b64": _b64_bytes(image_bytes)}
    out = aml_score(payload, config=cfg)

    # Your endpoint returns {"detections": [...]} so normalize to dict.
    if isinstance(out, dict):
        return out

    # If your endpoint ever returns a list at top-level, wrap it.
    return {"output": out}

