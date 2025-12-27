# app/utils/normalize_json.py
"""
Utilities for normalizing JSON-ish payloads returned by external services.

This module is designed to handle "double-JSON" (a JSON string that itself
contains JSON), and other common wrappers, without requiring redeploys.

Typical cases handled:
- dict / list already parsed JSON (pass-through)
- bytes / bytearray containing JSON text
- str containing JSON text
- "double-JSON": str -> JSON string -> JSON dict/list
- common wrapper keys: {"result": ...}, {"body": ...}, {"data": ...}, {"output": ...}

The main entry point is `normalize_maybe_double_json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Tuple, Union


JsonLike = Union[dict, list]
WrapperKeys = Tuple[str, ...]


@dataclass(frozen=True)
class NormalizedJSON:
    """Result of normalization with some light diagnostics."""
    value: Any
    # How many json.loads steps were applied on top of the original.
    decode_steps: int = 0
    # Wrapper keys unwrapped in order (outer -> inner).
    unwrapped_keys: Tuple[str, ...] = ()


class JSONNormalizationError(ValueError):
    """Raised when a payload cannot be normalized into valid JSON."""


def _looks_like_json_text(s: str) -> bool:
    """
    Quick heuristic: does the string look like JSON text?

    We purposely do not require full validity here; json.loads will validate.
    """
    t = s.strip()
    if not t:
        return False
    return t[0] in "{[\"" or t in ("true", "false", "null") or t[0].isdigit() or t[0] == "-"


def _try_json_loads(text: str) -> Tuple[bool, Any]:
    """Attempt json.loads; return (ok, value)."""
    try:
        return True, json.loads(text)
    except Exception:
        return False, None


def _unwrap_common_wrappers(
    obj: Any,
    wrapper_keys: WrapperKeys,
    max_unwrap_depth: int,
) -> Tuple[Any, Tuple[str, ...]]:
    """
    Unwrap common nesting patterns like {"result": {...}} or {"body": "..."}.
    Returns (unwrapped_obj, keys_unwrapped).
    """
    keys_used = []
    cur = obj

    for _ in range(max_unwrap_depth):
        if not isinstance(cur, Mapping):
            break

        found_key = None
        for k in wrapper_keys:
            if k in cur:
                found_key = k
                break

        if found_key is None:
            break

        keys_used.append(found_key)
        cur = cur[found_key]

    return cur, tuple(keys_used)


def normalize_maybe_double_json(
    payload: Any,
    *,
    wrapper_keys: WrapperKeys = ("result", "body", "data", "output", "outputs", "response"),
    max_decode_steps: int = 3,
    max_unwrap_depth: int = 3,
) -> NormalizedJSON:
    """
    Normalize an unknown response payload that may be JSON, JSON string, or double-JSON.

    Args:
        payload: Any value (dict/list/str/bytes/etc).
        wrapper_keys: Keys to unwrap if the payload is a mapping.
        max_decode_steps: Maximum number of json.loads iterations to attempt.
        max_unwrap_depth: Maximum wrapper layers to unwrap.

    Returns:
        NormalizedJSON containing the normalized value plus diagnostics.

    Raises:
        JSONNormalizationError if normalization fails.
    """
    decode_steps = 0

    # Step 0: bytes -> str (utf-8 with replacement to avoid hard crashes)
    cur: Any = payload
    if isinstance(cur, (bytes, bytearray)):
        cur = bytes(cur).decode("utf-8", errors="replace")

    # Step 1: unwrap wrapper keys early (common with proxy gateways)
    cur, unwrapped = _unwrap_common_wrappers(cur, wrapper_keys, max_unwrap_depth)

    # Step 2: iteratively decode JSON text if needed
    while decode_steps < max_decode_steps:
        if isinstance(cur, (dict, list)):
            # Already structured JSON.
            return NormalizedJSON(value=cur, decode_steps=decode_steps, unwrapped_keys=unwrapped)

        if isinstance(cur, str):
            s = cur.strip()

            # If it doesn't look like JSON, stop trying to decode.
            if not _looks_like_json_text(s):
                return NormalizedJSON(value=cur, decode_steps=decode_steps, unwrapped_keys=unwrapped)

            ok, parsed = _try_json_loads(s)
            if not ok:
                # Not valid JSON text; return as-is (caller can decide).
                return NormalizedJSON(value=cur, decode_steps=decode_steps, unwrapped_keys=unwrapped)

            # We successfully decoded one layer.
            cur = parsed
            decode_steps += 1

            # After decoding, wrapper keys might appear again.
            cur, more_unwrapped = _unwrap_common_wrappers(cur, wrapper_keys, max_unwrap_depth)
            if more_unwrapped:
                unwrapped = unwrapped + more_unwrapped

            continue

        # For other primitive types (int/float/bool/None), just return as-is.
        return NormalizedJSON(value=cur, decode_steps=decode_steps, unwrapped_keys=unwrapped)

    # If we exit loop due to max_decode_steps, return current value
    return NormalizedJSON(value=cur, decode_steps=decode_steps, unwrapped_keys=unwrapped)


def ensure_json_object(
    normalized: NormalizedJSON,
    *,
    allow_list: bool = True,
) -> JsonLike:
    """
    Ensure normalized.value is a JSON object (dict) or list.

    Args:
        normalized: Output from normalize_maybe_double_json.
        allow_list: Whether list is acceptable.

    Returns:
        dict or list.

    Raises:
        JSONNormalizationError: if the value is not acceptable.
    """
    v = normalized.value
    if isinstance(v, dict):
        return v
    if allow_list and isinstance(v, list):
        return v
    raise JSONNormalizationError(f"Expected JSON object/list, got {type(v).__name__}: {repr(v)[:200]}")

