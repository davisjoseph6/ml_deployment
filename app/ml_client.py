#!/usr/bin/env python3
"""
AzureML online endpoint client wrapper.

- Calls the AKS-backed AzureML online endpoint scoring URI.
- Handles the common case where the service returns JSON as an escaped string.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict


class AzureMLClient:
    """Minimal AzureML endpoint client with robust JSON parsing."""

    def __init__(self) -> None:
        self.scoring_uri = os.environ["SCORING_URI"]
        self.api_key = os.environ["AML_KEY"]

    def invoke(self, payload: Dict[str, Any], timeout_s: int = 30) -> Dict[str, Any]:
        """Invoke the endpoint and return parsed JSON."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.scoring_uri,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")

        # Common AML pattern: response is a JSON string that itself contains JSON.
        # Example: "\"{\\\"detections\\\": [...]}\""
        try:
            obj = json.loads(raw)
            if isinstance(obj, str):
                return json.loads(obj)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # Last resort: return raw
        return {"raw": raw}

