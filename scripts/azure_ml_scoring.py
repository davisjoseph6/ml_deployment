#!/usr/bin/env python3
"""
Azure ML scoring entrypoint.

- init(): load model once
- run(): accept request JSON and return a Python dict (NOT json.dumps)
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List

import numpy as np
from PIL import Image


def _to_py(x: Any) -> Any:
    """Convert numpy types / arrays recursively into JSON-serializable Python types."""
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_py(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_py(v) for k, v in x.items()}
    return x


model = None


def init() -> None:
    global model
    # TODO: load your model here
    # model = ...
    pass


def run(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Azure ML will JSON-decode the request body and pass a dict here (most setups).
    Return a dict; the server will serialize it as JSON.
    """
    try:
        # Example: accept base64 image
        if "image_b64" in data:
            img_bytes = base64.b64decode(data["image_b64"])
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            raise ValueError("Missing 'image_b64' in request.")

        # TODO: run inference
        # raw_out = model_predict(model, img, **data.get("params", {}))

        raw_out = {"detections": []}  # placeholder
        return _to_py(raw_out)

    except Exception as e:
        # Make failures explicit & easy to debug in logs
        return {"error": str(e)}

