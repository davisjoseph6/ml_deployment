#!/usr/bin/env python3
"""
Backend route that accepts an image upload and calls Azure ML Online Endpoint.
"""

import base64
import json
import os
import requests
from flask import Blueprint, request, jsonify

bp = Blueprint("predict", __name__)

AML_SCORING_URI = os.environ["AML_SCORING_URI"]
AML_KEY = os.environ["AML_KEY"]

def normalize_response(payload):
    """Normalize AzureML response which might be a JSON string or object."""
    if isinstance(payload, str):
        s = payload.strip()
        if s and s[0] in "{[":
            return json.loads(s)
    return payload

@bp.post("/api/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400

    f = request.files["file"]
    img_bytes = f.read()

    # Build request payload (adjust key names to match YOUR request.json format)
    payload = {
        "image": base64.b64encode(img_bytes).decode("utf-8")
    }

    headers = {
        "Authorization": f"Bearer {AML_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(AML_SCORING_URI, headers=headers, json=payload, timeout=30)

    # If Azure returns a JSON string, normalize it
    try:
        data = r.json()
    except Exception:
        data = r.text

    data = normalize_response(data)

    if r.status_code >= 400:
        return jsonify({"error": "upstream failed", "status": r.status_code, "detail": data}), 502

    return jsonify(data)

