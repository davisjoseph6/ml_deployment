#!/usr/bin/env python3
"""
call_and_annotate.py

Call an Azure ML online endpoint that expects {"image_b64": "..."} and returns:
{"detections":[{"cls":int,"conf":float,"box":[x1,y1,x2,y2],"mask":[[x,y],...]}, ...]}

Outputs:
- response.json (parsed JSON)
- annotated.png (image with boxes + polygon masks)
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Tuple

import requests
from PIL import Image, ImageDraw


def _b64_image(path: str) -> str:
    """Read image bytes and return base64 ASCII string (no newlines)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _parse_maybe_double_encoded(obj: Any) -> Any:
    """Handle cases where the response is JSON or a JSON string."""
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            return obj
    return obj


def call_endpoint(
    scoring_uri: str,
    api_key: str,
    image_path: str,
    deployment_name: str | None = None,
    timeout_s: int = 120,
) -> Dict[str, Any]:
    """POST request to scoring_uri using key auth and return parsed JSON."""
    payload = {"image_b64": _b64_image(image_path)}

    headers = {
        "Content-Type": "application/json",
        # Azure ML "key" auth typically uses Bearer with the key value
        "Authorization": f"Bearer {api_key}",
    }
    if deployment_name:
        # Useful even if traffic=100% to one deployment
        headers["azureml-model-deployment"] = deployment_name

    r = requests.post(scoring_uri, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()

    data = r.json()
    data = _parse_maybe_double_encoded(data)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected response type: {type(data)}")
    return data


def annotate_image(
    image_path: str,
    detections: List[Dict[str, Any]],
    out_path: str,
) -> None:
    """Draw boxes + polygons on the image and write to out_path."""
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for det in detections:
        cls_id = det.get("cls", -1)
        conf = det.get("conf", 0.0)
        box = det.get("box")
        mask = det.get("mask")

        # Box
        if isinstance(box, list) and len(box) == 4:
            x1, y1, x2, y2 = [float(v) for v in box]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255, 255), width=2)
            draw.text((x1, max(0, y1 - 12)), f"cls={cls_id} conf={conf:.3f}", fill=(255, 255, 255, 255))

        # Polygon mask
        if isinstance(mask, list) and len(mask) >= 3 and isinstance(mask[0], list):
            pts: List[Tuple[float, float]] = [(float(x), float(y)) for x, y in mask]
            # semi-transparent fill + solid outline
            draw.polygon(pts, fill=(255, 255, 255, 50), outline=(255, 255, 255, 200))

    out = Image.alpha_composite(img, overlay).convert("RGB")
    out.save(out_path)


def main() -> None:
    """Entrypoint via environment variables."""
    scoring_uri = os.environ["SCORING_URI"]
    api_key = os.environ["API_KEY"]
    image_path = os.environ.get("IMAGE_PATH", "test_images/board_01.jpg")
    deployment_name = os.environ.get("DEPLOYMENT_NAME", "blue")

    resp = call_endpoint(
        scoring_uri=scoring_uri,
        api_key=api_key,
        image_path=image_path,
        deployment_name=deployment_name,
    )

    with open("response.json", "w", encoding="utf-8") as f:
        json.dump(resp, f, indent=2)

    detections = resp.get("detections", [])
    if isinstance(detections, list):
        annotate_image(image_path, detections, "annotated.png")

    print("Wrote response.json and annotated.png")


if __name__ == "__main__":
    main()

