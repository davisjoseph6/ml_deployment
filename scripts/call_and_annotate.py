#!/usr/bin/env python3
"""
call_and_annotate.py

Call an Azure ML online endpoint that expects {"image_b64": "..."} and returns:
{"detections":[{"cls":int,"conf":float,"box":[x1,y1,x2,y2],"mask":[[x,y],...]}, ...]}

This version uses Option A (no redeploy): it normalizes “double-JSON” responses
(i.e., when the endpoint returns JSON that itself contains JSON strings).

Outputs:
- response.json (parsed JSON)
- annotated.png (image with boxes + polygon masks)
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

# Ensure repo root is on sys.path so `import app...` works even when running
# `python3 scripts/call_and_annotate.py` from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the AML client that handles double-JSON normalization.
from app.services.aml_client import AMLClientConfig, score as aml_score  # noqa: E402


def _b64_image(path: str) -> str:
    """Read image bytes and return base64 ASCII string (no newlines)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _get_env_first(*names: str, default: str | None = None) -> str | None:
    """Return first defined env var among names, else default."""
    for n in names:
        v = os.environ.get(n)
        if v is not None and str(v).strip() != "":
            return v
    return default


def _coerce_int(value: str | None, default: int) -> int:
    """Parse int from env-like string; fallback to default."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
            draw.text(
                (x1, max(0, y1 - 12)),
                f"cls={cls_id} conf={float(conf):.3f}",
                fill=(255, 255, 255, 255),
            )

        # Polygon mask
        if isinstance(mask, list) and len(mask) >= 3 and isinstance(mask[0], list):
            pts: List[Tuple[float, float]] = [(float(x), float(y)) for x, y in mask]
            draw.polygon(pts, fill=(255, 255, 255, 50), outline=(255, 255, 255, 200))

    out = Image.alpha_composite(img, overlay).convert("RGB")
    out.save(out_path)


def main() -> None:
    """Entrypoint via environment variables."""
    # Support BOTH naming schemes:
    # - New: AML_SCORING_URI / AML_KEY
    # - Legacy: SCORING_URI / API_KEY
    scoring_uri = _get_env_first("AML_SCORING_URI", "SCORING_URI")
    api_key = _get_env_first("AML_KEY", "API_KEY")

    if not scoring_uri:
        raise KeyError("Missing scoring URI. Set AML_SCORING_URI (or SCORING_URI).")
    if not api_key:
        raise KeyError("Missing API key. Set AML_KEY (or API_KEY).")

    image_path = _get_env_first("IMAGE_PATH", default="test_images/board_01.jpg") or "test_images/board_01.jpg"

    # Optional: header to select deployment (still useful even if traffic=100% to one deployment)
    deployment_name = _get_env_first("DEPLOYMENT_NAME", default="blue")

    # Optional: request timeout (seconds)
    timeout_s = _coerce_int(_get_env_first("TIMEOUT_S"), default=120)

    # Optional output file names
    response_path = _get_env_first("RESPONSE_JSON", default="response.json") or "response.json"
    annotated_path = _get_env_first("ANNOTATED_PNG", default="annotated.png") or "annotated.png"

    # Build config explicitly so we can pass timeout + deployment name.
    cfg = AMLClientConfig(
        scoring_uri=scoring_uri,
        api_key=api_key,
        deployment_name=deployment_name,
        timeout_s=float(timeout_s),
    )

    payload: Dict[str, Any] = {"image_b64": _b64_image(image_path)}

    # This returns a *normalized* dict even if AML sends JSON-as-string (double-JSON).
    resp = aml_score(payload, config=cfg)

    with open(response_path, "w", encoding="utf-8") as f:
        json.dump(resp, f, indent=2)

    detections = resp.get("detections", [])
    if isinstance(detections, list):
        annotate_image(image_path, detections, annotated_path)

    print(f"Wrote {response_path} and {annotated_path}")


if __name__ == "__main__":
    main()

