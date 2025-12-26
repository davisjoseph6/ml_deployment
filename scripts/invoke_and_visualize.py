#!/usr/bin/env python3
"""
Invoke Azure ML online endpoint with an image (base64) and optionally visualize detections.

Usage:
  python3 scripts/invoke_and_visualize.py \
    --endpoint pcbqc-endpt \
    --deployment blue \
    --image test_images/board_01.jpg \
    --out-json out/board_01.json \
    --out-vis out/board_01_overlay.png \
    --min-conf 0.95
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw


def b64_of_image(path: Path) -> str:
    """Return base64 (no newlines) of the image file."""
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def invoke_az(endpoint: str, deployment: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke endpoint via Azure CLI and return parsed JSON dict."""
    req = json.dumps(payload)
    proc = subprocess.run(
        [
            "az",
            "ml",
            "online-endpoint",
            "invoke",
            "-n",
            endpoint,
            "--deployment-name",
            deployment,
            "--request-file",
            "/dev/stdin",
        ],
        input=req.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"az invoke failed (code {proc.returncode})\nSTDERR:\n{proc.stderr.decode('utf-8', errors='replace')}"
        )

    out = proc.stdout.decode("utf-8").strip()

    # Many times it's a JSON string that contains JSON
    parsed = json.loads(out)
    if isinstance(parsed, str):
        parsed = json.loads(parsed)

    if not isinstance(parsed, dict):
        raise TypeError(f"Unexpected response type: {type(parsed)}")

    return parsed


def draw_overlay(img_path: Path, resp: Dict[str, Any], out_path: Path, min_conf: float) -> None:
    """Draw boxes + polygons for detections >= min_conf."""
    im = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    dets: List[Dict[str, Any]] = resp.get("detections", []) or []
    kept = 0

    for det in dets:
        conf = float(det.get("conf", 0.0))
        if conf < min_conf:
            continue

        box = det.get("box", None)
        if isinstance(box, list) and len(box) == 4:
            x1, y1, x2, y2 = map(float, box)
            d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 220), width=2)

        mask = det.get("mask", None)
        if isinstance(mask, list) and len(mask) >= 3:
            pts: List[Tuple[float, float]] = []
            for p in mask:
                if isinstance(p, list) and len(p) == 2:
                    pts.append((float(p[0]), float(p[1])))
            if len(pts) >= 3:
                d.polygon(pts, outline=(0, 255, 0, 220))

        kept += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(im, overlay).convert("RGB").save(out_path)
    print(f"Saved overlay: {out_path} (kept {kept} detections with conf >= {min_conf})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--deployment", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-vis", default="")
    ap.add_argument("--min-conf", type=float, default=0.95)
    args = ap.parse_args()

    img_path = Path(args.image).resolve()
    payload = {"image_b64": b64_of_image(img_path)}

    resp = invoke_az(args.endpoint, args.deployment, payload)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(resp, indent=2))
        print(f"Saved JSON: {out_json}")

    if args.out_vis:
        draw_overlay(img_path, resp, Path(args.out_vis), args.min_conf)


if __name__ == "__main__":
    main()

