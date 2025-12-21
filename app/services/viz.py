"""
Visualization helpers: create overlay images for UI.
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from .yolo_seg import InstanceSeg


def _apply_mask_overlay(image_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> None:
    """In-place alpha blend of a single mask."""
    overlay = image_bgr.copy()
    overlay[mask] = color_bgr
    cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0, dst=image_bgr)


def render_overlay(image_bgr: np.ndarray, instances: List[InstanceSeg], names: Dict[int, str]) -> np.ndarray:
    """
    Render masks + simple labels into an output image for UI.
    """
    out = image_bgr.copy()

    # Fixed, simple palette by class
    # bulle/void: blue-ish; chip: green-ish
    palette = {
        0: (255, 0, 0),   # BGR
        1: (0, 200, 0),
    }

    for idx, inst in enumerate(instances):
        color = palette.get(inst.cls_id, (200, 200, 0))
        _apply_mask_overlay(out, inst.mask, color, alpha=0.35)

        # Outline
        contours, _ = cv2.findContours(inst.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), 1)

        # Label at bounding rect
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            label = f"#{idx} {names.get(inst.cls_id, str(inst.cls_id))} {inst.conf:.2f}"
            cv2.putText(out, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(out, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return out


def encode_png(image_bgr: np.ndarray) -> bytes:
    """Encode BGR image to PNG bytes."""
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG.")
    return buf.tobytes()

