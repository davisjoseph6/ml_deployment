# app/services/viz_aml.py
"""
Render overlays (boxes + polygon masks) from AML detections.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def _as_int_tuple(x: float, y: float) -> Tuple[int, int]:
    return int(round(float(x))), int(round(float(y)))


def render_overlay_from_aml(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Draw boxes + polygon masks on a copy of image_bgr.

    Expects AML format:
      det["box"] = [x1,y1,x2,y2]
      det["mask"] = [[x,y], ...]
      det["cls"], det["conf"]
    """
    base = image_bgr.copy()
    overlay = image_bgr.copy()

    for det in detections:
        box = det.get("box")
        mask = det.get("mask")
        cls_id = det.get("cls", -1)
        conf = det.get("conf", 0.0)

        # Box
        if isinstance(box, list) and len(box) == 4:
            x1, y1 = _as_int_tuple(box[0], box[1])
            x2, y2 = _as_int_tuple(box[2], box[3])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(
                overlay,
                f"cls={cls_id} conf={float(conf):.3f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Polygon mask
        if isinstance(mask, list) and len(mask) >= 3 and isinstance(mask[0], list):
            pts = np.array([_as_int_tuple(x, y) for x, y in mask], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (255, 255, 255))
            cv2.polylines(overlay, [pts], isClosed=True, color=(255, 255, 255), thickness=2)

    # Alpha blend overlay onto base
    out = cv2.addWeighted(overlay, 0.25, base, 0.75, 0)
    return out

