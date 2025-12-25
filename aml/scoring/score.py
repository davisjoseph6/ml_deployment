"""
Azure ML inference scoring script for YOLOv8 segmentation.

Request (JSON):
{
  "image_b64": "<base64-encoded image bytes>"
}

Response (JSON):
{
  "detections": [
    {
      "cls": int,
      "conf": float,
      "box": [x1, y1, x2, y2],
      "mask": [[x, y], ...]  # polygon, may be null
    }
  ]
}
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:
    raise RuntimeError("opencv-python (cv2) is required in the inference image.") from e

try:
    from ultralytics import YOLO  # type: ignore
except Exception as e:
    raise RuntimeError("ultralytics is required in the inference image.") from e


_MODEL: Optional["YOLO"] = None


def _model_path() -> str:
    # If later you register a model asset, you can switch to AZUREML_MODEL_DIR.
    # For now, default to the repo-copied artifact path inside the image.
    return os.getenv("PCBQC_YOLO_WEIGHTS", "/app/ai_model_artifacts/yolo_segmentation/best.pt")


def _device() -> str:
    return os.getenv("YOLO_DEVICE", "cpu")


def init() -> None:
    global _MODEL
    weights = _model_path()
    if not os.path.exists(weights):
        raise FileNotFoundError(f"YOLO weights not found at: {weights}")
    _MODEL = YOLO(weights)


def run(raw_data: Any) -> Dict[str, Any]:
    if _MODEL is None:
        raise RuntimeError("Model not initialized. init() did not run.")

    # AML can pass str/bytes/dict depending on route; normalize to dict
    if isinstance(raw_data, (bytes, bytearray)):
        raw_data = raw_data.decode("utf-8")
    if isinstance(raw_data, str):
        payload = json.loads(raw_data)
    elif isinstance(raw_data, dict):
        payload = raw_data
    else:
        payload = json.loads(json.dumps(raw_data))

    b64 = payload.get("image_b64")
    if not b64:
        return {"error": "Missing 'image_b64' in request."}

    img_bytes = base64.b64decode(b64)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"error": "Failed to decode image bytes."}

    results = _MODEL.predict(
        source=bgr,
        device=_device(),
        conf=float(payload.get("conf", 0.25)),
        iou=float(payload.get("iou", 0.7)),
        verbose=False,
    )

    dets: List[Dict[str, Any]] = []
    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    masks = getattr(r0, "masks", None)

    if boxes is None or boxes.xyxy is None:
        return {"detections": dets}

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((xyxy.shape[0],), dtype=float)
    clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((xyxy.shape[0],), dtype=int)

    mask_polys = None
    if masks is not None and getattr(masks, "xy", None) is not None:
        mask_polys = masks.xy  # list of (N,2) polygons per instance

    for i in range(xyxy.shape[0]):
        poly = None
        if mask_polys is not None and i < len(mask_polys) and mask_polys[i] is not None:
            poly = [[float(x), float(y)] for x, y in mask_polys[i].tolist()]

        dets.append(
            {
                "cls": int(clss[i]),
                "conf": float(confs[i]),
                "box": [float(v) for v in xyxy[i].tolist()],
                "mask": poly,
            }
        )

    return {"detections": dets}
