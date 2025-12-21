"""
YOLOv8 instance segmentation service wrapper.

- Loads a YOLO segmentation model once.
- Runs prediction on a BGR numpy image (OpenCV).
- Returns list of InstanceSeg with (cls_id, conf, mask[bool]).
- Robust to CUDA failures: auto-fallback to CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO


@dataclass
class InstanceSeg:
    """Single instance segmentation prediction."""
    cls_id: int
    conf: float
    mask: np.ndarray  # bool (H, W)


class YoloSegService:
    """Ultralytics YOLO segmentation predictor."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.device = device  # "cpu" or "0" or "cuda:0"

    def _should_fallback_to_cpu(self, err: Exception) -> bool:
        msg = str(err).lower()
        return (
            "no kernel image is available" in msg
            or "cuda error" in msg
            or "sm_61" in msg
            or "not compatible with the current pytorch installation" in msg
        )

    def predict(self, image_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.7) -> List[InstanceSeg]:
        """
        Run YOLO segmentation prediction.

        Args:
            image_bgr: OpenCV BGR image (H, W, 3)
            conf: confidence threshold
            iou: iou threshold

        Returns:
            List of InstanceSeg
        """
        try:
            results = self.model.predict(
                source=image_bgr,
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            if self.device != "cpu" and self._should_fallback_to_cpu(e):
                self.device = "cpu"
                results = self.model.predict(
                    source=image_bgr,
                    conf=conf,
                    iou=iou,
                    device="cpu",
                    verbose=False,
                )
            else:
                raise

        if not results:
            return []

        r0 = results[0]
        if r0.masks is None or r0.boxes is None:
            return []

        # masks.data: (N, H, W) tensor
        masks = r0.masks.data.detach().cpu().numpy()
        cls = r0.boxes.cls.detach().cpu().numpy()
        confs = r0.boxes.conf.detach().cpu().numpy()

        instances: List[InstanceSeg] = []
        for i in range(masks.shape[0]):
            mask_bool = masks[i] > 0.5
            instances.append(
                InstanceSeg(
                    cls_id=int(cls[i]),
                    conf=float(confs[i]),
                    mask=mask_bool,
                )
            )

        return instances

