#!/usr/bin/env python3
"""
YOLOv8 instance segmentation service wrapper.

- Loads a YOLO segmentation model once.
- Runs prediction on a BGR numpy image (OpenCV).
- Returns list of InstanceSeg with (cls_id, conf, mask[bool]).
- Thread-safe: internal lock around model inference / reload.

GPU enforcement:
- If require_gpu=True, any CUDA-related failure will raise (no fallback).
- If require_gpu=True, device cannot be "cpu".
- If require_gpu=False, will attempt CPU fallback on common CUDA incompat errors.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional

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

    def __init__(self, model_path: str, device: str = "cpu", require_gpu: bool = False) -> None:
        """
        Args:
            model_path: path to YOLO-seg .pt weights
            device: "cpu" or "0" or "cuda:0" (Ultralytics accepts several forms)
            require_gpu: if True, never fall back to CPU (fail fast)
        """
        self._lock = threading.Lock()
        self.model_path = model_path
        self.device = device
        self.require_gpu = require_gpu

        if self.require_gpu and self.device == "cpu":
            raise ValueError("require_gpu=True but device='cpu'. Use device='0' (or 'cuda:0').")

        self.model = YOLO(model_path)

    def _should_fallback_to_cpu(self, err: Exception) -> bool:
        msg = str(err).lower()
        return (
            "no kernel image is available" in msg
            or "cuda error" in msg
            or "sm_" in msg
            or "not compatible with the current pytorch installation" in msg
            or "cudnn" in msg
            or "cublas" in msg
        )

    def reload(self, model_path: Optional[str] = None) -> None:
        """
        Reload model weights safely.

        Args:
            model_path: If provided, replaces current model_path.
        """
        with self._lock:
            if model_path is not None:
                self.model_path = model_path

            if self.require_gpu and self.device == "cpu":
                raise ValueError("require_gpu=True but device='cpu'. Use device='0' (or 'cuda:0').")

            self.model = YOLO(self.model_path)

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
        if self.require_gpu and self.device == "cpu":
            raise RuntimeError("GPU is required but YOLO device is 'cpu' (misconfiguration).")

        with self._lock:
            try:
                results = self.model.predict(
                    source=image_bgr,
                    conf=conf,
                    iou=iou,
                    device=self.device,
                    verbose=False,
                )
            except Exception as e:
                # GPU guarantee: never fallback if require_gpu=True
                if self.require_gpu:
                    raise

                # Best-effort fallback to CPU for known CUDA incompat errors
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

