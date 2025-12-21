"""
MobileSAM refinement service.

- Loads MobileSAM once.
- Refines a segmentation mask given an RGB image and a bbox (x1,y1,x2,y2).
- Defaults to CPU and auto-fallback if CUDA fails.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from mobile_sam import SamPredictor, sam_model_registry


class MobileSamService:
    """MobileSAM predictor wrapper."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.checkpoint_path = checkpoint_path
        self.device_str = device
        self.device = torch.device("cpu")  # force cpu by default
        self.model = None
        self.predictor = None
        self._load_model()

    def _load_model(self) -> None:
        # Always start on CPU; if you *really* want CUDA later, switch device_str
        self.device = torch.device("cpu" if self.device_str == "cpu" else self.device_str)

        model = sam_model_registry["vit_t"](checkpoint=self.checkpoint_path)
        model.to(self.device)
        model.eval()

        self.model = model
        self.predictor = SamPredictor(model)

    def _should_fallback_to_cpu(self, err: Exception) -> bool:
        msg = str(err).lower()
        return (
            "no kernel image is available" in msg
            or "cuda error" in msg
            or "sm_61" in msg
            or "not compatible with the current pytorch installation" in msg
        )

    def refine_with_box(
        self,
        image_rgb: np.ndarray,
        box_xyxy: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Refine using a bounding box prompt.

        Args:
            image_rgb: RGB uint8 image (H, W, 3)
            box_xyxy: (x1, y1, x2, y2) in image pixel coordinates

        Returns:
            mask bool array (H, W)
        """
        assert self.predictor is not None

        try:
            self.predictor.set_image(image_rgb)
            box = np.array(box_xyxy, dtype=np.float32)[None, :]
            masks, _, _ = self.predictor.predict(box=box, multimask_output=False)
            return masks[0].astype(bool)
        except Exception as e:
            # fallback to cpu if something CUDA-ish happens
            if self.device_str != "cpu" and self._should_fallback_to_cpu(e):
                self.device_str = "cpu"
                self._load_model()
                self.predictor.set_image(image_rgb)
                box = np.array(box_xyxy, dtype=np.float32)[None, :]
                masks, _, _ = self.predictor.predict(box=box, multimask_output=False)
                return masks[0].astype(bool)
            raise

