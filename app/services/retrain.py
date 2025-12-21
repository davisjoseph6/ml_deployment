"""
Local retraining orchestration (background thread).

- Merges pending_retrain into dataset
- Trains YOLO segmentation using config/yolo_train.yaml (plus generated data.yaml)
- Publishes new model to ai_model_artifacts/yolo_segmentation/best.pt + versioned copy
"""
from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from ultralytics import YOLO

from app.core.config import settings


@dataclass
class TrainStatus:
    state: str  # "idle" | "running" | "done" | "failed"
    message: str
    started_at: float | None = None
    finished_at: float | None = None
    model_version: str | None = None


class RetrainManager:
    """Background retraining manager."""
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._status = TrainStatus(state="idle", message="")

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self._status.state,
                "message": self._status.message,
                "started_at": self._status.started_at,
                "finished_at": self._status.finished_at,
                "model_version": self._status.model_version,
            }

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._status = TrainStatus(state="running", message="Training started", started_at=time.time())
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True

    def _run(self) -> None:
        try:
            # Merge pending -> dataset
            self._merge_pending()

            # Build / ensure data.yaml exists
            data_yaml = self._ensure_data_yaml()

            # Load train config
            cfg = {}
            if settings.train_config_path.exists():
                cfg = yaml.safe_load(settings.train_config_path.read_text(encoding="utf-8")) or {}

            # Train
            model = YOLO(str(settings.yolo_seg_model_path))
            results = model.train(
                data=str(data_yaml),
                **cfg,
            )

            # Publish model artifacts (Ultralytics writes into runs/segment/...)
            # Locate best.pt from the latest run directory
            best_pt = self._find_latest_best_pt()
            if best_pt is None:
                raise RuntimeError("Could not locate trained best.pt under runs/segment/")

            settings.model_versions_dir.mkdir(parents=True, exist_ok=True)
            version = time.strftime("v%Y%m%d_%H%M%S")
            version_path = settings.model_versions_dir / f"{version}.pt"

            shutil.copy2(best_pt, version_path)
            shutil.copy2(best_pt, settings.yolo_seg_model_path)

            with self._lock:
                self._status.state = "done"
                self._status.message = "Training finished"
                self._status.finished_at = time.time()
                self._status.model_version = version

        except Exception as e:
            with self._lock:
                self._status.state = "failed"
                self._status.message = f"{type(e).__name__}: {e}"
                self._status.finished_at = time.time()

    def _merge_pending(self) -> None:
        settings.dataset_images_dir.mkdir(parents=True, exist_ok=True)
        settings.dataset_labels_dir.mkdir(parents=True, exist_ok=True)
        settings.pending_images_dir.mkdir(parents=True, exist_ok=True)
        settings.pending_labels_dir.mkdir(parents=True, exist_ok=True)

        # Move all pending images/labels into dataset, renaming to avoid collisions
        for img_path in settings.pending_images_dir.glob("*"):
            dst = settings.dataset_images_dir / img_path.name
            if dst.exists():
                dst = settings.dataset_images_dir / f"{img_path.stem}_{int(time.time())}{img_path.suffix}"
            shutil.move(str(img_path), str(dst))

        for lbl_path in settings.pending_labels_dir.glob("*"):
            dst = settings.dataset_labels_dir / lbl_path.name
            if dst.exists():
                dst = settings.dataset_labels_dir / f"{lbl_path.stem}_{int(time.time())}{lbl_path.suffix}"
            shutil.move(str(lbl_path), str(dst))

    def _ensure_data_yaml(self) -> Path:
        """
        Create a minimal YOLO data.yaml pointing at dataset/images for train+val.
        For local dev, we use the same folder for train/val unless you add splits later.
        """
        data_yaml = settings.dataset_dir / "data.yaml"
        data = {
            "path": str(settings.dataset_dir),
            "train": "images",
            "val": "images",
            "names": {0: "bulle", 1: "chip"},
        }
        data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return data_yaml

    def _find_latest_best_pt(self) -> Path | None:
        runs = settings.project_root / "runs" / "segment"
        if not runs.exists():
            return None
        best_candidates = list(runs.glob("*/weights/best.pt"))
        if not best_candidates:
            return None
        best_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return best_candidates[0]


retrain_manager = RetrainManager()

