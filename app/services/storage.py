"""
Storage helpers: save uploads, export CSV, export YOLO-seg labels.

This implements a simple local filesystem storage suitable for local dev.
"""
from __future__ import annotations

import csv
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .metrics import ComponentMetrics
from .yolo_seg import InstanceSeg


def ensure_dirs(*dirs: Path) -> None:
    """Create required directories if they don't exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_bytes(path: Path, data: bytes) -> None:
    """Write raw bytes to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def decode_image_bytes_to_bgr(data: bytes) -> np.ndarray:
    """Decode image bytes into BGR np.ndarray."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image.")
    return img


def new_id(prefix: str) -> str:
    """Generate a short id."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def write_metrics_csv(csv_path: Path, image_name: str, metrics: List[ComponentMetrics]) -> None:
    """Write per-component metrics to a CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image",
            "component",
            "component_area_px",
            "total_void_area_px",
            "void_pct",
            "max_void_area_px",
            "max_void_pct",
        ])
        for m in metrics:
            writer.writerow([
                image_name,
                m.component_id,
                m.component_area_px,
                m.total_void_area_px,
                f"{m.void_pct:.6f}",
                m.max_void_area_px,
                f"{m.max_void_pct:.6f}",
            ])


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    """Get largest contour for a binary mask."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def mask_to_yolo_polygon(mask: np.ndarray, eps_frac: float = 0.002) -> List[Tuple[float, float]]:
    """
    Convert a binary mask to a YOLO-seg polygon (normalized points).
    Uses the largest external contour and simplifies it with approxPolyDP.
    """
    h, w = mask.shape[:2]
    cnt = _largest_contour(mask)
    if cnt is None:
        return []

    peri = cv2.arcLength(cnt, True)
    eps = max(1.0, eps_frac * peri)
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)

    pts: List[Tuple[float, float]] = []
    for x, y in approx:
        pts.append((float(x) / float(w), float(y) / float(h)))
    return pts


def write_yolo_seg_labels(label_path: Path, instances: List[InstanceSeg]) -> None:
    """
    Write YOLO-seg label file. One instance per line:
      cls x1 y1 x2 y2 ... (normalized polygon points)
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    for inst in instances:
        pts = mask_to_yolo_polygon(inst.mask)
        if len(pts) < 3:
            continue  # skip degenerate shapes
        coords: List[str] = []
        for x, y in pts:
            coords.extend([f"{x:.6f}", f"{y:.6f}"])
        lines.append(f"{inst.cls_id} " + " ".join(coords))

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

