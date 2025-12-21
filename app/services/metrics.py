"""
Void-rate computation: per-component total void% and max-void%.

Definitions:
- void_pct = (sum void areas inside component) / (component area)
- max_void_pct = (largest single void area inside component) / (component area)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ComponentMetrics:
    """Metrics per component."""
    component_id: int
    component_area_px: int
    total_void_area_px: int
    void_pct: float
    max_void_area_px: int
    max_void_pct: float


def mask_area(mask: np.ndarray) -> int:
    """Pixel area of a boolean mask."""
    return int(mask.astype(np.uint8).sum())


def assign_voids_to_components(
    component_masks: List[np.ndarray],
    void_masks: List[np.ndarray],
    overlap_thresh: float = 0.5,
) -> Tuple[Dict[int, List[np.ndarray]], List[np.ndarray]]:
    """
    Assign each void to the component with maximum overlap.

    Returns:
      assigned: dict component_index -> list(void masks)
      unassigned: list of void masks that didn't meet overlap threshold
    """
    assigned: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(component_masks))}
    unassigned: List[np.ndarray] = []

    for v in void_masks:
        v_area = mask_area(v)
        if v_area == 0 or len(component_masks) == 0:
            unassigned.append(v)
            continue

        best_i = -1
        best_inter = 0

        for i, c in enumerate(component_masks):
            inter = mask_area(v & c)
            if inter > best_inter:
                best_inter = inter
                best_i = i

        if best_i >= 0 and (best_inter / v_area) >= overlap_thresh:
            assigned[best_i].append(v)
        else:
            unassigned.append(v)

    return assigned, unassigned


def compute_metrics(
    component_masks: List[np.ndarray],
    void_masks: List[np.ndarray],
    overlap_thresh: float,
) -> Tuple[List[ComponentMetrics], List[np.ndarray]]:
    """
    Compute per-component metrics and return also any unassigned voids.
    """
    assigned, unassigned = assign_voids_to_components(
        component_masks,
        void_masks,
        overlap_thresh=overlap_thresh,
    )

    out: List[ComponentMetrics] = []

    for i, comp in enumerate(component_masks):
        comp_area = mask_area(comp)
        voids = assigned.get(i, [])

        void_areas = [mask_area(v) for v in voids]
        total_void = int(sum(void_areas))
        max_void = int(max(void_areas)) if void_areas else 0

        void_pct = (total_void / comp_area) if comp_area > 0 else 0.0
        max_void_pct = (max_void / comp_area) if comp_area > 0 else 0.0

        out.append(
            ComponentMetrics(
                component_id=i + 1,
                component_area_px=comp_area,
                total_void_area_px=total_void,
                void_pct=float(void_pct),
                max_void_area_px=max_void,
                max_void_pct=float(max_void_pct),
            )
        )

    return out, unassigned

