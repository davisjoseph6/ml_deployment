"""
FastAPI local app:
- Analyze page: upload 1+ images, run YOLO-seg, compute void metrics, export CSV
- Annotate page: prelabel with YOLO-seg, refine mask with MobileSAM, validate to pending retrain
- Retrain: background job + status polling
"""
from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import os
TORCH_DEVICE = os.getenv("TORCH_DEVICE", "cpu")

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO as UltralyticsYOLO

from app.core.config import settings
from app.services.metrics import compute_metrics
from app.services.retrain import retrain_manager
from app.services.sam_refine import MobileSamService
from app.services.storage import (
    decode_image_bytes_to_bgr,
    ensure_dirs,
    new_id,
    save_bytes,
    write_metrics_csv,
    write_yolo_seg_labels,
)
from app.services.viz import encode_png, render_overlay
from app.services.yolo_seg import InstanceSeg, YoloSegService

APP_VERSION = "0.1.0-local"

app = FastAPI(title="PCB QC Active Learning (Local)", version=APP_VERSION)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "ui" / "templates"))
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "ui" / "static")),
    name="static",
)

# -----------------------------
# Model singletons (initialized on startup)
# -----------------------------
_yolo: YoloSegService | None = None
_sam: MobileSamService | None = None
YOLO_NAMES: Dict[int, str] = {}

# -----------------------------
# Annotate session state
# -----------------------------
ANNOTATE_STATE: Dict[str, Dict[str, Any]] = {}
# structure:
# ANNOTATE_STATE[image_id] = {
#   "image_bgr": np.ndarray,
#   "instances": List[InstanceSeg],
#   "filename": str,
#   "source_path": str,
#   "created_at": float,
#   "last_access": float,
# }

MAX_ANNOTATE_SESSIONS = 20
ANNOTATE_TTL_SECONDS = 30 * 60  # 30 minutes


def _get_yolo() -> YoloSegService:
    if _yolo is None:
        raise RuntimeError("YOLO service not initialized.")
    return _yolo


def _get_sam() -> MobileSamService:
    if _sam is None:
        raise RuntimeError("SAM service not initialized.")
    return _sam


def _bgr_to_base64_png(image_bgr: np.ndarray) -> str:
    png_bytes = encode_png(image_bgr)
    return base64.b64encode(png_bytes).decode("utf-8")


def _split_instances(instances: List[InstanceSeg]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    chip_masks = [i.mask for i in instances if i.cls_id == settings.chip_cls_id]
    void_masks = [i.mask for i in instances if i.cls_id == settings.void_cls_id]
    return chip_masks, void_masks


def _replace_or_add_instance(
    instances: List[InstanceSeg],
    cls_id: int,
    new_mask: np.ndarray,
    conf: float = 1.0,
) -> List[InstanceSeg]:
    """
    Replace the instance of same cls_id with max overlap with new_mask; if none overlap, add it.
    """
    best_idx = -1
    best_inter = 0

    for idx, inst in enumerate(instances):
        if inst.cls_id != cls_id:
            continue
        inter = int((inst.mask & new_mask).astype(np.uint8).sum())
        if inter > best_inter:
            best_inter = inter
            best_idx = idx

    if best_idx >= 0 and best_inter > 0:
        instances[best_idx] = InstanceSeg(cls_id=cls_id, conf=conf, mask=new_mask)
    else:
        instances.append(InstanceSeg(cls_id=cls_id, conf=conf, mask=new_mask))

    return instances


def _prune_annotate_state(now: float | None = None) -> None:
    """
    Prune annotate sessions:
    - Drop entries older than TTL (based on last_access)
    - Enforce a max number of sessions by removing least-recently accessed
    """
    now = now if now is not None else time.time()

    # TTL prune
    expired = [
        k for k, v in ANNOTATE_STATE.items()
        if (now - float(v.get("last_access", v.get("created_at", now)))) > ANNOTATE_TTL_SECONDS
    ]
    for k in expired:
        ANNOTATE_STATE.pop(k, None)

    # Max sessions prune (LRU-ish)
    if len(ANNOTATE_STATE) > MAX_ANNOTATE_SESSIONS:
        items = sorted(
            ANNOTATE_STATE.items(),
            key=lambda kv: float(kv[1].get("last_access", kv[1].get("created_at", now))),
        )
        extra = len(items) - MAX_ANNOTATE_SESSIONS
        for i in range(extra):
            ANNOTATE_STATE.pop(items[i][0], None)


def _validate_yolo_class_mapping(model_path: Path) -> Dict[int, str]:
    """
    Load Ultralytics model metadata and validate expected class IDs/names.
    Fails fast if someone swaps weights with different class ordering.
    """
    m = UltralyticsYOLO(str(model_path))
    names = m.names

    # Ultralytics usually gives dict[int,str]. If it gives list, convert.
    if isinstance(names, list):
        names_dict: Dict[int, str] = {i: n for i, n in enumerate(names)}
    else:
        names_dict = {int(k): str(v) for k, v in names.items()}

    expected_void = "bulle"
    expected_chip = "chip"

    if names_dict.get(settings.void_cls_id) != expected_void or names_dict.get(settings.chip_cls_id) != expected_chip:
        raise RuntimeError(
            "Unexpected model class mapping. "
            f"Expected {settings.void_cls_id}='{expected_void}', {settings.chip_cls_id}='{expected_chip}', "
            f"got names={names_dict}"
        )

    return names_dict


@app.on_event("startup")
def startup() -> None:
    """
    Initialize directories and load models once.
    """
    ensure_dirs(
        settings.incoming_dir,
        settings.reports_dir,
        settings.dataset_images_dir,
        settings.dataset_labels_dir,
        settings.pending_images_dir,
        settings.pending_labels_dir,
        settings.model_versions_dir,
    )

    global _yolo, _sam, YOLO_NAMES  # noqa: PLW0603
    _yolo = YoloSegService(str(settings.yolo_seg_model_path), device=TORCH_DEVICE)
    _sam = MobileSamService(str(settings.mobile_sam_path), device=TORCH_DEVICE)

    YOLO_NAMES = _validate_yolo_class_mapping(settings.yolo_seg_model_path)


@app.get("/health")
def health() -> JSONResponse:
    y = _get_yolo()
    return JSONResponse(
        {
            "status": "ok",
            "version": APP_VERSION,
            "yolo_device": y.device,
            "yolo_model_path": str(settings.yolo_seg_model_path),
            "sam_model_path": str(settings.mobile_sam_path),
        }
    )


@app.get("/version")
def version() -> JSONResponse:
    return JSONResponse(
        {
            "app_version": APP_VERSION,
            "model_names": YOLO_NAMES,
            "yolo_model_path": str(settings.yolo_seg_model_path),
        }
    )


@app.get("/", response_class=HTMLResponse)
def root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("analyze.html", {"request": request})


@app.get("/analyze", response_class=HTMLResponse)
def analyze_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("analyze.html", {"request": request})


@app.get("/annotate", response_class=HTMLResponse)
def annotate_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("annotate.html", {"request": request})


@app.post("/api/analyze")
async def api_analyze(files: List[UploadFile] = File(...)) -> JSONResponse:
    yolo = _get_yolo()

    results: List[Dict[str, Any]] = []

    for f in files:
        content = await f.read()
        image_bgr = decode_image_bytes_to_bgr(content)

        img_id = new_id("img")
        safe_name = f"{img_id}_{f.filename}".replace("/", "_").replace("\\", "_")
        in_path = settings.incoming_dir / safe_name
        save_bytes(in_path, content)

        instances = yolo.predict(image_bgr, conf=settings.yolo_conf, iou=settings.yolo_iou)
        chip_masks, void_masks = _split_instances(instances)

        metrics, unassigned = compute_metrics(
            component_masks=chip_masks,
            void_masks=void_masks,
            overlap_thresh=settings.void_assign_thresh,
        )

        overlay = render_overlay(image_bgr, instances, YOLO_NAMES)
        overlay_b64 = _bgr_to_base64_png(overlay)

        csv_name = f"report_{img_id}.csv"
        csv_path = settings.reports_dir / csv_name
        write_metrics_csv(csv_path, image_name=f.filename, metrics=metrics)

        results.append(
            {
                "image_id": img_id,
                "filename": f.filename,
                "overlay_png_base64": overlay_b64,
                "metrics": [m.__dict__ for m in metrics],
                "unassigned_voids_count": len(unassigned),
                "csv_report": f"/api/reports/{csv_name}",
            }
        )

    return JSONResponse({"results": results})


@app.get("/api/reports/{name}")
def get_report(name: str) -> Response:
    path = settings.reports_dir / name
    if not path.exists():
        return Response(status_code=404, content=b"Not found")
    return Response(
        content=path.read_bytes(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@app.post("/api/prelabel")
async def api_prelabel(file: UploadFile = File(...)) -> JSONResponse:
    yolo = _get_yolo()

    content = await file.read()
    image_bgr = decode_image_bytes_to_bgr(content)

    image_id = new_id("anno")
    safe_name = f"{image_id}_{file.filename}".replace("/", "_").replace("\\", "_")
    in_path = settings.incoming_dir / safe_name
    save_bytes(in_path, content)

    instances = yolo.predict(image_bgr, conf=settings.yolo_conf, iou=settings.yolo_iou)
    overlay = render_overlay(image_bgr, instances, YOLO_NAMES)
    overlay_b64 = _bgr_to_base64_png(overlay)

    now = time.time()
    ANNOTATE_STATE[image_id] = {
        "image_bgr": image_bgr,
        "instances": instances,
        "filename": file.filename,
        "source_path": str(in_path),
        "created_at": now,
        "last_access": now,
    }
    _prune_annotate_state(now=now)

    return JSONResponse(
        {
            "image_id": image_id,
            "filename": file.filename,
            "overlay_png_base64": overlay_b64,
            "instances": [
                {
                    "id": idx,
                    "cls_id": inst.cls_id,
                    "cls_name": YOLO_NAMES.get(inst.cls_id, str(inst.cls_id)),
                    "conf": inst.conf,
                }
                for idx, inst in enumerate(instances)
            ],
        }
    )


@app.post("/api/refine")
def api_refine(
    image_id: str = Form(...),
    target_class: str = Form(...),  # "chip" or "bulle"
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...),
) -> JSONResponse:
    sam = _get_sam()

    _prune_annotate_state()

    if image_id not in ANNOTATE_STATE:
        return JSONResponse({"error": "Unknown image_id"}, status_code=404)

    if target_class not in ("chip", "bulle"):
        return JSONResponse({"error": "Invalid target_class. Use 'chip' or 'bulle'."}, status_code=400)

    state = ANNOTATE_STATE[image_id]
    state["last_access"] = time.time()

    image_bgr: np.ndarray = state["image_bgr"]
    instances: List[InstanceSeg] = state["instances"]

    cls_id = settings.chip_cls_id if target_class == "chip" else settings.void_cls_id

    # SAM requires RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    refined = sam.refine_with_box(image_rgb=image_rgb, box_xyxy=(x1, y1, x2, y2))

    instances = _replace_or_add_instance(instances, cls_id=cls_id, new_mask=refined, conf=1.0)
    state["instances"] = instances

    chip_masks, void_masks = _split_instances(instances)
    metrics, unassigned = compute_metrics(
        component_masks=chip_masks,
        void_masks=void_masks,
        overlap_thresh=settings.void_assign_thresh,
    )

    overlay = render_overlay(image_bgr, instances, YOLO_NAMES)
    overlay_b64 = _bgr_to_base64_png(overlay)

    return JSONResponse(
        {
            "image_id": image_id,
            "overlay_png_base64": overlay_b64,
            "metrics": [m.__dict__ for m in metrics],
            "unassigned_voids_count": len(unassigned),
        }
    )


@app.post("/api/validate")
def api_validate(image_id: str = Form(...)) -> JSONResponse:
    _prune_annotate_state()

    if image_id not in ANNOTATE_STATE:
        return JSONResponse({"error": "Unknown image_id"}, status_code=404)

    state = ANNOTATE_STATE[image_id]
    filename = state["filename"]
    src_path = Path(state["source_path"])
    instances: List[InstanceSeg] = state["instances"]

    # Save image + labels into pending_retrain
    base = f"{image_id}_{Path(filename).stem}"
    img_dst = settings.pending_images_dir / f"{base}{src_path.suffix}"
    lbl_dst = settings.pending_labels_dir / f"{base}.txt"

    img_bytes = src_path.read_bytes()
    save_bytes(img_dst, img_bytes)

    write_yolo_seg_labels(lbl_dst, instances)

    # Free memory: remove session after validation
    ANNOTATE_STATE.pop(image_id, None)

    return JSONResponse(
        {"ok": True, "pending_image": str(img_dst), "pending_label": str(lbl_dst)}
    )


@app.post("/api/retrain")
def api_retrain() -> JSONResponse:
    started = retrain_manager.start()
    if not started:
        return JSONResponse({"ok": False, "message": "Training already running"}, status_code=409)
    return JSONResponse({"ok": True, "message": "Training started"})


@app.get("/api/retrain/status")
def api_retrain_status() -> JSONResponse:
    return JSONResponse(retrain_manager.status())

