"""
Application configuration for local development.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Central configuration used by both services and API endpoints."""
    project_root: Path = Path(__file__).resolve().parents[2]

    # Data dirs
    data_dir: Path = project_root / "data"
    incoming_dir: Path = data_dir / "incoming"
    dataset_dir: Path = data_dir / "dataset"
    pending_dir: Path = data_dir / "pending_retrain"
    reports_dir: Path = data_dir / "reports"

    dataset_images_dir: Path = dataset_dir / "images"
    dataset_labels_dir: Path = dataset_dir / "labels"
    pending_images_dir: Path = pending_dir / "images"
    pending_labels_dir: Path = pending_dir / "labels"

    # Model artifacts
    yolo_seg_model_path: Path = project_root / "ai_model_artifacts" / "yolo_segmentation" / "best.pt"
    mobile_sam_path: Path = project_root / "ai_model_artifacts" / "segmentation" / "mobile_sam.pt"

    # YOLO inference settings
    yolo_conf: float = 0.25
    yolo_iou: float = 0.5

    # Assignment
    void_assign_thresh: float = 0.5  # overlap ratio wrt void area

    # YOLO class mapping (validated at startup)
    void_cls_id: int = 0  # 'bulle'
    chip_cls_id: int = 1  # 'chip'

    # Training config
    train_config_path: Path = project_root / "config" / "yolo_train.yaml"
    model_versions_dir: Path = project_root / "ai_model_artifacts" / "yolo_segmentation" / "versions"


settings = Settings()

