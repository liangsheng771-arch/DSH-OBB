"""The YOLOv8s OBB backbone used by the retained OCM experiments."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model_config" / "yolov8s-obb.yaml"


def selected_backbone(config):
    value = str(config.get("backbone", "yolov8s")).lower()
    if value != "yolov8s":
        raise ValueError(f"Only backbone: yolov8s is retained, got {value!r}")
    return value


def backbone_model_path(config):
    selected_backbone(config)
    if str(config.get("task", "obb")).lower() != "obb":
        raise ValueError("The retained model configuration supports task: obb")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Backbone model YAML not found: {MODEL_PATH}")
    return str(MODEL_PATH)


def is_rtdetr_config(config):
    return False


def validate_backbone_runtime(config):
    selected_backbone(config)
