"""Evaluate or predict with Geometry CCFF v1 or v1.1 OBB weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from common import ROOT, default_weights, load_config, ultralytics_args, validate_data_yaml  # noqa: E402
from backbone_config import is_rtdetr_config  # noqa: E402
import slot_yolo  # noqa: E402,F401 - registers checkpoint classes before torch.load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "ex_config" / "config_RTDETRV2_hyE_yolo_geometry_v1_1.yaml"),
        help="experiment YAML path",
    )
    parser.add_argument("--weights", help="override the configured .pt checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, config = load_config(args.config)
    test_config = config.get("test", {})
    split = test_config.get("split", "val")
    validate_data_yaml(config["data"], required_split=split)
    weights = Path(args.weights or test_config.get("weights") or default_weights(config)).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    from ultralytics import YOLO
    from ultralytics.nn.modules.head import OBB

    if is_rtdetr_config(config):
        from ultralytics.models.rtdetr import RTDETR
        model = RTDETR(str(weights))
    else:
        model = YOLO(str(weights), task=config["task"])
    is_slot_checkpoint = hasattr(model.model, "configure_runtime")
    checkpoint_model_type = "slot_yolo" if is_slot_checkpoint else "baseline_yolo"
    if checkpoint_model_type != config["model_type"]:
        raise ValueError(
            f"Config selects model_type={config['model_type']!r}, but "
            f"{weights.name!r} is a {checkpoint_model_type!r} checkpoint"
        )
    checkpoint_task = (
        model.model.slot_task_name()
        if hasattr(model.model, "slot_task_name")
        else "obb" if isinstance(model.model.model[-1], OBB) else "detect"
    )
    if checkpoint_task != config["task"]:
        raise ValueError(
            f"Config selects task={config['task']!r}, but {weights.name!r} "
            f"contains a {checkpoint_task!r} head. HBB and OBB weights are "
            "not interchangeable."
        )
    if is_slot_checkpoint:
        model.model.validate_neck_selection(config)
        model.model.configure_runtime(config.get("memory", {}))
        print(f"Module version: {getattr(model.model, 'module_version', 'v1')}")
    print(f"Model/task: {checkpoint_model_type} / {checkpoint_task}")
    metrics = model.val(data=config["data"], **ultralytics_args(test_config, "weights"))
    print(f"Metrics: {metrics.results_dict}")

    predict = config.get("predict", {})
    if predict.get("enabled", False):
        if not predict.get("source"):
            raise ValueError("predict.enabled is true but predict.source is null")
        model.predict(**ultralytics_args(predict, "enabled"))


if __name__ == "__main__":
    main()
