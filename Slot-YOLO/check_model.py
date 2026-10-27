"""Build Slot-YOLO and run a small forward/backward architecture check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from common import ROOT, load_config  # noqa: E402
from slot_yolo import SlotYOLODetectionModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "ex_config" / "config_RTDETRV2_hyE_yolo_geometry_v1_1.yaml"),
    )
    parser.add_argument("--imgsz", type=int, default=128, help="small multiple of 32 for the check")
    parser.add_argument("--backward", action="store_true", help="also backpropagate through model outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, config = load_config(args.config)
    # This architecture check uses synthetic images, so only class names are needed.
    with Path(config["data"]).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    names = data["names"]
    nc = len(names)
    if args.imgsz < 64 or args.imgsz % 32:
        raise ValueError("--imgsz must be a multiple of 32 and at least 64")

    model = SlotYOLODetectionModel(
        cfg=config["model"],
        nc=nc,
        ch=3,
        verbose=True,
        slot_config=config,
    )
    model.train(args.backward)
    image = torch.rand(1, 3, args.imgsz, args.imgsz)
    output = model(image)
    raw_predictions = output[1] if isinstance(output, tuple) else output
    predictions = (
        raw_predictions["one2many"]
        if isinstance(raw_predictions, dict) and "one2many" in raw_predictions
        else raw_predictions
    )
    features = predictions["feats"] if isinstance(predictions, dict) else []
    print(f"{config['task'].upper()} feature shapes:", [tuple(feature.shape) for feature in features])
    print("Stride:", model.stride.tolist())
    print("Feature pipeline:", model.feature_pipeline_description)
    print("Neck:", model.neck_description)
    print("Enhancer trainable:", any(parameter.requires_grad for parameter in model.enhancer.parameters()))
    if args.backward:
        scalar = sum(value.float().mean() for value in predictions.values() if isinstance(value, torch.Tensor))
        scalar.backward()
        print("Backward: OK")


if __name__ == "__main__":
    main()
