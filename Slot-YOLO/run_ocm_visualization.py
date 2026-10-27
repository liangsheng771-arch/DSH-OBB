"""Run the final OCM OBB checkpoint and render every image in the test split.

Run from the Ultralytics project root on the training server:
    python Slot-YOLO/run_ocm_visualization.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def class_names(data: dict[str, Any]) -> list[str]:
    names = data.get("names")
    if isinstance(names, list):
        result = [str(name) for name in names]
    elif isinstance(names, dict):
        result = [str(name) for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
    else:
        raise ValueError("Dataset YAML must contain a list or mapping of class names")
    if int(data.get("nc", len(result))) != len(result):
        raise ValueError("Dataset nc does not match the number of class names")
    return result


def image_source(data: dict[str, Any], data_yaml: Path) -> Path:
    root = resolve_path(str(data.get("path", data_yaml.parent)), data_yaml.parent)
    split = data.get("test")
    if not isinstance(split, str) or not split:
        raise ValueError("Dataset YAML must define a single test image directory")
    return resolve_path(split, root)


def new_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("predict_%Y%m%d_%H%M%S")
    for index in range(1, 1000):
        suffix = "" if index == 1 else f"_{index}"
        candidate = output_root / f"{stem}{suffix}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("Could not create a unique test output directory")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "ex_config" / "ocm_final_visualization.yaml",
        help="OCM image inference configuration YAML",
    )
    parser.add_argument("--weights", help="Override the checkpoint path")
    parser.add_argument("--source", help="Override the test image directory")
    parser.add_argument("--device", help="Override the CUDA device, e.g. 4 or cpu")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    data_yaml = resolve_path(str(config["data"]), config_path.parent)
    data = load_yaml(data_yaml)
    expected_names = class_names(data)
    weights = resolve_path(str(args.weights or config["weights"]), config_path.parent)
    source = (
        resolve_path(args.source, Path.cwd())
        if args.source
        else image_source(data, data_yaml)
    )
    output_root = resolve_path(str(config["output_dir"]), config_path.parent)
    device = str(args.device or config["device"])

    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    if not source.is_dir():
        raise NotADirectoryError(f"Test image directory not found: {source}")
    images = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No images found in: {source}")
    names = [path.name for path in images]
    if len(set(names)) != len(names) or len({path.stem for path in images}) != len(images):
        raise ValueError("Test image names or stems are duplicated")
    expected_count = config.get("expected_images")
    if expected_count is not None and len(images) != int(expected_count):
        raise ValueError(f"Expected {expected_count} test images, found {len(images)} in {source}")

    print(f"Checkpoint: {weights}")
    print(f"Dataset: {data_yaml}")
    print(f"Input images: {source} ({len(images)} files)")
    print(f"Classes: {expected_names}")
    print(f"Device: {device}")
    import slot_yolo  # noqa: F401,E402 - register custom checkpoint classes for torch.load
    from ultralytics import YOLO  # noqa: E402

    model = YOLO(str(weights), task="obb")
    checkpoint = model.model
    if not hasattr(checkpoint, "slot_task_name") or checkpoint.slot_task_name() != "obb":
        raise ValueError("Checkpoint is not a Slot-YOLO OBB model")
    neck = getattr(checkpoint, "rtdetrv2_hye_yolo_neck", None)
    if neck is None or not getattr(neck, "use_geometry_ccff", False) or getattr(neck, "use_geometry_ccff_v1_1", False):
        raise ValueError("Checkpoint does not contain the expected Geometry CCFF v1 neck")
    checkpoint_names = [str(model.names[index]) for index in range(len(expected_names))]
    if checkpoint_names != expected_names:
        raise ValueError(f"Checkpoint class order {checkpoint_names} differs from dataset {expected_names}")

    run_dir = new_run_dir(output_root)
    annotated_dir = run_dir / "images"
    labels_dir = run_dir / "labels"
    annotated_dir.mkdir()
    labels_dir.mkdir()
    print(f"Output: {run_dir}", flush=True)

    expected_files = set(names)
    seen: set[str] = set()
    manifest = run_dir / "predictions.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "detections", "visualization", "label_file"])
        results = model.predict(
            source=str(source),
            device=device,
            imgsz=int(config["imgsz"]),
            batch=int(config["batch"]),
            conf=float(config["conf"]),
            iou=float(config["iou"]),
            max_det=int(config["max_det"]),
            stream=True,
            save=False,
            verbose=False,
            project=str(run_dir),
            name="prediction_runtime",
            exist_ok=True,
        )
        for result in results:
            filename = Path(result.path).name
            if filename not in expected_files or filename in seen:
                raise RuntimeError(f"Unexpected or duplicate inference result: {filename}")
            if result.obb is None:
                raise RuntimeError(f"Expected OBB predictions for {filename}")
            target_image = annotated_dir / filename
            target_label = labels_dir / f"{Path(filename).stem}.txt"
            result.save(
                filename=str(target_image),
                labels=True,
                conf=True,
                boxes=True,
                line_width=int(config["line_width"]),
            )
            result.save_txt(target_label, save_conf=True)
            target_label.touch(exist_ok=True)  # also keep an empty label file for images with no detections
            if not target_image.is_file() or target_image.stat().st_size == 0:
                raise RuntimeError(f"Visualization was not saved: {target_image}")
            writer.writerow([filename, len(result.obb), str(target_image), str(target_label)])
            seen.add(filename)
            if len(seen) % 100 == 0 or len(seen) == len(images):
                print(f"Visualized {len(seen)}/{len(images)} images", flush=True)

    missing = expected_files - seen
    if missing:
        raise RuntimeError(f"Inference missed {len(missing)} images; first missing: {sorted(missing)[:5]}")
    if len(list(annotated_dir.iterdir())) != len(images):
        raise RuntimeError("Visualization count does not match the test image count")

    summary = {
        "checkpoint": str(weights),
        "data_yaml": str(data_yaml),
        "source": str(source),
        "classes": expected_names,
        "input_images": len(images),
        "visualized_images": len(seen),
        "conf": float(config["conf"]),
        "iou": float(config["iou"]),
        "imgsz": int(config["imgsz"]),
        "device": device,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Complete: {len(seen)} visualizations in {annotated_dir}")
    print(f"Summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
