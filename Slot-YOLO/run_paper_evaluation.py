"""Paper evaluation runner. Run with Python; no pytest or COCO evaluation package is required.

Default: python Slot-YOLO/run_paper_evaluation.py
See docs/paper_evaluation.md for metric definitions and the external adapter contract.
The METRIC CORE below is shared by every adapter; keep it unchanged for comparisons.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import importlib
import inspect
import json
import logging
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
METRIC_VERSION = "yolo-area-ap-v1"
OBB_AREA_POLICY = "absolute_predicted_sides_v1"
IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10, dtype=np.float32)
# COCO areaRng uses inclusive endpoints; an exact boundary can belong to two bins.
AREA_RANGES = {"all": (0.0, float("inf")), "S": (0.0, 32.0**2),
               "M": (32.0**2, 96.0**2), "L": (96.0**2, float("inf"))}


# ---------------- METRIC CORE: independent of model loading / inference ----------------
def match_iou_first(overlap: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Ultralytics default greedy matching (GT rows, detection columns).

    Wrong-class overlaps must already be zero. Preserve the original ordering of
    both np.unique calls: changing it changes the Ultralytics matching rule.
    """
    pairs = np.array(np.nonzero(overlap >= threshold)).T
    if len(pairs) > 1:
        pairs = pairs[overlap[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
        pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
        pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
    return pairs[:, 0], pairs[:, 1]


def integrate_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Ultralytics 101-point interpolation + trapezoidal integration, pinned here."""
    if not len(recall):
        return 0.0
    mrec = np.concatenate(([0.0], recall, [recall[-1]], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    grid = np.linspace(0, 1, 101)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(np.interp(grid, mrec, mpre), grid))


def validate_records(records: list[dict], nc: int, max_det: int) -> None:
    """Fail on invalid adapter data instead of silently producing paper numbers."""
    if not records:
        raise ValueError("The adapter returned no images")
    seen = set()
    for r in records:
        image_id = str(r["image_id"])
        if image_id in seen:
            raise ValueError(f"Duplicate image in evaluation: {image_id}")
        seen.add(image_id)
        for key in ("gt_cls", "pred_cls", "confidence", "gt_area", "pred_area", "overlap"):
            r[key] = np.asarray(r[key])
            if not np.isfinite(r[key]).all():
                raise ValueError(f"Non-finite {key}: {image_id}")
        ng, nd = len(r["gt_cls"]), len(r["pred_cls"])
        for key, size in (("gt_cls", ng), ("gt_area", ng), ("pred_cls", nd),
                          ("pred_area", nd), ("confidence", nd)):
            if r[key].shape != (size,):
                raise ValueError(f"Invalid {key} shape: {image_id}")
        if r["overlap"].shape != (ng, nd) or nd > max_det:
            raise ValueError(f"Invalid overlap shape / max_det exceeded: {image_id}")
        for key in ("gt_cls", "pred_cls"):
            values = r[key]
            if np.any((values < 0) | (values >= nc) | (values != values.astype(int))):
                raise ValueError(f"Class ID outside dataset names: {image_id}")
            r[key] = values.astype(np.int64)
        if any(np.any(r[k] < 0) for k in ("gt_area", "pred_area")):
            raise ValueError(f"Negative area: {image_id}")
        if np.any((r["confidence"] < 0) | (r["confidence"] > 1)):
            raise ValueError(f"Confidence outside [0, 1]: {image_id}")
        if np.any((r["overlap"] < -1e-6) | (r["overlap"] > 1 + 1e-6)):
            raise ValueError(f"Overlap outside [0, 1]: {image_id}")


def evaluate_records(records: list[dict], names: list[str], max_det: int) -> dict:
    """Compute macro AP for all / S / M / L using one fixed evaluation kernel.

    Size extension: match in-range GT first; then match remaining detections to
    out-of-range GT one-to-one and ignore these matches. Ignore unmatched
    out-of-range detections. A TP is never discarded because of predicted area.
    Ignore masks are computed separately at every IoU threshold.
    This is an Ultralytics-compatible area extension, not native COCOeval.
    """
    validate_records(records, len(names), max_det)
    scores = np.concatenate([r["confidence"] for r in records])
    pred_cls = np.concatenate([r["pred_cls"] for r in records])
    order = np.argsort(-scores)  # same global ordering as Ultralytics ap_per_class
    pred_cls = pred_cls[order]
    output = {}
    for group, (low, high) in AREA_RANGES.items():
        true_positives, ignored, targets = [], [], []
        for r in records:
            gt_in = (r["gt_area"] >= low) & (r["gt_area"] <= high)
            dt_in = (r["pred_area"] >= low) & (r["pred_area"] <= high)
            overlap = r["overlap"] * (r["gt_cls"][:, None] == r["pred_cls"][None, :])
            tp = np.zeros((len(r["pred_cls"]), len(IOU_THRESHOLDS)), dtype=bool)
            ignore = np.zeros_like(tp)
            for j, threshold in enumerate(IOU_THRESHOLDS):
                _, matched = match_iou_first(overlap[gt_in], float(threshold))
                tp[matched, j] = True
                if group != "all":
                    remaining = np.flatnonzero(~tp[:, j])
                    _, ignored_matches = match_iou_first(overlap[~gt_in][:, remaining], float(threshold))
                    ignore[remaining[ignored_matches], j] = True
                    ignore[:, j] |= (~tp[:, j]) & (~dt_in)
            true_positives.append(tp)
            ignored.append(ignore)
            targets.append(r["gt_cls"][gt_in])
        tp = np.concatenate(true_positives)[order]
        ignore = np.concatenate(ignored)[order]
        counts = np.bincount(np.concatenate(targets), minlength=len(names))
        aps = np.zeros((len(names), len(IOU_THRESHOLDS)), dtype=float)
        for c in np.flatnonzero(counts):
            selected = pred_cls == c
            for j in range(len(IOU_THRESHOLDS)):
                correct = tp[selected & ~ignore[:, j], j]
                if not len(correct):
                    continue
                tpc = correct.cumsum()
                fpc = (~correct).cumsum()
                aps[c, j] = integrate_ap(tpc / (counts[c] + 1e-16), tpc / (tpc + fpc))
        present = counts > 0
        output[group] = {
            "AP": float(aps[present].mean()) if present.any() else None,
            "AP50": float(aps[present, 0].mean()) if present.any() else None,
            "AP75": float(aps[present, 5].mean()) if present.any() else None,
            "instances": int(counts.sum()),
            "classes_with_gt": int(present.sum()),
            "per_class": [
                {"class_id": c, "class_name": name, "instances": int(counts[c]),
                 "AP50": float(aps[c, 0]) if counts[c] else None,
                 "AP75": float(aps[c, 5]) if counts[c] else None,
                 "AP": float(aps[c].mean()) if counts[c] else None}
                for c, name in enumerate(names)
            ],
        }
    return output


# ---------------- IO, reporting and configuration ----------------
def resolve_path(value: str, base: Path) -> Path:
    p = Path(value).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        result = yaml.safe_load(f)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return result


def names_list(names: Any) -> list[str]:
    if isinstance(names, dict) and names:
        names = {int(k): str(v) for k, v in names.items()}
        if sorted(names) != list(range(len(names))):
            raise ValueError("Class IDs must start at 0 and be contiguous")
        return [names[i] for i in range(len(names))]
    if isinstance(names, (list, tuple)) and names:
        return [str(x) for x in names]
    raise ValueError("Class names must be a non-empty list or indexed mapping")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prepare_dataset(spec: dict, config_dir: Path, run_dir: Path) -> tuple[Path, str, list[str]]:
    source = resolve_path(spec["data"], config_dir)
    data = load_yaml(source)
    label_names = names_list(data["names"])
    if int(data.get("nc", len(label_names))) != len(label_names):
        raise ValueError(f"Dataset nc/names mismatch: {source}")
    names = names_list(spec.get("model_names", label_names))
    if len(set(label_names)) != len(label_names) or len(set(names)) != len(names):
        raise ValueError("Class names must be unique for unambiguous label mapping")
    missing = [name for name in label_names if name not in names]
    if missing:
        raise ValueError(f"Dataset classes absent from configured model_names: {missing}")
    mapping = [names.index(name) for name in label_names]
    spec["class_mapping"] = {"label_names": label_names, "model_names": names,
                             "label_to_model_id": mapping, "source_yaml": str(source)}
    # The validator and model share the checkpoint class namespace. Raw labels
    # are mapped in build_dataset(), before worker processes start. Never rewrite labels/cache.
    data["names"] = dict(enumerate(names))
    data["nc"] = len(names)
    if mapping != list(range(len(names))):
        data["paper_evaluation_label_mapping"] = spec["class_mapping"]
        data["paper_evaluation_note"] = "Requires run_paper_evaluation.py to map raw label IDs in memory; not a standalone dataset config."
    root = (resolve_path(str(spec["data_root"]), config_dir) if spec.get("data_root")
            else resolve_path(str(data.get("path") or source.parent), source.parent))
    data["path"] = str(root)
    data.pop("download", None)
    for split in ("train", "val", "test"):
        values = data.get(split)
        if values:
            paths = [str(resolve_path(str(x), root)) for x in (values if isinstance(values, list) else [values])]
            data[split] = paths if isinstance(values, list) else paths[0]
    def available(split: str) -> bool:
        paths = data.get(split)
        return bool(paths) and all(Path(p).exists() for p in (paths if isinstance(paths, list) else [paths]))
    split = str(spec.get("split", "auto"))
    if split == "auto":
        split = "test" if available("test") else "val"
    if split not in {"test", "val"} or not available(split):
        raise FileNotFoundError(f"Requested {split} split is missing: {data.get(split)} (source: {source})")
    resolved = run_dir / "data.resolved.yaml"
    resolved.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return resolved, split, names


class Reporter:
    def __init__(self, path: Path):
        self.console = sys.stdout
        self.file = path.open("w", encoding="utf-8", buffering=1)

    def __call__(self, message: str = "") -> None:
        print(message, file=self.console, flush=True)
        print(message, file=self.file, flush=True)


def format_table(headers: list[str], rows: list[list[Any]]) -> str:
    text = [[str(v) for v in row] for row in rows]
    widths = [max([len(h)] + [len(row[i]) for row in text]) for i, h in enumerate(headers)]
    def line(row):
        return "  ".join(str(v).ljust(w) for v, w in zip(row, widths))
    return "\n".join([line(headers), line(["-" * w for w in widths])] + [line(row) for row in text])


def number(value: Any, digits: int = 2, percent: bool = False) -> str:
    return "--" if value is None else f"{float(value) * (100 if percent else 1):.{digits}f}"


@contextlib.contextmanager
def route_logger(logger: logging.Logger, stream):
    old_handlers, old_level, old_propagate = logger.handlers[:], logger.level, logger.propagate
    handler = logging.StreamHandler(stream)
    logger.handlers, logger.propagate = [handler], False
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.handlers, logger.propagate = old_handlers, old_propagate
        logger.setLevel(old_level)
        handler.close()


def import_object(value: str):
    module, name = value.split(":", 1)
    return getattr(importlib.import_module(module), name)


# ---------------- ULTRALYTICS ADAPTER: localize future API adjustments here ----------------
def remap_dataset_labels(dataset, class_mapping: dict) -> dict:
    """Align raw GT IDs to model IDs by declared names, before DataLoader workers start."""
    labels = getattr(dataset, "labels", None)
    if not isinstance(labels, list):
        raise ValueError("This adapter requires a dataset.labels list to audit and map class IDs")
    source_names, target_names = class_mapping["label_names"], class_mapping["model_names"]
    mapping = np.asarray(class_mapping["label_to_model_id"], dtype=np.int64)
    source_counts = np.zeros(len(source_names), dtype=np.int64)
    target_counts = np.zeros(len(target_names), dtype=np.int64)
    for label in labels:
        raw = np.asarray(label["cls"])
        if not np.isfinite(raw).all() or np.any(raw != np.floor(raw)):
            raise ValueError(f"Invalid label class ID: {label.get('im_file')}")
        ids = raw.astype(np.int64)
        if np.any((ids < 0) | (ids >= len(source_names))):
            raise ValueError(f"Raw label IDs {np.unique(ids).tolist()} do not match source names {source_names}: "
                             f"{label.get('im_file')}. Check the dataset YAML/labels; do not guess a mapping.")
        mapped = mapping[ids]
        source_counts += np.bincount(ids.reshape(-1), minlength=len(source_names))
        target_counts += np.bincount(mapped.reshape(-1), minlength=len(target_names))
        label["cls"] = mapped.astype(raw.dtype, copy=True)
    return {**class_mapping, "source_gt_counts": source_counts.tolist(), "model_gt_counts": target_counts.tolist()}


def collect_ultralytics_records(preds, batch, task: str) -> list[dict]:
    """Accept both recent dict predictions and older N x (6/7) tensor predictions."""
    import torch
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import batch_probiou, box_iou

    result = []
    height, width = batch["img"].shape[2:]
    for i, pred in enumerate(preds):
        if isinstance(pred, dict):
            boxes, cls, confidence = pred["bboxes"], pred["cls"], pred["conf"]
            if task == "obb" and boxes.shape[-1] == 4:
                boxes = torch.cat((boxes, pred["extra"][:, :1]), dim=-1)
        else:
            cls, confidence = pred[:, 5], pred[:, 4]
            boxes = torch.cat((pred[:, :4], pred[:, -1:]), -1) if task == "obb" else pred[:, :4]
        boxes = boxes.detach().float().clone()
        mask = batch["batch_idx"] == i
        gt_cls = batch["cls"][mask].reshape(-1)
        gt = batch["bboxes"][mask].detach().float().clone()
        scale = gt.new_tensor([width, height, width, height])
        signed_obb = None
        if task == "obb":
            if gt.shape[-1] != 5 or boxes.shape[-1] != 5:
                raise ValueError("OBB adapter expects xywhr boxes")
            gt[:, :4] *= scale
            if (gt[:, 2:4] < 0).any():
                raise ValueError(f"Negative GT OBB side length: {batch['im_file'][i]}")
            # DFL-free heads can emit signed side lengths. Native ProbIoU uses
            # w**2 and h**2, so keep the original boxes/matching and use their
            # geometric side magnitudes ONLY for area grouping.
            signed_mask = (boxes[:, 2:4] < 0).any(dim=1)
            signed_count = int(signed_mask.sum().item())
            if signed_count:
                signed_obb = {
                    "count": signed_count,
                    "examples": [
                        {"xywhr": b, "confidence": c, "class_id": int(k)}
                        for b, c, k in zip(boxes[signed_mask][:3].cpu().tolist(),
                                           confidence[signed_mask][:3].detach().cpu().tolist(),
                                           cls[signed_mask][:3].detach().cpu().tolist())
                    ],
                }
        else:
            gt = ops.xywh2xyxy(gt) * scale
        overlap = (batch_probiou if task == "obb" else box_iou)(gt, boxes)
        shape, ratio = batch["ori_shape"][i], batch["ratio_pad"][i]
        # Respect both resize axes: RT-DETR stretches; YOLO rounding can also give
        # slightly different h/w gains. A single uniform gain misclassifies areas.
        if ratio is None:
            gain_h = gain_w = min(height / shape[0], width / shape[1])
            pad_x, pad_y = (width - shape[1] * gain_w) / 2, (height - shape[0] * gain_h) / 2
        elif isinstance(ratio[0], (tuple, list)):
            (gain_h, gain_w), (pad_x, pad_y) = ratio
        else:  # older stretch-resize datasets store only (height_gain, width_gain)
            gain_h, gain_w = ratio
            pad_x = pad_y = 0
        if not all(np.isfinite(g) and g > 0 for g in (gain_h, gain_w)):
            raise ValueError(f"Invalid resize gains: {batch['im_file'][i]}: {gain_h}, {gain_w}")
        def area(b):
            if task == "obb":
                return (b[:, 2].abs() * b[:, 3].abs() / (gain_h * gain_w)).cpu().numpy()
            b = b.clone()
            b[:, [0, 2]] = ((b[:, [0, 2]] - pad_x) / gain_w).clamp(0, shape[1])
            b[:, [1, 3]] = ((b[:, [1, 3]] - pad_y) / gain_h).clamp(0, shape[0])
            wh = (b[:, 2:4] - b[:, :2]).clamp_min(0)
            return (wh[:, 0] * wh[:, 1]).cpu().numpy()
        result.append({
            "image_id": str(batch["im_file"][i]), "gt_cls": gt_cls.cpu().numpy(),
            "pred_cls": cls.detach().cpu().numpy(), "confidence": confidence.detach().cpu().numpy(),
            "gt_area": area(gt), "pred_area": area(boxes), "overlap": overlap.cpu().numpy(),
        })
        if signed_obb:
            result[-1]["signed_obb_predictions"] = signed_obb
    return result


def profile_model(model, settings: dict) -> dict:
    """Full-resolution FP32 forward; never extrapolate attention FLOPs from a tiny image."""
    import torch
    result = {"parameters": sum(p.numel() for p in model.parameters()), "gflops": None,
              "flops_method": settings["flops_method"], "flops_error": None,
              "parameter_scope": "checkpoint model before inference fusion",
              "input_shape_bchw": [1, 3, int(settings["imgsz"]), int(settings["imgsz"])],
              "flops_scope": "supported forward operators, 1 MAC = 2 FLOPs; not every elementwise/custom op"}
    if settings["flops_method"] == "none":
        result["flops_error"] = "Profiling disabled in configuration"
        return result
    try:
        cloned = copy.deepcopy(model).float().eval()
        device = next(cloned.parameters()).device
        inputs = torch.zeros(1, 3, int(settings["imgsz"]), int(settings["imgsz"]), device=device)
        # inference_mode bypasses FlopCounterMode dispatch on some PyTorch releases.
        with torch.no_grad():
            cloned(inputs)  # materialize caches outside the counter
            if settings["flops_method"] == "torch":
                from torch.utils.flop_counter import FlopCounterMode
                with FlopCounterMode(display=False) as counter:
                    cloned(inputs)
                flops = counter.get_total_flops()
            elif settings["flops_method"] == "thop":
                import thop
                macs, _ = thop.profile(cloned, inputs=(inputs,), verbose=False)
                flops = 2 * macs
            else:
                raise ValueError("flops_method must be torch, thop or none")
        if not np.isfinite(flops) or flops <= 0:
            raise ValueError(f"Profiler returned {flops}")
        result["gflops"] = float(flops / 1e9)
    except Exception as exc:
        result["flops_error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return result


def training_metadata(yolo, spec: dict) -> dict:
    ckpt = getattr(yolo, "ckpt", None) or {}
    args = ckpt.get("train_args") or {}
    result = {"epochs": None, "epochs_source": None, "planned_epochs": args.get("epochs"),
              "training_data": args.get("data"), "checkpoint_names": names_list(yolo.names),
              "best_epoch": int(ckpt["epoch"]) + 1 if ckpt.get("epoch", -1) >= 0 else None}
    if spec.get("epochs") is not None:
        result.update(epochs=int(spec["epochs"]), epochs_source="configuration override")
    else:
        path = Path(spec["weights"]).parent.parent / "results.csv"
        if path.is_file():
            with path.open(encoding="utf-8-sig", newline="") as f:
                rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
            epochs = [int(float(row["epoch"])) for row in rows if row.get("epoch")]
            if epochs:
                result.update(epochs=max(epochs), epochs_source=str(path))
    return result


def run_ultralytics(spec: dict, settings: dict, run_dir: Path, progress, raw_log) -> dict:
    """Return normalized records + metadata. No metric math belongs in this adapter."""
    import torch
    import ultralytics
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.utils import LOGGER
    from ultralytics.utils.torch_utils import select_device

    with route_logger(LOGGER, raw_log):
        for module in spec.get("imports", []):
            importlib.import_module(module)  # register classes used in checkpoint pickles
        loader = import_object(spec.get("loader", "ultralytics:YOLO"))
        model = loader(spec["weights"], task=settings["task"])
        actual_names = names_list(model.names)
        if actual_names != spec["names"]:
            raise ValueError(f"Class order mismatch: checkpoint={actual_names}; dataset={spec['names']}")
        class_mapping = spec["class_mapping"]
        for name in spec.get("expected_empty_classes", []):
            if name not in actual_names:
                raise ValueError(f"Expected empty class {name!r} is absent from checkpoint names")
        changed = [(i, j) for i, j in enumerate(class_mapping["label_to_model_id"]) if i != j]
        for i, j in changed:
            progress(f"Label mapping: {i} ({class_mapping['label_names'][i]}) -> model class {j} ({actual_names[j]})")
        actual_task = getattr(model, "task", None)
        if actual_task != settings["task"]:
            raise ValueError(f"Checkpoint task={actual_task}, evaluation task={settings['task']}")
        actual_backbone = getattr(model.model, "backbone_name", None)
        if spec.get("expected_backbone") and actual_backbone != spec["expected_backbone"]:
            raise ValueError(f"Expected backbone {spec['expected_backbone']}, found {actual_backbone}")
        device = select_device(str(settings["device"]))
        model.model.to(device).float().eval()
        progress("Profiling parameters and FLOPs...")
        benchmark = settings.get("speed_benchmark", {"enabled": False})
        timing_settings = {k: v for k, v in settings.items() if k != "speed_benchmark"}
        if benchmark.get("enabled", False):
            timing_settings.update({k: v for k, v in benchmark.items() if k != "enabled"})
        complexity = profile_model(model.model, timing_settings)
        metadata = training_metadata(model, spec)
        metadata.update(backbone=spec.get("backbone") or actual_backbone or "unspecified",
                        checkpoint_backbone=actual_backbone, model_class=type(model.model).__name__)
        records = []
        # Use the model's own validator (e.g. RT-DETR) and attach only record collection.
        base_validator = model._smart_load("validator")

        def validate_pass(options: dict, collect: bool):
            phase = "Accuracy" if collect else "Speed"
            observed_shapes, observed_dtypes, image_paths = set(), set(), []
            label_audit = {}

            class RecordingValidator(base_validator):
                def build_dataset(self, *args, **kwargs):
                    dataset = super().build_dataset(*args, **kwargs)
                    label_audit.update(remap_dataset_labels(dataset, class_mapping))
                    for name in spec.get("expected_empty_classes", []):
                        count = label_audit["model_gt_counts"][actual_names.index(name)]
                        if count:
                            raise ValueError(f"Expected empty class {name} has {count} GT. Check raw label IDs. "
                                             "For restored CMDPC labels, run restore_cmdpc_label_ids.py first: Drill_tail must use ID 4.")
                    if collect:
                        progress("GT audit: " + ", ".join(f"{name}={n}" for name, n in
                                 zip(actual_names, label_audit["model_gt_counts"])))
                    return dataset

                def update_metrics(self, preds, batch):
                    shape = tuple(int(x) for x in batch["img"].shape)
                    dtype = str(batch["img"].dtype)
                    expected_dtype = "torch.float16" if options["half"] else "torch.float32"
                    valid = (len(shape) == 4 and 0 < shape[0] <= int(options["batch"])
                             and shape[1] == 3 and dtype == expected_dtype)
                    if not options["rect"]:
                        valid = valid and shape[2:] == (int(options["imgsz"]), int(options["imgsz"]))
                    if not valid:
                        raise ValueError(f"{phase} actual input {shape}/{dtype} differs from batch={options['batch']}, "
                                         f"imgsz={options['imgsz']}, rect={options['rect']}, {expected_dtype}")
                    observed_shapes.add(shape)
                    observed_dtypes.add(dtype)
                    image_paths.extend(str(p) for p in batch["im_file"])
                    if collect:
                        records.extend(collect_ultralytics_records(preds, batch, settings["task"]))
                        return super().update_metrics(preds, batch)
                    self.seen += shape[0]

                def init_metrics(self, backend_model):
                    super().init_metrics(backend_model)
                    if not label_audit:
                        raise ValueError("Validator bypassed build_dataset(); class mapping was not applied. Adjust the adapter.")
                    # Keep native metrics in the YOLO protocol even for COCO-named data.
                    self.args.save_json = False

                def get_stats(self):
                    return super().get_stats() if collect else {}

                def print_results(self):
                    if collect:
                        return super().print_results()

            last_progress = [0.0]
            def show_progress(validator):
                now = time.monotonic()
                total = len(validator.dataloader.dataset)
                if now - last_progress[0] >= 20 or len(image_paths) == total:
                    progress(f"{phase}: {len(image_paths)}/{total} images")
                    last_progress[0] = now

            kwargs = {k: options[k] for k in ("imgsz", "batch", "device", "workers", "conf", "iou", "max_det", "rect")}
            kwargs.update(data=spec["resolved_data"], split=spec["split"], task=settings["task"],
                          project=str(run_dir), name="validator_accuracy" if collect else "validator_speed",
                          exist_ok=True, verbose=False, plots=False, save_json=False, save_txt=False,
                          augment=False, single_cls=False, agnostic_nms=False, classes=None, cache=False)
            for key, value in (("compile", False), ("fraction", 1.0)):
                if key in DEFAULT_CFG_DICT:
                    kwargs[key] = value
            # New releases use quantize; older releases use half.
            kwargs["quantize" if "quantize" in DEFAULT_CFG_DICT else "half"] = (
                (16 if options["half"] else 32) if "quantize" in DEFAULT_CFG_DICT else bool(options["half"])
            )
            model.add_callback("on_val_batch_end", show_progress)
            try:
                native = model.val(validator=RecordingValidator, **kwargs)
            finally:
                model.callbacks["on_val_batch_end"].remove(show_progress)
            runtime = {"input_shapes_bchw": [list(s) for s in sorted(observed_shapes)],
                       "input_dtypes": sorted(observed_dtypes), "images": len(image_paths),
                       "class_mapping": label_audit,
                       "image_set_sha256": hashlib.sha256("\n".join(sorted(image_paths)).encode()).hexdigest(),
                       "validator_class": base_validator.__module__ + "." + base_validator.__name__,
                       "end2end_head": bool(getattr(model.model, "end2end", False))}
            return native, runtime

        native, runtime = validate_pass(settings, collect=True)
        if settings["task"] == "obb":
            signed_records = [r for r in records if r.get("signed_obb_predictions")]
            geometry_audit = {
                "area_policy": OBB_AREA_POLICY,
                "prediction_area": "abs(w) * abs(h) / (gain_h * gain_w)",
                "native_boxes_and_overlap_unchanged": True,
                "images_with_signed_predictions": len(signed_records),
                "signed_predictions": sum(r["signed_obb_predictions"]["count"] for r in signed_records),
                "examples": [{"image_id": r["image_id"], **r["signed_obb_predictions"]}
                             for r in signed_records[:10]],
            }
            runtime["obb_geometry"] = geometry_audit
            dump_json(run_dir / "obb_geometry_audit.json", geometry_audit)
            if signed_records:
                progress(f"OBB area audit: {geometry_audit['signed_predictions']} signed-side predictions in "
                         f"{len(signed_records)} images; area=abs(w)*abs(h), native AP unchanged.")
        observed_counts = np.bincount(np.concatenate([r["gt_cls"] for r in records]).astype(int), minlength=len(actual_names))
        if observed_counts.tolist() != runtime["class_mapping"]["model_gt_counts"]:
            raise ValueError("Collected GT counts differ from mapped dataset counts; check worker/dataset handling")
        dump_json(run_dir / "class_mapping.json", runtime["class_mapping"])
        accuracy_speed = {k: float(v) for k, v in native.speed.items()}
        native_aps = np.asarray(native.box.all_ap)
        native_summary = {"AP": float(native.box.map), "AP50": float(native.box.map50),
                          "AP75": float(native_aps[:, 5].mean()) if len(native_aps) else 0.0}
        native_per_class = {int(c): float(ap[0]) for c, ap in zip(native.box.ap_class_index, native.box.all_ap)}
        if benchmark.get("enabled", False):
            timed, speed_runtime = validate_pass(timing_settings, collect=False)
            if (speed_runtime["images"] != runtime["images"]
                    or speed_runtime["image_set_sha256"] != runtime["image_set_sha256"]
                    or speed_runtime["class_mapping"] != runtime["class_mapping"]):
                raise ValueError("Accuracy and speed passes evaluated different images")
            speed = {k: float(v) for k, v in timed.speed.items()}
            speed_source = "separate_native_validation"
        else:
            speed, speed_runtime, speed_source = accuracy_speed, runtime, "accuracy_validation"
        environment = {"python": platform.python_version(), "platform": platform.platform(),
                       "torch": str(torch.__version__), "ultralytics": str(ultralytics.__version__),
                       "ultralytics_source": str(Path(ultralytics.__file__).resolve()),
                       "numpy": np.__version__, "cuda": torch.version.cuda, "device": str(device),
                       "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                       "cudnn": torch.backends.cudnn.version(), "cpu_threads": torch.get_num_threads()}
        return {"records": records, "speed_ms": speed, "speed_source": speed_source,
                "speed_settings": timing_settings, "speed_runtime": speed_runtime,
                "accuracy_speed_ms": accuracy_speed,
                "metadata": metadata, "complexity": complexity, "environment": environment,
                "runtime": runtime,
                "native_metrics": native_summary, "native_per_class_ap50": native_per_class,
                "overlap_kind": "ProbIoU" if settings["task"] == "obb" else "box_IoU"}


# ---------------- SHARED ORCHESTRATION / EXPORT ----------------
def metric_protocol(settings: dict) -> dict:
    source = "\n".join(inspect.getsource(f) for f in (match_iou_first, integrate_ap, validate_records, evaluate_records))
    kernel_hash = hashlib.sha256(source.encode()).hexdigest()
    protocol = {"version": METRIC_VERSION, "kernel_sha256": kernel_hash,
                "iou_thresholds": [round(float(x), 2) for x in IOU_THRESHOLDS],
                "area_ranges_px2_inclusive": {k: [lo, hi if np.isfinite(hi) else None] for k, (lo, hi) in AREA_RANGES.items()},
                "area_coordinate_system": "original image; OBB area = abs(rotated width) * abs(rotated height)",
                "obb_area_policy": OBB_AREA_POLICY,
                "matching": "Ultralytics IoU-first greedy, eligible GT before ignored GT",
                "ap_integration": "101-point interpolation followed by trapezoidal integration",
                "empty_class_or_size": "N/A, excluded from macro average",
                "fps_inference": "1000 / native validator inference_ms_per_image",
                "fps_pipeline": "1000 / (preprocess_ms + inference_ms + postprocess_ms)",
                "timing_excludes": "disk decoding/loading, metric accumulation, plotting, file output",
                "settings": settings}
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol


def export_summaries(root: Path, results: list[dict], failures: list[dict], protocol: dict) -> None:
    headers = ["Dataset", "Split", "Method", "Backbone", "Epochs", "Params(M)", "GFLOPs", "FPS",
               "mAP50(%)", "mAP75(%)", "mAP50-95(%)", "APS(%)", "APM(%)", "APL(%)"]
    rows, class_rows = [], []
    for r in results:
        m = r["metrics"]
        rows.append([r["dataset"], r["split"], r["method"], r["metadata"]["backbone"],
                     r["metadata"].get("epochs") or "--", number(r["complexity"]["parameters"] / 1e6, 3),
                     number(r["complexity"]["gflops"], 3), number(r["fps_table"]),
                     *[number(m["all"][k], percent=True) for k in ("AP50", "AP75", "AP")],
                     *[number(m[k]["AP"], percent=True) for k in ("S", "M", "L")]])
        for c in m["all"]["per_class"]:
            class_rows.append([r["dataset"], r["split"], r["method"], c["class_id"], c["class_name"],
                               c["instances"], number(c["AP50"], percent=True)])
    class_headers = ["Dataset", "Split", "Method", "ClassID", "Class", "Instances", "AP50(%)"]
    for filename, columns, data in (("summary.csv", headers, rows), ("per_class_ap50.csv", class_headers, class_rows)):
        with (root / filename).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(data)
    def markdown(columns, data):
        escape = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
        return "\n".join("| " + " | ".join(escape(v) for v in row) + " |" for row in [columns, ["---"] * len(columns), *data])
    md = ["# Paper evaluation", "", "All AP values below are percentages: 80.30% = 0.8030. mAP50-95 averages IoU 0.50:0.05:0.95.",
          f"FPS column: **{protocol['settings']['fps_column']}**. Protocol: `{protocol['version']}`.",
          f"Accuracy: imgsz={protocol['settings']['imgsz']}, batch={protocol['settings']['batch']}, rect={protocol['settings']['rect']}.",
          f"Separate speed benchmark: `{json.dumps(protocol['settings'].get('speed_benchmark', {}), sort_keys=True)}`.",
          "Size AP uses COCO area cutoffs with the fixed YOLO matching/AP extension; it is not native COCOeval.",
          "No averaging across different datasets. `--` means unavailable, not zero.", "", markdown(headers, rows),
          "", "## Per-class AP50", "", markdown(class_headers, class_rows)]
    if failures:
        md += ["", "## Failed runs", "", *[f"- {x['id']}: {x['error']}" for x in failures]]
    (root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    def tex(value):
        escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                   "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
        return "".join(escapes.get(c, c) for c in str(value))
    latex = ["% Requires \\usepackage{booktabs}. AP values are percentages.",
             "% FPS: " + protocol["settings"]["fps_column"] + "; size AP: YOLO area extension, COCO cutoffs."]
    for dataset, split in dict.fromkeys((r[0], r[1]) for r in rows):
        selected = [r[2:] for r in rows if r[0:2] == [dataset, split]]
        latex += [r"\begin{table*}[t]", r"\centering", r"\small",
                  r"\caption{Comparison on " + tex(dataset) + " (" + tex(split) + ").}",
                  r"\begin{tabular}{llrrrrrrrrrr}", r"\toprule",
                  r"Method & Backbone & Epochs & Params (M) & GFLOPs & FPS & mAP$_{50}$ & mAP$_{75}$ & mAP$_{50:95}$ & AP$_S$ & AP$_M$ & AP$_L$ \\",
                  r"\midrule", *[" & ".join(tex(x) for x in row) + r" \\" for row in selected],
                  r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (root / "summary.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    dump_json(root / "summary.json", {"protocol": protocol, "results": results, "failures": failures})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "ex_config" / "paper_evaluation.yaml")
    parser.add_argument("--only", nargs="+", help="Experiment IDs, e.g. OCM CMDPC")
    parser.add_argument("--device", help="Override the common device, e.g. 4 or cpu")
    parser.add_argument("--split", choices=("val", "test", "auto"), help="Override the split for selected experiments")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    settings = config["evaluation"].copy()
    if args.device is not None:
        settings["device"] = args.device
    settings["device"] = str(settings["device"])
    benchmark = config.get("speed_benchmark", {"enabled": False}).copy()
    allowed_speed_keys = {"enabled", "imgsz", "batch", "rect", "half"}
    if set(benchmark) - allowed_speed_keys:
        raise ValueError(f"Unknown speed_benchmark options: {set(benchmark) - allowed_speed_keys}")
    if benchmark.get("enabled", False):
        benchmark = {"imgsz": settings["imgsz"], "batch": 1, "rect": False,
                     "half": settings["half"], **benchmark}
        if int(benchmark["imgsz"]) <= 0 or int(benchmark["batch"]) != 1 or bool(benchmark["rect"]):
            raise ValueError("The separate speed benchmark requires positive imgsz, batch=1 and rect=false")
    settings["speed_benchmark"] = benchmark
    if settings["task"] not in {"obb", "detect"} or settings["fps_column"] not in {"inference", "pipeline"}:
        raise ValueError("Invalid task or fps_column")
    if settings["flops_method"] not in {"torch", "thop", "none"}:
        raise ValueError("flops_method must be torch, thop or none")
    if int(settings["imgsz"]) <= 0 or int(settings["max_det"]) <= 0:
        raise ValueError("imgsz and max_det must be positive")
    if not 0 <= float(settings["conf"]) <= 1 or not 0 <= float(settings["iou"]) <= 1:
        raise ValueError("conf and NMS iou must be within [0, 1]")
    if int(settings["batch"]) <= 0:
        raise ValueError("Accuracy batch must be a positive integer")
    if "," in settings["device"] or settings["device"].startswith("["):
        raise ValueError("Use one device for comparable FPS; distributed evaluation is not supported")
    experiments = [x for x in config["experiments"] if x.get("enabled", True)]
    ids = [str(x["id"]) for x in experiments]
    if len(set(ids)) != len(ids) or any(not x or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in x) for x in ids):
        raise ValueError("Experiment IDs must be unique and contain only letters, digits, '_' or '-'")
    if args.only:
        if set(args.only) - set(ids):
            raise ValueError(f"Unknown experiment IDs: {set(args.only) - set(ids)}")
        experiments = [x for x in experiments if x["id"] in args.only]
    if not experiments:
        raise ValueError("No enabled experiments")
    for value in reversed(config.get("python_paths", [])):
        sys.path.insert(0, str(resolve_path(value, config_path.parent)))
    output = args.output_dir.resolve() if args.output_dir else resolve_path(config["output_dir"], config_path.parent)
    root = output / datetime.now().strftime("evaluation_%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True)
    reporter = Reporter(root / "run.log")
    protocol = metric_protocol(settings)
    dump_json(root / "protocol.json", protocol)
    snapshot = copy.deepcopy(config)
    snapshot.update(evaluation={k: v for k, v in settings.items() if k != "speed_benchmark"},
                    speed_benchmark=benchmark, selected_experiments=[x["id"] for x in experiments],
                    split_override=args.split)
    (root / "config.snapshot.yaml").write_text(yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8")
    reporter("=" * 90)
    reporter("PAPER EVALUATION | Table values: % (80.30% = 0.8030) | mAP50 / mAP75 / mAP50-95")
    reporter(f"Accuracy: imgsz={settings['imgsz']} | batch={settings['batch']} | rect={settings['rect']} | "
             f"precision={'FP16' if settings['half'] else 'FP32'} | device={settings['device']}")
    if benchmark.get("enabled", False):
        reporter(f"Speed: separate pass | {benchmark['imgsz']}x{benchmark['imgsz']} | batch=1 | rect=False | "
                 f"precision={'FP16' if benchmark['half'] else 'FP32'}")
    else:
        reporter("Speed: measured during accuracy validation, with the accuracy input settings above")
    reporter(f"FPS column: {settings['fps_column']} | task={settings['task']} | FLOPs={settings['flops_method']}")
    reporter(f"Output: {root}")
    results, failures = [], []
    for index, original_spec in enumerate(experiments, 1):
        spec = copy.deepcopy(original_spec)
        if args.split is not None:
            spec["split"] = args.split
        run_dir = root / spec["id"]
        run_dir.mkdir()
        raw_path = run_dir / "raw.log"
        reporter("")
        reporter(f"[{index}/{len(experiments)}] {spec['id']} | {spec['method']}")
        progress = lambda message: reporter(f"  {message}")
        try:
            weights = resolve_path(spec["weights"], config_path.parent)
            if not weights.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {weights}")
            spec["weights"] = str(weights)
            data, split, names = prepare_dataset(spec, config_path.parent, run_dir)
            spec.update(resolved_data=str(data), split=split, names=names)
            progress(f"Dataset={spec['dataset']} | split={split} | label_classes={len(spec['class_mapping']['label_names'])} | model_classes={len(names)}")
            with raw_path.open("w", encoding="utf-8", buffering=1) as raw:
                with contextlib.redirect_stdout(raw), contextlib.redirect_stderr(raw):
                    try:
                        adapter = spec.get("adapter", "ultralytics")
                        function = run_ultralytics if adapter == "ultralytics" else import_object(adapter)
                        outcome = function(spec, settings, run_dir, progress, raw)
                        metrics = evaluate_records(outcome["records"], names, int(settings["max_det"]))
                    except Exception:
                        traceback.print_exc()
                        raise
            expected_overlap = "ProbIoU" if settings["task"] == "obb" else "box_IoU"
            if outcome["overlap_kind"] != expected_overlap:
                raise ValueError(f"Adapter overlap={outcome['overlap_kind']}; required={expected_overlap}")
            if benchmark.get("enabled", False):
                if outcome.get("speed_source") not in {"separate_native_validation", "separate_adapter_benchmark"}:
                    raise ValueError("Adapter must implement the separate speed_benchmark and return speed_source/speed_settings")
                if any(outcome.get("speed_settings", {}).get(k) != benchmark[k] for k in ("imgsz", "batch", "rect", "half")):
                    raise ValueError("Adapter speed_settings do not match speed_benchmark")
            speed = outcome["speed_ms"]
            inference_ms = speed["inference"]
            pipeline_ms = sum(speed[k] for k in ("preprocess", "inference", "postprocess"))
            if not all(np.isfinite(speed[k]) and speed[k] >= 0 for k in ("preprocess", "inference", "postprocess")) or inference_ms <= 0:
                raise ValueError(f"Invalid per-image speed from adapter: {speed}")
            fps = {"inference": 1000 / inference_ms, "pipeline": 1000 / pipeline_ms}
            native = outcome.get("native_metrics")
            delta = max(abs(native[k] - metrics["all"][k]) for k in ("AP", "AP50", "AP75")) if native and metrics["all"]["AP"] is not None else None
            notes = []
            if delta is not None and delta > 1e-4:
                notes.append(f"Native YOLO / fixed kernel AP differs by {delta:.6g}; inspect version/API differences before publication.")
            if outcome["complexity"].get("flops_error"):
                notes.append("GFLOPs unavailable: " + outcome["complexity"]["flops_error"])
            if outcome["metadata"].get("epochs") is None:
                notes.append("Actual training epochs unavailable; set experiments[].epochs if needed.")
            manifest = "\n".join(str(r["image_id"]) for r in outcome["records"])
            (run_dir / "evaluated_images.txt").write_text(manifest + "\n", encoding="utf-8")
            result = {"id": spec["id"], "dataset": spec["dataset"], "method": spec["method"], "split": split,
                      "weights": str(weights), "weights_sha256": file_hash(weights), "data": str(data),
                      "images": len(outcome["records"]), "image_list_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
                      "metadata": outcome["metadata"], "complexity": outcome["complexity"],
                      "class_mapping": spec["class_mapping"],
                      "environment": outcome["environment"], "runtime": outcome.get("runtime"),
                      "accuracy_settings": {k: v for k, v in settings.items() if k != "speed_benchmark"},
                      "accuracy_speed_ms": outcome.get("accuracy_speed_ms"),
                      "speed_source": outcome.get("speed_source", "adapter"),
                      "speed_settings": outcome.get("speed_settings"), "speed_runtime": outcome.get("speed_runtime"),
                      "metrics": metrics, "native_metrics": native,
                      "overall_metrics_0_1": {"mAP50": metrics["all"]["AP50"],
                                              "mAP75": metrics["all"]["AP75"],
                                              "mAP50-95": metrics["all"]["AP"]},
                      "native_per_class_ap50": outcome.get("native_per_class_ap50"), "native_max_ap_difference": delta,
                      "speed_ms": speed, "fps": fps, "fps_table": fps[settings["fps_column"]],
                      "protocol_sha256": protocol["protocol_sha256"], "notes": notes}
            dump_json(run_dir / "metrics.json", result)
            results.append(result)
            progress(f"Images={result['images']} | Backbone={result['metadata']['backbone']}")
            for label, runtime in (("Accuracy", result["runtime"]), ("Speed", result["speed_runtime"])):
                if runtime:
                    progress(f"{label} input BCHW={runtime['input_shapes_bchw']} | dtype={runtime['input_dtypes']}")
            progress(f"Epochs={result['metadata'].get('epochs') or '--'} | Params={number(result['complexity']['parameters'] / 1e6, 3)} M | GFLOPs={number(result['complexity']['gflops'], 3)}")
            progress("Metrics (0-1): " + " | ".join(f"{name}={number(value, 6)}" for name, value in result["overall_metrics_0_1"].items()))
            reporter(format_table(["mAP50(%)", "mAP75(%)", "mAP50-95(%)", "APS(%)", "APM(%)", "APL(%)", "FPS(infer)", "FPS(pipe)"], [[
                *[number(metrics["all"][k], percent=True) for k in ("AP50", "AP75", "AP")],
                *[number(metrics[k]["AP"], percent=True) for k in ("S", "M", "L")],
                number(fps["inference"]), number(fps["pipeline"])]]))
            reporter(format_table(["Class", "GT", "AP50(%)"], [[c["class_name"], c["instances"], number(c["AP50"], percent=True)] for c in metrics["all"]["per_class"]]))
            progress("GT by area: " + ", ".join(f"{k}={metrics[k]['instances']}" for k in ("S", "M", "L")))
            for note in notes:
                progress("NOTE: " + note)
            del outcome
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            failures.append({"id": spec["id"], "error": error, "log": str(raw_path)})
            progress(f"FAILED: {error}")
            progress(f"Details: {run_dir / 'error.txt'}; {raw_path}")
        export_summaries(root, results, failures, protocol)
    reporter("")
    reporter("=" * 90)
    reporter(f"Finished: {len(results)} succeeded, {len(failures)} failed")
    if results:
        reporter(format_table(
            ["Dataset", "Split", "Method", "Params(M)", "GFLOPs", "FPS", "mAP50(%)", "mAP75(%)", "mAP50-95(%)", "APS(%)", "APM(%)", "APL(%)"],
            [[r["dataset"], r["split"], r["method"], number(r["complexity"]["parameters"] / 1e6, 3),
              number(r["complexity"]["gflops"], 3), number(r["fps_table"]),
              *[number(r["metrics"]["all"][k], percent=True) for k in ("AP50", "AP75", "AP")],
              *[number(r["metrics"][k]["AP"], percent=True) for k in ("S", "M", "L")]] for r in results]))
    reporter(f"Tables: {root / 'summary.csv'} | summary.md | summary.tex")
    reporter(f"Per-class AP50: {root / 'per_class_ap50.csv'}")
    reporter.file.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
