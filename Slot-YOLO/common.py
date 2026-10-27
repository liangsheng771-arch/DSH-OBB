"""Configuration helpers shared by Slot-YOLO scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ig_config import validate_ig_config, validate_p5_config
from module_version import bind_config_version
from backbone_config import backbone_model_path, selected_backbone
from rtdetrv2_hye_config import (
    ccff_variant_name,
    normalize_rtdetrv2_hye_yolo_config,
    validate_rtdetrv2_hye_yolo_selection,
)


ROOT = Path(__file__).resolve().parent
TASK_ALIASES = {
    "detect": "detect",
    "hbb": "detect",
    "obb": "obb",
}
MODEL_TYPE_ALIASES = {
    "slot": "slot_yolo",
    "slot_yolo": "slot_yolo",
    "slot-yolo": "slot_yolo",
    "baseline": "baseline_yolo",
    "baseline_yolo": "baseline_yolo",
    "baseline-yolo": "baseline_yolo",
}


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _looks_like_path(value: Any) -> bool:
    if not isinstance(value, (str, Path)):
        return False
    text = str(value)
    return Path(text).is_absolute() or "/" in text or "\\" in text or text.startswith(".")


def _select_variant(
    value: Any,
    *,
    task: str,
    model_type: str,
    key: str,
    allow_none: bool = False,
) -> Any:
    """Select an optional model-family and task-specific configuration value."""
    selected = value
    if isinstance(selected, dict):
        family_keys = set(MODEL_TYPE_ALIASES.values())
        if model_type in selected:
            selected = selected[model_type]
        elif set(selected).intersection(family_keys):
            raise ValueError(f"{key} has no '{model_type}' entry")

    if isinstance(selected, dict):
        task_keys = (task, "hbb") if task == "detect" else (task,)
        selected_key = next((name for name in task_keys if name in selected), None)
        if selected_key is None:
            raise ValueError(f"{key} has no '{task}' entry")
        selected = selected[selected_key]

    if isinstance(selected, dict):
        raise TypeError(
            f"{key} must resolve to one scalar value after selecting "
            f"model_type={model_type!r}, task={task!r}"
        )
    if selected is None and not allow_none:
        raise ValueError(
            f"{key} is null for model_type={model_type!r}, task={task!r}"
        )
    return selected


def load_config(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load the experiment YAML and resolve every filesystem setting."""
    path = Path(config_path).expanduser().resolve()
    # Resolve a bare experiment name against the retained ex_config directory.
    requested = Path(config_path)
    if not path.is_file() and requested.parent == Path("."):
        name = requested.name if requested.suffix.lower() == ".yaml" else requested.name + ".yaml"
        path = ROOT / "ex_config" / name
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Experiment config must be a YAML mapping: {path}")

    required = ("task", "data", "train")
    missing = [key for key in required if key not in config]
    if "model" not in config and "models" not in config and "backbone" not in config:
        missing.append("model or models")
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    task_name = str(config["task"]).strip().lower()
    if task_name not in TASK_ALIASES:
        raise ValueError("task must be detect/hbb or obb")
    task = TASK_ALIASES[task_name]
    model_type_name = str(config.get("model_type", "slot_yolo")).strip().lower()
    if model_type_name not in MODEL_TYPE_ALIASES:
        raise ValueError("model_type must be slot_yolo or baseline_yolo")
    model_type = MODEL_TYPE_ALIASES[model_type_name]
    config["task"] = task
    config["model_type"] = model_type
    bind_config_version(config, path)
    backbone = selected_backbone(config)
    if backbone is not None:
        config["backbone"] = backbone

    switch_defaults = {
        "use_slotdce_isaf": model_type == "slot_yolo",
        "use_efficient_hybrid_encoder": False,
        "use_rtdetrv2_hye_yolo": False,
        "use_sgdc": False,
        "use_ig_aifi": False,
        "use_ig_fs_ccff": False,
        "use_p5_dual_branch": False,
    }
    for key, default in switch_defaults.items():
        config.setdefault(key, default)
        if not isinstance(config[key], bool):
            raise TypeError(f"{key} must be true or false")
        if config[key] and model_type != "slot_yolo":
            raise ValueError(f"{key}=true requires model_type: slot_yolo")
    if "rtdetrv2_hye_yolo" in config or config["use_rtdetrv2_hye_yolo"]:
        config["rtdetrv2_hye_yolo"] = normalize_rtdetrv2_hye_yolo_config(
            config.get("rtdetrv2_hye_yolo", {})
        )
    validate_rtdetrv2_hye_yolo_selection(config)
    validate_ig_config(config)
    validate_p5_config(config)
    hybrid = config.get("efficient_hybrid_encoder", {})
    if not isinstance(hybrid, dict):
        raise TypeError("efficient_hybrid_encoder must be a YAML mapping")
    unknown_hybrid = set(hybrid)
    if unknown_hybrid:
        raise ValueError(f"Unsupported efficient_hybrid_encoder settings: {sorted(unknown_hybrid)}; P5 output residual has been removed")

    base = path.parent
    model_variants = config.get("models", config.get("model"))
    selected_model = backbone_model_path(config) or _select_variant(
        model_variants,
        task=task,
        model_type=model_type,
        key="models" if "models" in config else "model",
    )
    selected_data = _select_variant(
        config["data"],
        task=task,
        model_type=model_type,
        key="data",
    )
    if not isinstance(selected_model, (str, Path)):
        raise TypeError("The selected model must be a YAML path")
    if not isinstance(selected_data, (str, Path)):
        raise TypeError("The selected data value must be a dataset YAML path")
    config["model"] = str(resolve_path(selected_model, base))
    config["data"] = str(resolve_path(selected_data, base))

    if config["use_slotdce_isaf"] and "enhancer" not in config:
        raise ValueError("enhancer is required when use_slotdce_isaf=true")
    enhancer = config.get("enhancer", {})
    if not isinstance(enhancer, dict):
        raise TypeError("'enhancer' must be a YAML mapping")
    if config["use_slotdce_isaf"]:
        if bool(enhancer.get("pretrained", True)) and not enhancer.get("weights"):
            raise ValueError("enhancer.weights is required when enhancer.pretrained=true")
        if enhancer.get("pretrained") is False and enhancer.get("weights"):
            raise ValueError("enhancer.weights must be null when enhancer.pretrained=false")
    freeze_epochs = int(enhancer.get("freeze_epochs", 0))
    if freeze_epochs < 0:
        raise ValueError("enhancer.freeze_epochs must be zero or positive")
    lr_scale = float(enhancer.get("lr_scale", 0.05))
    if not 0.0 < lr_scale <= 1.0:
        raise ValueError("enhancer.lr_scale must lie in (0, 1]")
    enhancer_decay = float(enhancer.get("weight_decay", 0.0))
    if enhancer_decay < 0.0:
        raise ValueError("enhancer.weight_decay must be zero or positive")
    for key in ("weights", "architecture_config"):
        if enhancer.get(key) and _looks_like_path(enhancer[key]):
            enhancer[key] = str(resolve_path(enhancer[key], base))

    fusion = config.get("fusion", {})
    if not isinstance(fusion, dict):
        raise TypeError("'fusion' must be a YAML mapping")
    for key in ("p5_gate_init",):
        value = float(fusion.get(key, 0.25))
        if not 0.0 < value < 1.0:
            raise ValueError(f"fusion.{key} must lie in (0, 1)")

    feature_modules = config.get("feature_modules")
    if feature_modules is not None:
        if not isinstance(feature_modules, dict):
            raise TypeError("'feature_modules' must be a YAML mapping")
        unknown = set(feature_modules).difference(
            {"guidance", "isaf", "sgdc"}
        )
        if unknown:
            raise ValueError(
                "Unknown feature_modules entries: "
                f"{sorted(unknown)}"
            )
        for key in ("guidance", "isaf"):
            value = feature_modules.get(key)
            if value is not None and not isinstance(value, (str, dict)):
                raise TypeError(
                    f"feature_modules.{key} must be a module name or YAML mapping"
                )
        sgdc = feature_modules.get("sgdc")
        if sgdc is not None and not isinstance(sgdc, dict):
            raise TypeError(
                "feature_modules.sgdc must be a YAML mapping"
            )

    memory = config.get("memory", {})
    if not isinstance(memory, dict):
        raise TypeError("'memory' must be a YAML mapping")
    unknown_memory = set(memory).difference(
        {
            "checkpoint_enhancer",
            "checkpoint_backbone",
            "enhancer_amp",
            "enhancer_amp_dtype",
        }
    )
    if unknown_memory:
        raise ValueError(f"Unknown memory settings: {sorted(unknown_memory)}")
    for key in ("checkpoint_enhancer", "checkpoint_backbone", "enhancer_amp"):
        value = memory.get(key)
        if value is None:
            continue
        if not isinstance(value, bool):
            raise TypeError(f"memory.{key} must be true or false")
    enhancer_amp_dtype = str(memory.get("enhancer_amp_dtype", "bfloat16")).lower()
    if enhancer_amp_dtype not in {"bfloat16", "bf16", "float16", "fp16"}:
        raise ValueError(
            "memory.enhancer_amp_dtype must be bfloat16/bf16 or float16/fp16"
        )

    training = config["train"]
    if not isinstance(training, dict):
        raise TypeError("'train' must be a YAML mapping")
    for section in ("train", "test", "predict"):
        values = config.get(section, {})
        if not isinstance(values, dict):
            raise TypeError(f"'{section}' must be a YAML mapping")
        if "output_dir" in values:
            output_dir = _select_variant(
                values["output_dir"],
                task=task,
                model_type=model_type,
                key=f"{section}.output_dir",
                allow_none=True,
            )
            sgdc_settings = config.get("feature_modules", {}).get("sgdc", {})
            sgdc_name = (
                str(sgdc_settings.get("name", "sgdc")).lower()
                if isinstance(sgdc_settings, dict)
                else str(sgdc_settings or "sgdc").lower()
            )
            sgdc_variant = sgdc_name if config.get("use_sgdc", False) else "no_sgdc"
            resolved_output = str(output_dir).replace("{backbone}", backbone or "yolo26s")
            resolved_output = resolved_output.replace("{sgdc_variant}", sgdc_variant)
            neck_settings = config.get("rtdetrv2_hye_yolo", {})
            neck_variant = "_".join((
                "iglsf" if neck_settings.get("use_ig_lsf_aifi", False) else "native_aifi",
                ccff_variant_name(neck_settings),
            ))
            resolved_output = resolved_output.replace("{neck_variant}", neck_variant)
            values["output_dir"] = str(resolve_path(resolved_output, base)) if output_dir else None

    resume = _select_variant(
        config["train"].get("resume"),
        task=task,
        model_type=model_type,
        key="train.resume",
        allow_none=True,
    )
    config["train"]["resume"] = resume
    if resume not in (None, False, True) and _looks_like_path(resume):
        config["train"]["resume"] = str(resolve_path(resume, base))
    test_weights = _select_variant(
        config.get("test", {}).get("weights"),
        task=task,
        model_type=model_type,
        key="test.weights",
        allow_none=True,
    )
    if "test" in config:
        config["test"]["weights"] = test_weights
    if test_weights and _looks_like_path(test_weights):
        config["test"]["weights"] = str(resolve_path(test_weights, base))
    source = _select_variant(
        config.get("predict", {}).get("source"),
        task=task,
        model_type=model_type,
        key="predict.source",
        allow_none=True,
    )
    if "predict" in config:
        config["predict"]["source"] = source
    if source and _looks_like_path(source):
        config["predict"]["source"] = str(resolve_path(source, base))
    return path, config


def validate_data_yaml(data_yaml: str | Path, required_split: str | None = None) -> dict[str, Any]:
    """Fail before training when the dataset YAML is incomplete."""
    path = Path(data_yaml)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Dataset YAML must be a mapping: {path}")
    required = ["train", "val", "names"]
    if required_split:
        required.append(required_split)
    missing = [key for key in dict.fromkeys(required) if not data.get(key)]
    if missing:
        raise ValueError(f"Dataset YAML is missing: {', '.join(missing)}")
    return data


def ultralytics_args(values: dict[str, Any], *excluded: str) -> dict[str, Any]:
    """Translate the friendly output_dir setting to Ultralytics project/name."""
    ignored = {"output_dir", *excluded}
    args = {key: value for key, value in values.items() if key not in ignored and value is not None}
    if values.get("output_dir"):
        output = Path(values["output_dir"])
        args["project"] = str(output.parent)
        args["name"] = output.name
    return args


def default_weights(config: dict[str, Any]) -> Path:
    output_dir = config.get("train", {}).get("output_dir")
    if not output_dir:
        raise ValueError("train.output_dir is required when test.weights is null")
    return Path(output_dir) / "weights" / "best.pt"
