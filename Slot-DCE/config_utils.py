"""Construction and checkpoint helpers for the current SlotDCE model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union

import torch
import yaml
from torch import nn

from losses import SlotDCELoss
from models import SlotDCE, SlotDCEConfig


ARCHITECTURE = "slotdce_v9_8_detector_lite"
LOSS_KEYS = {"weights"}


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping.")
    for section in ("model", "loss", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing mapping section: {section}")
    if config.get("architecture") != ARCHITECTURE:
        raise ValueError(f"Only architecture={ARCHITECTURE!r} is supported.")
    return config


def build_model(config: Mapping[str, Any]) -> SlotDCE:
    return SlotDCE(SlotDCEConfig.from_mapping(config))


def build_loss(
    config: Mapping[str, Any],
    model_config: Mapping[str, Any] | None = None,
) -> SlotDCELoss:
    if model_config is None:
        raise ValueError("The loss requires the model configuration.")
    unknown = set(config) - LOSS_KEYS
    if unknown:
        raise ValueError(f"Unknown loss option(s): {sorted(unknown)}")
    settings = SlotDCEConfig.from_mapping(model_config)
    anchors = tuple(float(value) for value in settings.strength_anchors)
    return SlotDCELoss(
        dark_threshold=settings.dark_threshold,
        dark_floor=float(settings.dark_floor),
        dark_target_strength=anchors[0],
        bright_min_scale=settings.bright_min_scale,
        bright_target_strength=anchors[4],
        white_min_scale=settings.white_min_scale,
        white_target_strength=anchors[3],
        core_min_scale=settings.core_min_scale,
        halo_min_scale=settings.halo_min_scale,
        halo_context_strength=settings.halo_context_strength,
        halo_maximum_fraction=settings.halo_maximum_fraction,
        texture_minimum_luminance=settings.detail_minimum_luminance,
        texture_saturation_ceiling=settings.detail_saturation_ceiling,
        **dict(config),
    )


def choose_device(requested: str = "auto") -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.isdecimal():
        normalized = f"cuda:{normalized}"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as error:
        raise ValueError("device must be auto, cpu, cuda, cuda:N, or a GPU index") from error
    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device.index} is not available.")
    return device


def torch_load(path: Union[str, Path], device: torch.device) -> Any:
    return torch.load(path, map_location=device, weights_only=False)


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch_load(checkpoint_path, device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary.")
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError(f"Checkpoint must use architecture={ARCHITECTURE!r}.")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary.")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("Checkpoint parameters do not match the current model.") from error
    return checkpoint
