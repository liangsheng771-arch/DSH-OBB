"""Configuration validation for the RTDETRV2_hyE_yolo neck."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


NECK_NAME = "rtdetrv2_hye_yolo"
SUPPORTED_CHANNELS = (128, 256, 512)
SUPPORTED_STRIDES = (8, 16, 32)

_AIFI_DEFAULTS = {
    "num_encoder_layers": 1,
    "nhead": 8,
    "dim_feedforward": 1024,
    "dropout": 0.0,
    "activation": "gelu",
    "normalize_before": False,
    "pe_temperature": 10000.0,
}
_CCFF_DEFAULTS = {
    "num_blocks": 3,
    "expansion": 0.5,
    "activation": "silu",
}
_GEOMETRY_DEFAULTS = {
    "use_axis_sampling": True,
    "use_shape_routing": True,
    "use_geometry_supervision": True,
    "aux_loss_weight": 0.2,
    "area_scale": 64.0,
}
_CCFF_SWITCHES = (
    "use_wavelet_ccff",
    "use_sga_ccff",
    "use_selective_ccff",
    "use_geometry_ccff",
    "use_geometry_ccff_v1_1",
)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _triplet(value: Any, name: str) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three integers")
    result = tuple(_positive_int(item, f"{name}[{index}]") for index, item in enumerate(value))
    return result  # type: ignore[return-value]


def normalize_rtdetrv2_hye_yolo_config(
    settings: Mapping[str, Any] | None,
    *,
    actual_in_channels: Sequence[int] | None = None,
    actual_feat_strides: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate and return a canonical, checkpoint-safe neck configuration."""

    if settings is None:
        settings = {}
    if not isinstance(settings, Mapping):
        raise TypeError("rtdetrv2_hye_yolo must be a YAML mapping")
    settings = dict(settings)
    unknown = set(settings).difference({
        "in_channels", "feat_strides", "aifi", "ccff", "use_ig_lsf_aifi",
        "ig_lsf_aifi", "sga_ccff", "selective_ccff", "geometry_ccff",
        *_CCFF_SWITCHES,
    })
    if unknown:
        raise ValueError(f"Unknown rtdetrv2_hye_yolo settings: {sorted(unknown)}")

    in_channels = _triplet(settings.get("in_channels", SUPPORTED_CHANNELS), "rtdetrv2_hye_yolo.in_channels")
    feat_strides = _triplet(settings.get("feat_strides", SUPPORTED_STRIDES), "rtdetrv2_hye_yolo.feat_strides")
    if in_channels != SUPPORTED_CHANNELS:
        raise ValueError(
            "RTDETRV2_hyE_yolo currently requires in_channels=[128, 256, 512], "
            f"got {list(in_channels)}"
        )
    if feat_strides != SUPPORTED_STRIDES:
        raise ValueError(
            "RTDETRV2_hyE_yolo currently requires feat_strides=[8, 16, 32], "
            f"got {list(feat_strides)}"
        )
    if actual_in_channels is not None and tuple(int(value) for value in actual_in_channels) != in_channels:
        raise ValueError(
            "RTDETRV2_hyE_yolo configured in_channels do not match the backbone metadata: "
            f"configured={list(in_channels)}, actual={list(actual_in_channels)}"
        )
    if actual_feat_strides is not None and tuple(int(value) for value in actual_feat_strides) != feat_strides:
        raise ValueError(
            "RTDETRV2_hyE_yolo configured feat_strides do not match the backbone metadata: "
            f"configured={list(feat_strides)}, actual={list(actual_feat_strides)}"
        )

    aifi = settings.get("aifi", {})
    if not isinstance(aifi, Mapping):
        raise TypeError("rtdetrv2_hye_yolo.aifi must be a YAML mapping")
    unknown_aifi = set(aifi).difference(_AIFI_DEFAULTS)
    if unknown_aifi:
        raise ValueError(f"Unknown rtdetrv2_hye_yolo.aifi settings: {sorted(unknown_aifi)}")
    aifi = {**_AIFI_DEFAULTS, **dict(aifi)}
    aifi["num_encoder_layers"] = _positive_int(
        aifi["num_encoder_layers"], "rtdetrv2_hye_yolo.aifi.num_encoder_layers"
    )
    aifi["nhead"] = _positive_int(aifi["nhead"], "rtdetrv2_hye_yolo.aifi.nhead")
    if SUPPORTED_CHANNELS[-1] % aifi["nhead"]:
        raise ValueError("rtdetrv2_hye_yolo.aifi.nhead must divide the 512-channel P5 width")
    aifi["dim_feedforward"] = _positive_int(
        aifi["dim_feedforward"], "rtdetrv2_hye_yolo.aifi.dim_feedforward"
    )
    dropout = aifi["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValueError("rtdetrv2_hye_yolo.aifi.dropout must lie in [0, 1)")
    dropout = float(dropout)
    if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError("rtdetrv2_hye_yolo.aifi.dropout must lie in [0, 1)")
    aifi["dropout"] = dropout
    aifi["activation"] = str(aifi["activation"]).lower()
    if aifi["activation"] not in {"relu", "gelu", "silu", "leaky_relu"}:
        raise ValueError("rtdetrv2_hye_yolo.aifi.activation is unsupported")
    if not isinstance(aifi["normalize_before"], bool):
        raise TypeError("rtdetrv2_hye_yolo.aifi.normalize_before must be true or false")
    aifi["pe_temperature"] = _positive_finite(
        aifi["pe_temperature"], "rtdetrv2_hye_yolo.aifi.pe_temperature"
    )

    ccff = settings.get("ccff", {})
    if not isinstance(ccff, Mapping):
        raise TypeError("rtdetrv2_hye_yolo.ccff must be a YAML mapping")
    unknown_ccff = set(ccff).difference(_CCFF_DEFAULTS)
    if unknown_ccff:
        raise ValueError(f"Unknown rtdetrv2_hye_yolo.ccff settings: {sorted(unknown_ccff)}")
    ccff = {**_CCFF_DEFAULTS, **dict(ccff)}
    ccff["num_blocks"] = _positive_int(
        ccff["num_blocks"], "rtdetrv2_hye_yolo.ccff.num_blocks"
    )
    ccff["expansion"] = _positive_finite(
        ccff["expansion"], "rtdetrv2_hye_yolo.ccff.expansion"
    )
    ccff["activation"] = str(ccff["activation"]).lower()
    if ccff["activation"] not in {"relu", "gelu", "silu", "leaky_relu"}:
        raise ValueError("rtdetrv2_hye_yolo.ccff.activation is unsupported")

    use_ig_lsf_aifi = settings.get("use_ig_lsf_aifi", False)
    if not isinstance(use_ig_lsf_aifi, bool):
        raise TypeError("rtdetrv2_hye_yolo.use_ig_lsf_aifi must be true or false")
    switches = {key: settings.get(key, False) for key in _CCFF_SWITCHES}
    for key, enabled in switches.items():
        if not isinstance(enabled, bool):
            raise TypeError(f"rtdetrv2_hye_yolo.{key} must be true or false")
    if sum(switches.values()) > 1:
        raise ValueError("Only one CCFF variant may be enabled at a time")
    if any(switches[key] for key in ("use_wavelet_ccff", "use_sga_ccff", "use_selective_ccff")):
        raise ValueError("This project retains only Geometry CCFF v1 and v1.1")

    ig_settings = settings.get("ig_lsf_aifi", {})
    if not isinstance(ig_settings, Mapping):
        raise TypeError("rtdetrv2_hye_yolo.ig_lsf_aifi must be a YAML mapping")
    if use_ig_lsf_aifi:
        from slot_yolo.ig_lsf_aifi import normalize_ig_lsf_aifi_config
        ig_settings = normalize_ig_lsf_aifi_config(ig_settings)
    elif ig_settings:
        raise ValueError("ig_lsf_aifi settings require use_ig_lsf_aifi=true")

    geometry_settings = settings.get("geometry_ccff", {})
    if not isinstance(geometry_settings, Mapping):
        raise TypeError("rtdetrv2_hye_yolo.geometry_ccff must be a YAML mapping")
    if switches["use_geometry_ccff"]:
        unknown_geometry = set(geometry_settings).difference(_GEOMETRY_DEFAULTS)
        if unknown_geometry:
            raise ValueError(f"Unknown geometry_ccff settings: {sorted(unknown_geometry)}")
        geometry_settings = {**_GEOMETRY_DEFAULTS, **dict(geometry_settings)}
        for key in ("use_axis_sampling", "use_shape_routing", "use_geometry_supervision"):
            if not isinstance(geometry_settings[key], bool):
                raise TypeError(f"geometry_ccff.{key} must be true or false")
        geometry_settings["area_scale"] = _positive_finite(geometry_settings["area_scale"], "geometry_ccff.area_scale")
        weight = geometry_settings["aux_loss_weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError("geometry_ccff.aux_loss_weight must be finite and nonnegative")
        geometry_settings["aux_loss_weight"] = float(weight)
    elif geometry_settings:
        raise ValueError("geometry_ccff settings require use_geometry_ccff=true")

    selective_settings = settings.get("selective_ccff", {})
    if not isinstance(selective_settings, Mapping):
        raise TypeError("rtdetrv2_hye_yolo.selective_ccff must be a YAML mapping")
    if set(selective_settings).difference({"use_scale_gate", "use_spatial_refine"}):
        raise ValueError("Unsupported selective_ccff settings")
    if any(selective_settings.values()):
        raise ValueError("Selective CCFF is no longer included in this project")
    sga_settings = settings.get("sga_ccff", {})
    if not isinstance(sga_settings, Mapping) or sga_settings:
        raise ValueError("SGA CCFF is no longer included in this project")

    return {
        "in_channels": list(in_channels),
        "feat_strides": list(feat_strides),
        "aifi": aifi,
        "ccff": ccff,
        "use_ig_lsf_aifi": use_ig_lsf_aifi,
        **switches,
        "ig_lsf_aifi": dict(ig_settings),
        "sga_ccff": {},
        "selective_ccff": {},
        "geometry_ccff": dict(geometry_settings),
    }


def validate_rtdetrv2_hye_yolo_selection(config: Mapping[str, Any]) -> bool:
    """Validate neck-switch relationships without constructing the neck."""

    enabled = config.get("use_rtdetrv2_hye_yolo", False)
    if not isinstance(enabled, bool):
        raise TypeError("use_rtdetrv2_hye_yolo must be true or false")
    normalize_rtdetrv2_hye_yolo_config(config.get("rtdetrv2_hye_yolo", {}))
    if not enabled:
        return False
    if config.get("module_version", "v2") != "v2":
        raise ValueError("RTDETRV2_hyE_yolo requires module_version: v2")
    if config.get("use_efficient_hybrid_encoder", False) is True:
        raise ValueError(
            "use_rtdetrv2_hye_yolo and use_efficient_hybrid_encoder are mutually exclusive"
        )
    for key in ("use_ig_aifi", "use_ig_fs_ccff", "use_p5_dual_branch"):
        if config.get(key, False) is True:
            raise ValueError(
                f"{key}=true belongs to the existing Efficient Hybrid experiment path and "
                "is not connected to RTDETRV2_hyE_yolo"
            )
    model_type = str(config.get("model_type", "slot_yolo")).lower().replace("-", "_")
    if model_type not in {"slot", "slot_yolo"}:
        raise ValueError("use_rtdetrv2_hye_yolo=true requires model_type: slot_yolo")
    backbone = str(config.get("backbone", "")).lower()
    if backbone != "yolov8s":
        raise ValueError("use_rtdetrv2_hye_yolo=true currently supports only backbone: yolov8s")
    task = str(config.get("task", "detect")).lower()
    if task not in {"detect", "hbb", "obb"}:
        raise ValueError("RTDETRV2_hyE_yolo supports task: detect/hbb/obb")
    return True


def ccff_variant_name(settings: Mapping[str, Any]) -> str:
    """Return the stable experiment name for the selected CCFF."""
    if settings.get("use_geometry_ccff_v1_1", False):
        return "geometry_ccff_v1_1"
    if settings.get("use_geometry_ccff", False):
        return "geometry_ccff_full"
    return "native_ccff"


__all__ = [
    "NECK_NAME",
    "SUPPORTED_CHANNELS",
    "SUPPORTED_STRIDES",
    "normalize_rtdetrv2_hye_yolo_config",
    "validate_rtdetrv2_hye_yolo_selection",
    "ccff_variant_name",
]
