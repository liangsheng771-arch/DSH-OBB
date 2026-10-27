"""Validation shared by the experiment loader and direct model construction."""

from collections.abc import Mapping
import math


def validate_ig_config(config):
    flags = {}
    for name in ("ig_aifi", "ig_fs_ccff"):
        key = f"use_{name}"
        enabled = config.get(key, False)
        if not isinstance(enabled, bool):
            raise TypeError(f"{key} must be true or false")
        flags[key] = enabled
        if enabled and (config.get("use_efficient_hybrid_encoder", False) is not True
                        or config.get("use_slotdce_isaf", True) is not True):
            raise ValueError(f"{key}=true requires use_efficient_hybrid_encoder=true and use_slotdce_isaf=true")
        settings = config.get(name, {})
        if not isinstance(settings, Mapping):
            raise TypeError(f"{name} must be a YAML mapping")
        allowed = ({"gate_hidden_dim", "lambda_init", "relation_matrix_std", "pair_chunk_size"}
                   if name == "ig_aifi" else {"hidden_dim", "wavelet", "gate_hidden_channels"})
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")
        for key, value in settings.items():
            if key == "wavelet":
                if value != "haar":
                    raise ValueError("ig_fs_ccff.wavelet currently supports only haar")
            elif key == "hidden_dim":
                if isinstance(value, bool) or not isinstance(value, int) or value != 256:
                    raise ValueError("ig_fs_ccff.hidden_dim must be 256")
            elif key in {"gate_hidden_dim", "gate_hidden_channels", "pair_chunk_size"}:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{name}.{key} must be a positive integer")
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name}.{key} must be finite")
            elif key == "relation_matrix_std" and value <= 0:
                raise ValueError("ig_aifi.relation_matrix_std must be positive")
    return flags


def validate_p5_config(config):
    """Validate the independent P5 branch without constructing any modules."""
    enabled = config.get("use_p5_dual_branch", False)
    if not isinstance(enabled, bool):
        raise TypeError("use_p5_dual_branch must be true or false")
    if enabled and config.get("use_efficient_hybrid_encoder", False) is not True:
        raise ValueError("use_p5_dual_branch=true requires use_efficient_hybrid_encoder=true")
    settings = config.get("p5_semantic_refine", {})
    if not isinstance(settings, Mapping):
        raise TypeError("p5_semantic_refine must be a YAML mapping")
    unknown = set(settings) - {"channel_reduction", "local_kernel", "context_kernel", "delta_init_std"}
    if unknown:
        raise ValueError(f"Unknown p5_semantic_refine settings: {sorted(unknown)}")
    for key, value in settings.items():
        if key == "delta_init_std":
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError("p5_semantic_refine.delta_init_std must be finite and positive")
        elif isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"p5_semantic_refine.{key} must be a positive integer")
        elif key == "channel_reduction" and 256 % value:
            raise ValueError("p5_semantic_refine.channel_reduction must divide 256")
        elif key in {"local_kernel", "context_kernel"} and value % 2 == 0:
            raise ValueError(f"p5_semantic_refine.{key} must be odd")
    return enabled
