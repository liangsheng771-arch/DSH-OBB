"""Configurable Raw/Slot-DCE feature fusion for the Slot-YOLO pyramid."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from . import ISAF, SlotGuidanceAdapter
from rtdetrv2_hye_config import ccff_variant_name

STAGE_NAMES = ("p3", "p4", "p5")


def _mapping(value: Any, key: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        return {"name": value}
    if not isinstance(value, Mapping):
        raise TypeError(f"feature_modules.{key} must be a name or mapping")
    return dict(value)


def _selected_sgdc_config(value: Any) -> tuple[str, dict[str, Any]]:
    """Select one nested SGDC preset while keeping flat legacy configs valid."""
    config = _mapping(value, "sgdc")
    variant = str(config.pop("name", "sgdc")).lower()
    presets = config.pop("variants", None)
    if presets is not None:
        if not isinstance(presets, Mapping):
            raise TypeError("feature_modules.sgdc.variants must be a YAML mapping")
        unknown = set(presets).difference({"sgdc", "sgdcv2"})
        if unknown:
            raise ValueError(f"Unknown SGDC variant presets: {sorted(unknown)}")
        selected = presets.get(variant, {})
        if not isinstance(selected, Mapping):
            raise TypeError(f"feature_modules.sgdc.variants.{variant} must be a YAML mapping")
        overlap = set(config).intersection(selected).difference({"placement"})
        if overlap:
            raise ValueError(f"Duplicate SGDC settings in common and selected preset: {sorted(overlap)}")
        config.update(dict(selected))
    return variant, config






class _ResidualChannelGate(nn.Module):
    """Blend the enhanced P5 candidate into the raw P5 residual path."""

    def __init__(self, channels: int, initial_gate: float = 0.25) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must lie in (0, 1)")

        self.channels = channels
        self.projection = nn.Conv2d(
            2 * channels,
            channels,
            kernel_size=1,
            groups=channels,
            bias=True,
        )
        nn.init.zeros_(self.projection.weight)
        nn.init.constant_(
            self.projection.bias,
            math.log(initial_gate / (1.0 - initial_gate)),
        )

    def forward(
        self,
        base: Tensor,
        candidate: Tensor,
        return_gate: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if base.shape != candidate.shape or base.ndim != 4:
            raise ValueError(
                "P5 residual fusion requires aligned [B,C,H,W] features, got "
                f"{tuple(base.shape)} and {tuple(candidate.shape)}"
            )
        if base.shape[1] != self.channels:
            raise ValueError(f"configured for {self.channels} channels, got {base.shape[1]}")

        # Group c must see (raw_c, enhanced_c), not two adjacent raw channels.
        descriptor = torch.stack(
            (base.mean((2, 3), keepdim=True), candidate.mean((2, 3), keepdim=True)),
            dim=2,
        ).flatten(1, 2)
        gate = torch.sigmoid(self.projection(descriptor))
        output = base + gate * (candidate - base)
        return (output, gate) if return_gate else output


class P5ResidualFusion(nn.Module):
    """Keep the existing lightweight P5 residual gate independent of ISAF."""

    def __init__(self, channels: int, initial_gate: float = 0.25) -> None:
        super().__init__()
        self.residual_gate = _ResidualChannelGate(channels, initial_gate)

    def forward(
        self,
        raw_feature: Tensor,
        enhanced_feature: Tensor,
        return_gate: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        return self.residual_gate(
            raw_feature,
            enhanced_feature,
            return_gate=return_gate,
        )


class ConfiguredFeatureStage(nn.Module):
    """Adapt Slot-DCE decisions and apply frequency-aware ISAF."""

    def __init__(
        self,
        *,
        stage: str,
        channels: int,
        guidance_channels: int,
        guidance_config: Mapping[str, Any],
        isaf_config: Mapping[str, Any],
    ) -> None:
        nn.Module.__init__(self)
        self.stage = stage
        adapted_channels = int(guidance_config.get("channels", 16))
        hidden_channels = guidance_config.get("hidden_channels")
        hidden_channels = None if hidden_channels is None else int(hidden_channels)
        guidance_norm = str(guidance_config.get("norm", "bn")).lower()
        if guidance_channels <= 0 or adapted_channels <= 0:
            raise ValueError("ISAF guidance channel counts must be positive")
        self.guidance_adapter = SlotGuidanceAdapter(
            in_channels=guidance_channels,
            out_channels=adapted_channels,
            hidden_channels=hidden_channels,
            norm=guidance_norm,
        )
        self.fusion = ISAF(
            channels=channels,
            illum_channels=adapted_channels,
            reduction=int(isaf_config.get("reduction", 4)),
            norm=str(isaf_config.get("norm", guidance_norm)).lower(),
            tone_gate_init=float(isaf_config.get("tone_gate_init", 0.20)),
            texture_gate_init=float(isaf_config.get("texture_gate_init", 0.35)),
            alignment_scale_init=float(isaf_config.get("alignment_scale_init", 0.0)),
            texture_scale_init=float(isaf_config.get("texture_scale_init", 0.0)),
        )

    def forward(
        self,
        raw_feature: Tensor,
        enhanced_feature: Tensor,
        detector_guidance: Tensor,
    ) -> Tensor:
        guidance = self.guidance_adapter(
            detector_guidance,
            target_size=raw_feature.shape[-2:],
        )
        return self.fusion(raw_feature, enhanced_feature, guidance)


def build_feature_stages(
    channels: Sequence[int],
    slot_config: Mapping[str, Any],
    guidance_channels: int = 3,
) -> tuple[tuple[nn.Module, nn.Module, nn.Module], bool, str]:
    """Build the switched Slot-DCE + ISAF feature path."""
    if len(channels) != len(STAGE_NAMES):
        raise ValueError(f"Expected three pyramid channel values, got {len(channels)}")

    enabled = slot_config.get("use_slotdce_isaf", True)
    if not isinstance(enabled, bool):
        raise TypeError("use_slotdce_isaf must be true or false")
    if not enabled:
        return (nn.Identity(), nn.Identity(), nn.Identity()), False, (
            "Slot-DCE+ISAF:OFF (raw backbone features)"
        )

    feature_config = slot_config.get("feature_modules", {})
    if not isinstance(feature_config, Mapping):
        raise TypeError("feature_modules must be a YAML mapping")

    feature_config = dict(feature_config)
    unknown = set(feature_config).difference(
        {"guidance", "isaf", "sgdc"}
    )
    if unknown:
        raise ValueError(
            "Unknown feature_modules entries: "
            f"{sorted(unknown)}"
        )
    guidance_config = _mapping(feature_config.get("guidance"), "guidance")
    isaf_config = _mapping(feature_config.get("isaf"), "isaf")
    fusion_config = slot_config.get("fusion", {})
    if not isinstance(fusion_config, Mapping):
        raise TypeError("fusion must be a YAML mapping")

    p3 = ConfiguredFeatureStage(
        stage="p3",
        channels=int(channels[0]),
        guidance_channels=guidance_channels,
        guidance_config=guidance_config,
        isaf_config=isaf_config,
    )
    p4 = ConfiguredFeatureStage(
        stage="p4",
        channels=int(channels[1]),
        guidance_channels=guidance_channels,
        guidance_config=guidance_config,
        isaf_config=isaf_config,
    )
    p5 = P5ResidualFusion(
        int(channels[2]),
        initial_gate=float(fusion_config.get("p5_gate_init", 0.25)),
    )
    return (p3, p4, p5), True, "P3:ISAF, P4:ISAF, P5:RESIDUAL"


def build_neck(
    input_channels: Sequence[int],
    output_channels: Sequence[int],
    slot_config: Mapping[str, Any],
    *,
    input_strides: Sequence[int] | None = None,
) -> tuple[nn.Module, bool, str]:
    """Build the selected native YOLO PAN or RT-DETR encoder.

    A disabled encoder preserves the original YOLO PAN. The returned identity
    is only a placeholder; the model executes the original parsed PAN layers.
    """

    if len(input_channels) != len(STAGE_NAMES):
        raise ValueError(
            f"Expected three neck input channel values, got {len(input_channels)}"
        )
    if len(output_channels) != len(STAGE_NAMES):
        raise ValueError(
            f"Expected three neck output channel values, got {len(output_channels)}"
        )

    from rtdetrv2_hye_config import validate_rtdetrv2_hye_yolo_selection
    validate_rtdetrv2_hye_yolo_selection(slot_config)
    from ig_config import validate_ig_config, validate_p5_config
    ig_flags = validate_ig_config(slot_config)
    p5_dual_branch = validate_p5_config(slot_config)

    use_new_neck = slot_config.get("use_rtdetrv2_hye_yolo", False)
    if use_new_neck:
        from ..rtdetrv2_hye_yolo import build_rtdetrv2_hye_yolo

        with torch.random.fork_rng(devices=[]):
            module = build_rtdetrv2_hye_yolo(
                input_channels,
                output_channels,
                slot_config.get("rtdetrv2_hye_yolo", {}),
                input_strides=input_strides,
            )
        attention = "IGLSFAIFI" if module.use_ig_lsf_aifi else "native AIFI"
        fusion = ccff_variant_name(module.architecture_config)
        return module, True, f"Neck:RTDETRV2_hyE_yolo (P5-512 {attention}, [128,256,512] {fusion})"

    # False or a missing switch selects the original parsed PAN layers unless
    # the existing Efficient Hybrid path is explicitly enabled.
    enabled = slot_config.get("use_efficient_hybrid_encoder", False)
    if not isinstance(enabled, bool):
        raise TypeError("use_efficient_hybrid_encoder must be true or false")
    if enabled:
        raise ValueError("Efficient Hybrid is not included in the retained model")

    return nn.Identity(), False, "Neck:YOLO-PAN"


def build_sgdc(
    channels: Sequence[int],
    slot_config: Mapping[str, Any],
    calibration_channels: Sequence[int] | None = None,
) -> tuple[nn.Module, bool, str]:
    """Build SGDC settings from ``feature_modules.sgdc`` when ``use_sgdc`` is true.

    Disabled SGDC is represented by ``nn.Identity`` and is never called by the
    model, so the selected neck outputs pass directly to the Detect head.
    """

    if len(channels) != len(STAGE_NAMES):
        raise ValueError(f"Expected three pyramid channel values, got {len(channels)}")
    calibration_channels = channels if calibration_channels is None else calibration_channels
    if len(calibration_channels) != len(STAGE_NAMES):
        raise ValueError(
            "Expected three SGDC calibration channel values, got "
            f"{len(calibration_channels)}"
        )

    switch = slot_config.get("use_sgdc", False)
    if not isinstance(switch, bool):
        raise TypeError("use_sgdc must be true or false")

    feature_config = slot_config.get("feature_modules", {})
    if not isinstance(feature_config, Mapping):
        raise TypeError("feature_modules must be a YAML mapping")

    variant, sgdc_config = _selected_sgdc_config(dict(feature_config).get("sgdc"))
    if not switch:
        return nn.Identity(), False, "SGDC:OFF"

    raise ValueError("SGDC is not included in the retained model")


__all__ = [
    "ConfiguredFeatureStage",
    "P5ResidualFusion",
    "STAGE_NAMES",
    "build_feature_stages",
    "build_neck",
    "build_sgdc",
]
