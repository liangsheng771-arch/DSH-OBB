"""Validated scalar settings for the current :class:`SlotDCE` renderer."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SlotDCEConfig:
    # Feature and exposure routing.
    num_slots: int = 5
    feature_channels: int = 48
    context_channels: int = 64
    exposure_prior_strength: float = 1.25
    full_resolution_prior_strength: float = 0.50
    adaptive_prior_histogram_bins: int = 16
    adaptive_prior_hidden_channels: int = 16
    adaptive_prior_adaptation_ratio: float = 0.25
    adaptive_prior_logit_limit: float = 1.0
    adaptive_prior_width_to_sigma_ratio: float = 0.60

    # Tone curve and exposure masks.
    dark_threshold: float = 0.32
    dark_floor: float = 0.16
    bright_curve_offset: float = 0.17
    bright_curve_slope: float = 0.43
    bright_high_start: float = 0.75
    bright_high_rolloff: float = 0.10
    bright_min_scale: float = 0.56
    white_threshold: float = 0.72
    white_saturation_threshold: float = 0.32
    white_core_threshold: float = 0.90
    white_local_contrast_threshold: float = 0.04
    white_min_scale: float = 0.54
    core_min_scale: float = 0.43
    halo_threshold: float = 0.46
    halo_saturation_threshold: float = 0.50
    halo_local_contrast_threshold: float = 0.08
    halo_min_scale: float = 0.68
    halo_context_inner_kernel_size: int = 31
    halo_context_outer_kernel_size: int = 121
    halo_context_strength: float = 0.52
    halo_maximum_fraction: float = 0.20
    halo_neighbour_floor_ratio: float = 0.82
    mask_temperature: float = 0.035

    # Spatial support.
    illumination_kernel_size: int = 63
    white_expansion_kernel_size: int = 31
    white_feather_kernel_size: int = 15
    halo_expansion_kernel_size: int = 81
    halo_feather_kernel_size: int = 31
    halo_reference_size: int = 384

    # Learned correction strengths.
    dark_max_strength: float = 0.98
    white_max_strength: float = 0.98
    minimum_strength: float = 0.10
    strength_anchors: Sequence[float] = (0.72, 0.62, 0.0, 0.78, 0.90)
    slot_adjustment_scale: float = 0.35
    local_modulation_scale: float = 0.25

    # Detail reconstruction.
    detail_kernel_sizes: Sequence[int] = (5, 13, 31)
    detail_gain_anchors: Sequence[float] = (1.35, 1.22, 1.10)
    detail_gain_adjustment_scale: float = 0.25
    detail_compensation_scale: float = 1.0
    detail_reliable_std: float = 0.003
    detail_minimum_luminance: float = 0.03
    detail_saturation_ceiling: float = 0.985

    # Saturated-core context recovery.
    recovery_saturation_threshold: float = 0.985
    recovery_context_kernel_size: int = 41
    recovery_detail_kernel_size: int = 13
    recovery_residual_scale: float = 0.30
    recovery_detail_scale: float = 0.85
    recovery_color_scale: float = 0.20
    recovery_flat_std_threshold: float = 0.008

    direction_minimum_fraction: float = 0.25
    chroma_epsilon: float = 1.0e-6

    @classmethod
    def option_names(cls) -> set[str]:
        return {item.name for item in fields(cls)}

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SlotDCEConfig":
        unknown = set(values) - cls.option_names()
        if unknown:
            raise ValueError(f"Unknown model option(s): {sorted(unknown)}")
        return cls(**dict(values))

    def runtime_attributes(self) -> dict[str, Any]:
        """Return non-buffer settings consumed directly by the renderer."""

        buffer_names = {"strength_anchors", "detail_gain_anchors"}
        values = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name not in buffer_names
        }
        values["detail_kernel_sizes"] = tuple(
            int(value) for value in self.detail_kernel_sizes
        )
        return values

    @staticmethod
    def _validate_odd_kernels(values: Mapping[str, int]) -> None:
        for name, kernel in values.items():
            if kernel <= 1 or kernel % 2 == 0:
                raise ValueError(f"{name} must be an odd integer > 1.")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "strength_anchors",
            tuple(float(value) for value in self.strength_anchors),
        )
        object.__setattr__(
            self,
            "detail_kernel_sizes",
            tuple(int(value) for value in self.detail_kernel_sizes),
        )
        object.__setattr__(
            self,
            "detail_gain_anchors",
            tuple(float(value) for value in self.detail_gain_anchors),
        )

        if self.num_slots != 5:
            raise ValueError("The ordered exposure design requires exactly 5 Slots.")
        if self.feature_channels <= 0 or self.context_channels <= 0:
            raise ValueError("Feature channel counts must be positive.")
        if len(self.strength_anchors) != self.num_slots:
            raise ValueError("strength_anchors must contain one value per Slot.")

        if self.exposure_prior_strength < 0.0 or self.full_resolution_prior_strength < 0.0:
            raise ValueError("Exposure prior strengths must be non-negative.")
        if not 0.0 < self.dark_threshold < self.white_threshold < 1.0:
            raise ValueError("Thresholds must satisfy 0 < dark < white < 1.")
        if not 0.0 < self.dark_floor < self.dark_threshold:
            raise ValueError("dark_floor must lie between 0 and dark_threshold.")
        if not 0.0 < self.core_min_scale < self.white_min_scale < self.bright_min_scale < 1.0:
            raise ValueError("Scales must satisfy core < white < bright < 1.")
        if not 0.0 < self.halo_min_scale < 1.0:
            raise ValueError("halo_min_scale must lie in (0, 1).")
        if not 0.0 <= self.halo_context_strength <= 2.0:
            raise ValueError("halo_context_strength must lie in [0, 2].")
        if not 0.0 < self.halo_maximum_fraction <= 0.5:
            raise ValueError("halo_maximum_fraction must lie in (0, 0.5].")
        if not 0.0 < self.halo_neighbour_floor_ratio <= 1.0:
            raise ValueError("halo_neighbour_floor_ratio must lie in (0, 1].")
        if not 0.0 < self.bright_curve_offset < self.dark_threshold:
            raise ValueError("bright_curve_offset must lie below dark_threshold.")
        if not 0.0 < self.bright_curve_slope < 1.0:
            raise ValueError("bright_curve_slope must lie in (0, 1).")
        if not self.dark_threshold < self.bright_high_start < 1.0:
            raise ValueError("bright_high_start must exceed dark_threshold.")
        if not 0.0 <= self.bright_high_rolloff < 1.0:
            raise ValueError("bright_high_rolloff must lie in [0, 1).")
        if not 0.0 < self.white_saturation_threshold < 1.0:
            raise ValueError("white_saturation_threshold must lie in (0, 1).")
        if not 0.0 < self.halo_saturation_threshold < 1.0:
            raise ValueError("halo_saturation_threshold must lie in (0, 1).")
        if not self.dark_threshold < self.halo_threshold < self.white_threshold:
            raise ValueError("halo_threshold must lie between dark and white thresholds.")
        if self.halo_local_contrast_threshold < self.white_local_contrast_threshold:
            raise ValueError(
                "halo_local_contrast_threshold must not be weaker than the core threshold."
            )
        if not self.white_threshold < self.white_core_threshold <= 1.0:
            raise ValueError("white_core_threshold must exceed white_threshold.")
        if self.white_local_contrast_threshold < 0.0 or self.mask_temperature <= 0.0:
            raise ValueError("Invalid white-mask settings.")
        if not 0.0 < self.minimum_strength < 1.0:
            raise ValueError("minimum_strength must lie in (0, 1).")
        if not self.minimum_strength < self.dark_max_strength <= 1.0:
            raise ValueError("dark_max_strength must exceed minimum_strength.")
        if not self.minimum_strength < self.white_max_strength <= 1.0:
            raise ValueError("white_max_strength must exceed minimum_strength.")
        if self.chroma_epsilon <= 0.0:
            raise ValueError("chroma_epsilon must be positive.")
        if not 0.0 <= self.local_modulation_scale < 1.0:
            raise ValueError("local_modulation_scale must lie in [0, 1).")
        if not 0.0 < self.slot_adjustment_scale <= 1.0:
            raise ValueError("slot_adjustment_scale must lie in (0, 1].")

        self._validate_odd_kernels(
            {
                "illumination_kernel_size": self.illumination_kernel_size,
                "white_expansion_kernel_size": self.white_expansion_kernel_size,
                "white_feather_kernel_size": self.white_feather_kernel_size,
                "halo_expansion_kernel_size": self.halo_expansion_kernel_size,
                "halo_feather_kernel_size": self.halo_feather_kernel_size,
                "halo_context_inner_kernel_size": self.halo_context_inner_kernel_size,
                "halo_context_outer_kernel_size": self.halo_context_outer_kernel_size,
                "recovery_context_kernel_size": self.recovery_context_kernel_size,
                "recovery_detail_kernel_size": self.recovery_detail_kernel_size,
            }
        )
        if self.halo_reference_size <= 0:
            raise ValueError("halo_reference_size must be positive.")
        if self.halo_context_inner_kernel_size >= self.halo_context_outer_kernel_size:
            raise ValueError("The outer halo context must exceed the inner context.")

        detail_kernels = tuple(int(value) for value in self.detail_kernel_sizes)
        if len(detail_kernels) != 3:
            raise ValueError("detail_kernel_sizes must contain three scales.")
        if any(value <= 1 or value % 2 == 0 for value in detail_kernels):
            raise ValueError("detail_kernel_sizes must be odd integers > 1.")
        if any(left >= right for left, right in zip(detail_kernels, detail_kernels[1:])):
            raise ValueError("detail_kernel_sizes must be strictly increasing.")
        detail_anchors = tuple(float(value) for value in self.detail_gain_anchors)
        if len(detail_anchors) != 3 or not all(1.0 <= value <= 1.5 for value in detail_anchors):
            raise ValueError("detail_gain_anchors must contain three values in [1, 1.5].")
        if not 0.0 <= self.detail_gain_adjustment_scale <= 0.5:
            raise ValueError("detail_gain_adjustment_scale must lie in [0, 0.5].")
        if not 0.0 < self.detail_compensation_scale <= 1.0:
            raise ValueError("detail_compensation_scale must lie in (0, 1].")
        if self.detail_reliable_std < 0.0:
            raise ValueError("detail_reliable_std must be non-negative.")
        if not 0.0 < self.detail_minimum_luminance < self.detail_saturation_ceiling < 1.0:
            raise ValueError("Invalid reliable-detail luminance range.")
        if not 0.0 < self.direction_minimum_fraction <= 1.0:
            raise ValueError("direction_minimum_fraction must lie in (0, 1].")
        if not self.detail_saturation_ceiling <= self.recovery_saturation_threshold < 1.0:
            raise ValueError(
                "recovery_saturation_threshold must be at least the detail ceiling and below 1."
            )
        if self.recovery_detail_kernel_size >= self.recovery_context_kernel_size:
            raise ValueError("The recovery context must exceed its detail scale.")
        if not 0.0 < self.recovery_residual_scale <= 0.5:
            raise ValueError("recovery_residual_scale must lie in (0, 0.5].")
        if not 0.0 <= self.recovery_detail_scale <= 1.5:
            raise ValueError("recovery_detail_scale must lie in [0, 1.5].")
        if not 0.0 <= self.recovery_color_scale <= 1.0:
            raise ValueError("recovery_color_scale must lie in [0, 1].")
        if self.recovery_flat_std_threshold < 0.0:
            raise ValueError("recovery_flat_std_threshold must be non-negative.")

        anchors = tuple(float(value) for value in self.strength_anchors)
        if not (
            self.minimum_strength <= anchors[0] <= self.dark_max_strength
            and self.minimum_strength <= anchors[1] <= self.dark_max_strength
            and anchors[2] == 0.0
            and self.minimum_strength <= anchors[3] <= self.white_max_strength
            and self.minimum_strength <= anchors[4] <= self.white_max_strength
        ):
            raise ValueError("strength_anchors do not match the Slot directions.")
