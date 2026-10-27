"""Four-objective loss used by the current SlotDCE model."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import Tensor, nn

from image_utils import luminance


DEFAULT_WEIGHTS = {
    "tone": 1.00,
    "texture": 0.25,
    "recovery": 0.50,
    "smoothness": 0.05,
}

REQUIRED_OUTPUTS = frozenset(
    {
        "input_luminance",
        "output_luminance",
        "illumination_luminance",
        "base_tone_delta",
        "bright_curve_scale",
        "halo_context_excess",
        "halo_total_delta",
        "dark_weight",
        "bright_weight",
        "white_weight",
        "near_light_weight",
        "white_core_weight",
        "halo_weight",
    }
)
REQUIRED_RECOVERY_OUTPUTS = frozenset({"recovery_rgb", "recovery_confidence"})


def _masked_mean(value: Tensor, weight: Tensor) -> Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


class EdgeAwareSmoothnessLoss(nn.Module):
    """Smooth a correction field without smoothing across image edges."""

    def __init__(self, edge_scale: float = 12.0) -> None:
        super().__init__()
        if edge_scale < 0.0:
            raise ValueError("edge_scale must be non-negative.")
        self.edge_scale = float(edge_scale)

    def forward(self, input_image: Tensor, correction: Tensor) -> Tensor:
        value = luminance(input_image)
        correction_dx = (correction[..., 1:] - correction[..., :-1]).abs()
        correction_dy = (correction[..., 1:, :] - correction[..., :-1, :]).abs()
        value_dx = (value[..., 1:] - value[..., :-1]).abs()
        value_dy = (value[..., 1:, :] - value[..., :-1, :]).abs()
        return 0.5 * (
            (correction_dx * torch.exp(-self.edge_scale * value_dx)).mean()
            + (correction_dy * torch.exp(-self.edge_scale * value_dy)).mean()
        )


class SlotDCELoss(nn.Module):
    """Train tone, texture, clipped-detail recovery and field smoothness."""

    def __init__(
        self,
        *,
        dark_threshold: float,
        dark_floor: float,
        dark_target_strength: float,
        bright_min_scale: float,
        bright_target_strength: float,
        white_min_scale: float,
        white_target_strength: float,
        core_min_scale: float,
        halo_min_scale: float,
        halo_context_strength: float,
        halo_maximum_fraction: float,
        texture_minimum_luminance: float,
        texture_saturation_ceiling: float,
        weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        super().__init__()
        resolved = dict(DEFAULT_WEIGHTS)
        if weights is not None:
            unknown = set(weights) - set(resolved)
            if unknown:
                raise ValueError(f"Unknown loss weights: {sorted(unknown)}")
            resolved.update({name: float(value) for name, value in weights.items()})
        if any(value < 0.0 for value in resolved.values()):
            raise ValueError("Loss weights must be non-negative.")
        if not 0.0 < dark_floor < dark_threshold < 1.0:
            raise ValueError("Invalid dark tone target.")
        if not 0.0 < core_min_scale < white_min_scale < bright_min_scale < 1.0:
            raise ValueError("Invalid highlight tone scales.")
        if not 0.0 < halo_min_scale < 1.0:
            raise ValueError("Invalid halo tone scale.")
        if not 0.0 < texture_minimum_luminance < texture_saturation_ceiling < 1.0:
            raise ValueError("Invalid texture reliability interval.")

        self.weights = resolved
        self.dark_threshold = float(dark_threshold)
        self.dark_floor = float(dark_floor)
        self.dark_target_strength = float(dark_target_strength)
        self.bright_min_scale = float(bright_min_scale)
        self.bright_target_strength = float(bright_target_strength)
        self.white_min_scale = float(white_min_scale)
        self.white_target_strength = float(white_target_strength)
        self.core_min_scale = float(core_min_scale)
        self.core_target_strength = 0.92
        self.halo_min_scale = float(halo_min_scale)
        self.halo_target_strength = 0.86
        self.halo_context_strength = float(halo_context_strength)
        self.halo_maximum_fraction = float(halo_maximum_fraction)
        self.tone_error_scale = 0.10
        self.texture_highlight_gain = 0.15
        self.texture_minimum_luminance = float(texture_minimum_luminance)
        self.texture_saturation_ceiling = float(texture_saturation_ceiling)
        self.recovery_confidence_floor = 0.05
        self.smoothness_halo_weight = 0.50
        self.robust_epsilon = 1.0e-3
        self.smoothness = EdgeAwareSmoothnessLoss(edge_scale=12.0)

    @property
    def loss_names(self) -> tuple[str, ...]:
        return tuple(self.weights)

    @property
    def required_output_names(self) -> frozenset[str]:
        return REQUIRED_OUTPUTS

    @property
    def required_recovery_output_names(self) -> frozenset[str]:
        return REQUIRED_RECOVERY_OUTPUTS

    def _robust_l1(self, difference: Tensor) -> Tensor:
        epsilon = self.robust_epsilon
        return (difference.square() + epsilon**2).sqrt() - epsilon

    def _tone_target_and_weight(
        self,
        outputs: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        value = outputs["input_luminance"]
        illumination = outputs["illumination_luminance"]
        dark = outputs["dark_weight"]
        bright = outputs["bright_weight"]
        near_white = outputs["near_light_weight"]
        core = outputs["white_core_weight"]
        halo = outputs["halo_weight"]

        dark_candidate = self.dark_floor + (
            self.dark_threshold - self.dark_floor
        ) * (illumination / self.dark_threshold).clamp(0.0, 1.0)
        target_dark_delta = (
            dark
            * self.dark_target_strength
            * (dark_candidate - illumination).clamp_min(0.0)
        )
        target_bright_scale = (
            1.0
            - self.bright_target_strength
            * (1.0 - outputs["bright_curve_scale"])
        ).clamp(self.bright_min_scale, 1.0)
        target_white_scale = (
            1.0
            - near_white
            * self.white_target_strength
            * (1.0 - self.white_min_scale)
        ).clamp(self.white_min_scale, 1.0)
        target_core_scale = (
            1.0
            - core
            * self.core_target_strength
            * (1.0 - self.core_min_scale)
        ).clamp(self.core_min_scale, 1.0)
        target_base_scale = torch.minimum(
            target_bright_scale,
            torch.minimum(target_white_scale, target_core_scale),
        )
        target_base_compression = (
            (1.0 - dark) * value * (1.0 - target_base_scale)
        )
        target_halo_scale = (
            1.0
            - halo
            * self.halo_target_strength
            * (1.0 - self.halo_min_scale)
        ).clamp(self.halo_min_scale, 1.0)
        target_halo_delta = (1.0 - dark) * (
            target_base_scale * value * (1.0 - target_halo_scale)
            + halo
            * self.halo_target_strength
            * self.halo_context_strength
            * outputs["halo_context_excess"]
        )
        target_halo_delta = torch.minimum(
            target_halo_delta,
            value * self.halo_maximum_fraction,
        )
        target_delta = (
            target_dark_delta - target_base_compression - target_halo_delta
        ).detach()
        correction = torch.maximum(
            dark,
            torch.maximum(bright, outputs["white_weight"]),
        )
        weight = (correction * (1.0 + 2.0 * core + halo)).detach()
        return target_delta, weight

    def _tone_objective(self, outputs: Mapping[str, Tensor]) -> Tensor:
        target_delta, weight = self._tone_target_and_weight(outputs)
        error = self._robust_l1(outputs["base_tone_delta"] - target_delta)
        return _masked_mean(error / self.tone_error_scale, weight)

    def _texture_objective(
        self,
        input_image: Tensor,
        outputs: Mapping[str, Tensor],
    ) -> Tensor:
        source = luminance(input_image)
        prediction = outputs["output_luminance"]
        source_log = torch.log(source.clamp_min(self.texture_minimum_luminance))
        prediction_log = torch.log(
            prediction.clamp_min(self.texture_minimum_luminance)
        )
        source_maximum = input_image.amax(dim=1, keepdim=True)
        reliable = torch.sigmoid(
            (source - self.texture_minimum_luminance) / 0.015
        ) * torch.sigmoid(
            (self.texture_saturation_ceiling - source_maximum) / 0.010
        )
        highlight = torch.maximum(
            outputs["bright_weight"], outputs["white_weight"]
        ).detach()
        gain = 1.0 + self.texture_highlight_gain * highlight

        source_dx = source_log[..., 1:] - source_log[..., :-1]
        source_dy = source_log[..., 1:, :] - source_log[..., :-1, :]
        prediction_dx = prediction_log[..., 1:] - prediction_log[..., :-1]
        prediction_dy = prediction_log[..., 1:, :] - prediction_log[..., :-1, :]
        gain_x = 0.5 * (gain[..., 1:] + gain[..., :-1])
        gain_y = 0.5 * (gain[..., 1:, :] + gain[..., :-1, :])
        weight_x = torch.minimum(reliable[..., 1:], reliable[..., :-1])
        weight_y = torch.minimum(reliable[..., 1:, :], reliable[..., :-1, :])
        target_dx = (gain_x * source_dx).detach()
        target_dy = (gain_y * source_dy).detach()
        error = 0.5 * (
            _masked_mean(self._robust_l1(prediction_dx - target_dx), weight_x)
            + _masked_mean(self._robust_l1(prediction_dy - target_dy), weight_y)
        )
        reference = 0.5 * (
            _masked_mean(target_dx.abs(), weight_x)
            + _masked_mean(target_dy.abs(), weight_y)
        )
        return error / reference.clamp_min(0.01)

    def _recovery_objective(
        self,
        target: Tensor,
        outputs: Mapping[str, Tensor],
        mask: Tensor,
    ) -> Tensor:
        prediction = outputs["recovery_rgb"]
        confidence = outputs["recovery_confidence"].clamp(1.0e-5, 1.0 - 1.0e-5)
        color_error = self._robust_l1(prediction - target).mean(
            dim=1, keepdim=True
        )
        likelihood = (
            confidence * color_error
            - self.recovery_confidence_floor * torch.log(confidence)
        )
        return _masked_mean(likelihood, mask.to(prediction.dtype))

    def _smoothness_objective(
        self,
        input_image: Tensor,
        outputs: Mapping[str, Tensor],
    ) -> Tensor:
        return self.smoothness(
            input_image, outputs["base_tone_delta"]
        ) + self.smoothness_halo_weight * self.smoothness(
            input_image, outputs["halo_total_delta"]
        )

    def forward(
        self,
        input_image: Tensor,
        outputs: Mapping[str, Tensor],
        recovery_target: Optional[Tensor] = None,
        recovery_outputs: Optional[Mapping[str, Tensor]] = None,
        recovery_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        recovery_items = (recovery_target, recovery_outputs, recovery_mask)
        if any(item is not None for item in recovery_items) and not all(
            item is not None for item in recovery_items
        ):
            raise ValueError(
                "recovery_target, recovery_outputs and recovery_mask "
                "must be provided together."
            )

        zero = outputs["input_luminance"].new_zeros(())
        raw = {
            "tone": self._tone_objective(outputs) if self.weights["tone"] else zero,
            "texture": (
                self._texture_objective(input_image, outputs)
                if self.weights["texture"]
                else zero
            ),
            "recovery": zero,
            "smoothness": (
                self._smoothness_objective(input_image, outputs)
                if self.weights["smoothness"]
                else zero
            ),
        }
        if recovery_target is not None and self.weights["recovery"]:
            assert recovery_outputs is not None and recovery_mask is not None
            raw["recovery"] = self._recovery_objective(
                recovery_target, recovery_outputs, recovery_mask
            )

        contributions = {
            name: self.weights[name] * raw[name] for name in self.loss_names
        }
        result = {"total": sum(contributions.values()), **contributions}
        return result
