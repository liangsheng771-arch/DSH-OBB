"""Current detector-oriented SlotDCE illumination correction model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from image_utils import luminance
from local_filters import local_max, local_mean

from .adaptive_exposure_prior import AdaptiveExposurePrior
from .slot_dce_config import SlotDCEConfig


@dataclass(frozen=True)
class _BaseImageContext:
    """Parameter-free image statistics shared by every Slot-DCE branch."""

    input_value: Tensor
    input_maximum: Tensor
    input_white: Tensor
    illumination: Tensor
    value_base: Tensor
    dark_weight: Tensor
    bright_weight: Tensor
    white_weight: Tensor
    white_core_weight: Tensor
    near_light_weight: Tensor
    halo_weight: Tensor
    input_saturation: Tensor
    exposure_signal: Tensor
    bright_curve_scale: Tensor
    routing_value: Tensor


@dataclass(frozen=True)
class _PreparedDetectionContext:
    """Static renderer inputs retained instead of recomputed in backward."""

    base: _BaseImageContext
    halo_context_excess: Tensor
    detail_means: tuple[Tensor, ...]
    fine_std: Tensor
    recovery_input: Tensor
    clipped_white_weight: Tensor


class DepthwiseSeparableResidualBlock(nn.Module):
    """Residual spatial mixing with depthwise and pointwise convolutions.

    Each dense 3x3 convolution is replaced by a depthwise convolution followed
    by a 1x1 channel mixer. All learned spatial mixing runs at H/4.
    """

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1

        def stage(stage_dilation: int, activate: bool) -> list[nn.Module]:
            modules: list[nn.Module] = [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=stage_dilation,
                    dilation=stage_dilation,
                    groups=channels,
                    bias=False,
                ),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.GroupNorm(groups, channels),
            ]
            if activate:
                modules.append(nn.SiLU(inplace=True))
            return modules

        self.block = nn.Sequential(
            *stage(dilation, activate=True),
            *stage(1, activate=False),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, features: Tensor) -> Tensor:
        return self.activation(features + self.block(features))


class DepthwiseSeparableConv(nn.Module):
    """Shape-changing depthwise spatial convolution plus 1x1 channel mixing."""

    def __init__(self, inputs: int, outputs: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, outputs)
        while outputs % groups:
            groups -= 1
        self.depthwise = nn.Conv2d(
            inputs,
            inputs,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=inputs,
            bias=False,
        )
        self.pointwise = nn.Conv2d(inputs, outputs, kernel_size=1, bias=False)
        self.normalization = nn.GroupNorm(groups, outputs)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, features: Tensor) -> Tensor:
        features = self.depthwise(features)
        features = self.pointwise(features)
        return self.activation(self.normalization(features))


def _pad_to_factor(value: Tensor, factor: int) -> tuple[Tensor, int, int]:
    """Pad only the bottom/right edge for a reversible space-to-depth step."""

    height, width = value.shape[-2:]
    pad_height = (-height) % factor
    pad_width = (-width) % factor
    if pad_height or pad_width:
        mode = "reflect" if height > pad_height and width > pad_width else "replicate"
        value = F.pad(value, (0, pad_width, 0, pad_height), mode=mode)
    return value, height, width


class SpaceToDepthStem(nn.Module):
    """Losslessly rearrange RGB pixels before learned spatial processing."""

    def __init__(self, input_channels: int, output_channels: int, factor: int) -> None:
        super().__init__()
        groups = min(8, output_channels)
        while output_channels % groups:
            groups -= 1
        self.factor = int(factor)
        self.projection = nn.Conv2d(
            input_channels * factor * factor,
            output_channels,
            kernel_size=1,
            bias=False,
        )
        self.normalization = nn.GroupNorm(groups, output_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, image: Tensor) -> Tensor:
        padded, _, _ = _pad_to_factor(image, self.factor)
        packed = F.pixel_unshuffle(padded, self.factor)
        return self.activation(self.normalization(self.projection(packed)))


class SpaceToDepthRecoveryHead(nn.Module):
    """Predict full-resolution recovery fields without full-resolution convolutions."""

    def __init__(self, channels: int, factor: int) -> None:
        super().__init__()
        self.factor = int(factor)
        self.input = nn.Conv2d(7 * factor * factor, channels, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)
        # Dilation is divided by the spatial reduction factor, preserving the
        # approximate 2/4/8-pixel context of the original full-resolution head.
        dilations = (1, 2, 4) if factor <= 2 else (1, 1, 2)
        self.blocks = nn.Sequential(
            *(DepthwiseSeparableResidualBlock(channels, dilation=dilation)
              for dilation in dilations)
        )
        self.output = nn.Conv2d(
            channels,
            4 * factor * factor,
            kernel_size=1,
        )

    def forward(self, features: Tensor) -> Tensor:
        packed, height, width = _pad_to_factor(features, self.factor)
        packed = F.pixel_unshuffle(packed, self.factor)
        packed = self.activation(self.input(packed))
        prediction = F.pixel_shuffle(self.output(self.blocks(packed)), self.factor)
        return prediction[..., :height, :width]


class SlotDCE(nn.Module):
    """Correct low-frequency illumination with five ordered exposure Slots.

    The model predicts one image-conditioned correction strength for each of
    five ordered exposure Slots.  The rendering equation enforces the intended
    sign by construction: S1/S2 can only brighten, S3 is neutral, and S4/S5
    compress bright/white illumination.  Tone correction is predicted on a
    low-frequency illumination layer; reliable relative detail is reconstructed
    afterwards.  This prevents non-clipped lamp texture from being flattened.
    """

    def __init__(self, config: SlotDCEConfig) -> None:
        super().__init__()
        if not isinstance(config, SlotDCEConfig):
            raise TypeError("config must be a SlotDCEConfig instance.")
        settings = config

        # Scalar rendering settings are not learned weights and do not belong
        # in the state dictionary. Modules and learned buffers are explicit.
        self.config = settings
        for name, value in settings.runtime_attributes().items():
            setattr(self, name, value)
        # Five Slot probabilities plus tone delta, absolute tone delta, raw
        # luminance, reliable detail and saturated-region recovery confidence.
        self.detection_guidance_channels = settings.num_slots + 5

        num_slots = settings.num_slots
        feature_channels = settings.feature_channels
        context_channels = settings.context_channels
        anchors = tuple(float(value) for value in settings.strength_anchors)
        detail_anchors = tuple(float(value) for value in settings.detail_gain_anchors)

        self.exposure_prior = AdaptiveExposurePrior(
            num_slots=num_slots,
            histogram_bins=settings.adaptive_prior_histogram_bins,
            hidden_channels=settings.adaptive_prior_hidden_channels,
            adaptation_ratio=settings.adaptive_prior_adaptation_ratio,
            logit_limit=settings.adaptive_prior_logit_limit,
            width_to_sigma_ratio=settings.adaptive_prior_width_to_sigma_ratio,
        )
        self.register_buffer(
            "slot_directions",
            torch.tensor((1.0, 1.0, 0.0, -1.0, -1.0)),
        )
        self.register_buffer(
            "strength_anchors",
            torch.tensor(anchors, dtype=torch.float32),
        )
        self.register_buffer(
            "detail_gain_anchors",
            torch.tensor(detail_anchors, dtype=torch.float32),
        )

        # PixelUnshuffle moves the image to H/4 without discarding pixels.
        # Every learned 3x3 operation below is depthwise separable and remains
        # on that compact grid.
        self.encoder = nn.Sequential(
            SpaceToDepthStem(3, feature_channels, factor=4),
            DepthwiseSeparableResidualBlock(feature_channels),
            DepthwiseSeparableConv(feature_channels, feature_channels),
            DepthwiseSeparableResidualBlock(feature_channels),
            DepthwiseSeparableConv(feature_channels, context_channels),
            DepthwiseSeparableResidualBlock(context_channels, dilation=1),
            DepthwiseSeparableResidualBlock(context_channels, dilation=2),
            DepthwiseSeparableResidualBlock(context_channels, dilation=3),
        )
        self.attention_head = nn.Conv2d(context_channels, num_slots, kernel_size=1)
        local_channels = max(16, context_channels // 2)
        prediction_groups = min(8, local_channels)
        while local_channels % prediction_groups:
            prediction_groups -= 1
        self.prediction_trunk = nn.Sequential(
            nn.Conv2d(
                context_channels,
                context_channels,
                kernel_size=3,
                padding=1,
                groups=context_channels,
                bias=False,
            ),
            nn.Conv2d(context_channels, local_channels, kernel_size=1, bias=False),
            nn.GroupNorm(prediction_groups, local_channels),
            nn.SiLU(inplace=True),
        )
        self.local_head = nn.Conv2d(local_channels, 3, kernel_size=1)
        self.halo_head = nn.Conv2d(local_channels, 1, kernel_size=1)
        self.detail_head = nn.Conv2d(local_channels, 4, kernel_size=1)
        self.recovery_head = SpaceToDepthRecoveryHead(local_channels, factor=4)
        hidden = max(32, context_channels // 2)
        self.strength_head = nn.Sequential(
            nn.Linear(context_channels + 2, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # The brightness prior defines stable Slot semantics at initialization.
        nn.init.zeros_(self.attention_head.weight)
        nn.init.zeros_(self.attention_head.bias)
        nn.init.zeros_(self.local_head.weight)
        nn.init.zeros_(self.local_head.bias)
        nn.init.zeros_(self.halo_head.weight)
        nn.init.zeros_(self.halo_head.bias)
        nn.init.zeros_(self.detail_head.weight)
        nn.init.zeros_(self.detail_head.bias)
        nn.init.zeros_(self.recovery_head.output.weight)
        nn.init.zeros_(self.recovery_head.output.bias)
        final = self.strength_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.exposure_prior.reset_output()

    def _prior_logits(
        self,
        value: Tensor,
        centers: Tensor,
        sigmas: Tensor,
        strength: float,
    ) -> Tensor:
        centers = centers.to(value).view(value.shape[0], self.num_slots, 1, 1)
        sigma = sigmas.to(value).view(value.shape[0], self.num_slots, 1, 1)
        return -strength * (value - centers).square() / (
            2.0 * sigma.square().clamp_min(1.0e-8)
        )

    def _slot_strengths(
        self,
        slots: Tensor,
        usage: Tensor,
        centers: Tensor,
    ) -> Tensor:
        inputs = torch.cat((slots, centers.to(slots).unsqueeze(-1), usage.unsqueeze(-1)), dim=-1)
        adjustment = self.slot_adjustment_scale * torch.tanh(
            self.strength_head(inputs).squeeze(-1)
        )

        maximum = slots.new_tensor(
            (
                self.dark_max_strength,
                self.dark_max_strength,
                1.0,
                self.white_max_strength,
                self.white_max_strength,
            )
        ).view(1, self.num_slots)
        anchor = self.strength_anchors.to(slots).view(1, self.num_slots)
        ratio = (
            (anchor - self.minimum_strength)
            / (maximum - self.minimum_strength).clamp_min(1e-6)
        ).clamp(1e-4, 1.0 - 1e-4)
        anchor_logit = torch.logit(ratio)
        strength = self.minimum_strength + (
            maximum - self.minimum_strength
        ) * torch.sigmoid(anchor_logit + adjustment)
        return strength * self.slot_directions.to(strength).abs().view(1, -1)

    @staticmethod
    def _local_mean(value: Tensor, kernel_size: int) -> Tensor:
        return local_mean(value, kernel_size)

    def _guided_illumination_native(
        self,
        value: Tensor,
        kernel_size: int,
    ) -> Tensor:
        # Box filtering is channel-independent. Group tensors that use the same
        # kernel so CUDA launches two separable filters here instead of four.
        first_moments = self._local_mean(
            torch.cat((value, value.square()), dim=1),
            kernel_size,
        )
        mean, correlation = first_moments.chunk(2, dim=1)
        variance = (correlation - mean.square()).clamp_min(0.0)
        slope = variance / (variance + 1.0e-3)
        intercept = mean - slope * mean
        affine = self._local_mean(
            torch.cat((slope, intercept), dim=1),
            kernel_size,
        )
        smooth_slope, smooth_intercept = affine.chunk(2, dim=1)
        return (smooth_slope * value + smooth_intercept).clamp(0.0, 1.0)

    @staticmethod
    def _scaled_odd_kernel(kernel_size: int, factor: int) -> int:
        scaled = max(3, round(kernel_size / factor))
        return scaled if scaled % 2 else scaled + 1

    def _guided_illumination(self, value: Tensor) -> Tensor:
        """Edge-aware low-frequency illumination used by every tone branch."""

        factor = 2
        if min(value.shape[-2:]) < 2 * factor:
            return self._guided_illumination_native(
                value,
                self.illumination_kernel_size,
            )
        height, width = value.shape[-2:]
        working = F.interpolate(
            value,
            size=((height + factor - 1) // factor, (width + factor - 1) // factor),
            mode="area",
        )
        illumination = self._guided_illumination_native(
            working,
            self._scaled_odd_kernel(self.illumination_kernel_size, factor),
        )
        return F.interpolate(
            illumination,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    def _halo_support(self, white_seed: Tensor) -> Tensor:
        """Grow lamp seeds at a fixed scene-relative reference resolution."""

        height, width = white_seed.shape[-2:]
        scale = min(1.0, self.halo_reference_size / max(min(height, width), 1))
        if scale < 1.0:
            target = (
                max(1, round(height * scale)),
                max(1, round(width * scale)),
            )
            working = F.interpolate(
                white_seed,
                size=target,
                mode="bilinear",
                align_corners=False,
            )
        else:
            working = white_seed
        expansion = local_max(working, self.halo_expansion_kernel_size)
        support = self._local_mean(expansion, self.halo_feather_kernel_size)
        if support.shape[-2:] != (height, width):
            support = F.interpolate(
                support,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        return support.clamp(0.0, 1.0)

    def _reference_local_mean(self, value: Tensor, kernel_size: int) -> Tensor:
        """Measure neighbourhood illumination at the halo reference scale."""

        height, width = value.shape[-2:]
        scale = min(1.0, self.halo_reference_size / max(min(height, width), 1))
        if scale < 1.0:
            target = (
                max(1, round(height * scale)),
                max(1, round(width * scale)),
            )
            working = F.interpolate(
                value, size=target, mode="bilinear", align_corners=False
            )
        else:
            working = value
        result = self._local_mean(working, kernel_size)
        if result.shape[-2:] != (height, width):
            result = F.interpolate(
                result,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        return result

    def _exposure_masks(
        self,
        maximum: Tensor,
        minimum: Tensor,
        illumination: Tensor,
        value_base: Tensor,
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """Build dark, broad-bright, white-core and surrounding-halo masks."""

        saturation = (maximum - minimum) / maximum.clamp_min(self.chroma_epsilon)
        temperature = self.mask_temperature
        exposure_signal = 0.75 * illumination + 0.25 * value_base
        low_saturation = torch.sigmoid(
            (self.white_saturation_threshold - saturation) / temperature
        )
        white_level = torch.sigmoid((minimum - self.white_threshold) / temperature)
        white_peak = torch.sigmoid(
            (maximum - self.white_core_threshold) / temperature
        )
        local_peak = torch.sigmoid(
            (maximum - value_base - self.white_local_contrast_threshold)
            / temperature
        )
        white_seed = white_level * low_saturation * torch.maximum(white_peak, local_peak)
        core_expansion = local_max(white_seed, self.white_expansion_kernel_size)
        core_feather = self._local_mean(
            core_expansion, self.white_feather_kernel_size
        )
        # A core must remain compact: expanding it would incorrectly label
        # white machinery and the whole lamp neighbourhood as lamp filament.
        white_core_weight = white_seed.clamp(0.0, 1.0)
        near_light_weight = (
            0.30 * white_seed + 0.70 * core_feather
        ).clamp(0.0, 1.0)

        # The halo is a ring grown only from a compact white-light seed.  Local
        # exposure and low saturation prevent the expanded support from
        # suppressing unrelated dark structure around the lamp.
        halo_local_peak = torch.sigmoid(
            (
                maximum
                - value_base
                - self.halo_local_contrast_threshold
            )
            / temperature
        )
        halo_seed = white_level * low_saturation * white_peak * halo_local_peak
        halo_support = self._halo_support(halo_seed)
        halo_level = torch.sigmoid(
            (exposure_signal - self.halo_threshold) / temperature
        )
        halo_low_saturation = torch.sigmoid(
            (self.halo_saturation_threshold - saturation) / (1.5 * temperature)
        )
        raw_halo_weight = (
            halo_support
            * halo_level
            * halo_low_saturation
            * (1.0 - white_core_weight)
        ).clamp(0.0, 1.0)
        # Square-root remapping keeps the outer ring visible after three soft
        # gates without turning weak distant support into a hard binary mask.
        halo_weight = torch.sqrt(raw_halo_weight)
        # Keep the core separate but allow the learned halo field to cover both
        # the bright inner ring and the outer veil. Excluding the V9 near-light
        # field here would leave a conspicuous bright annulus around the core.
        white_weight = torch.maximum(near_light_weight, halo_weight)

        # A wide exposure signal includes low-frequency RGB value so metallic
        # reflections and bright backgrounds are compressed together with the
        # lamp neighbourhood. The candidate curve is calibrated on mine scenes;
        # Slots learn the spatial fraction applied to it.
        high_fraction = (
            (exposure_signal - self.bright_high_start)
            / (1.0 - self.bright_high_start)
        ).clamp(0.0, 1.0)
        bright_candidate = (
            self.bright_curve_offset
            + self.bright_curve_slope * exposure_signal
            - self.bright_high_rolloff * high_fraction
        )
        raw_bright_curve_scale = (
            bright_candidate / exposure_signal.clamp_min(self.chroma_epsilon)
        ).clamp(self.bright_min_scale, 1.0)
        dark_gate = torch.sigmoid(
            (self.dark_threshold - illumination) / temperature
        )
        # Dark illumination exits the shoulder continuously. Dark texture on a
        # genuinely bright object still follows that object's exposure field,
        # because its low-frequency illumination is not dark.
        bright_curve_scale = 1.0 - (
            (1.0 - raw_bright_curve_scale) * (1.0 - dark_gate)
        )
        bright_weight = (
            (1.0 - bright_curve_scale) / (1.0 - self.bright_min_scale)
        ).clamp(0.0, 1.0)
        dark_weight = (dark_gate * (1.0 - white_weight)).clamp(0.0, 1.0)
        return (
            dark_weight,
            bright_weight,
            white_weight,
            white_core_weight,
            near_light_weight,
            halo_weight,
            saturation,
            exposure_signal,
            bright_curve_scale,
        )

    def _validate_image(self, image: Tensor) -> None:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected an RGB tensor shaped [B, 3, H, W].")

    def _prepare_base_image_context(self, image: Tensor) -> _BaseImageContext:
        """Compute the parameter-free exposure statistics used by Slot-DCE."""

        input_value = luminance(image)
        input_maximum = image.amax(dim=1, keepdim=True)
        input_white = image.amin(dim=1, keepdim=True)
        illumination, value_base = self._guided_illumination(
            torch.cat((input_value, input_maximum), dim=1)
        ).chunk(2, dim=1)
        (
            dark_weight,
            bright_weight,
            white_weight,
            white_core_weight,
            near_light_weight,
            halo_weight,
            input_saturation,
            exposure_signal,
            bright_curve_scale,
        ) = self._exposure_masks(
            input_maximum,
            input_white,
            illumination,
            value_base,
        )
        routing_value = torch.where(
            white_weight > 0.25,
            torch.maximum(exposure_signal, input_white),
            exposure_signal,
        )
        return _BaseImageContext(
            input_value=input_value,
            input_maximum=input_maximum,
            input_white=input_white,
            illumination=illumination,
            value_base=value_base,
            dark_weight=dark_weight,
            bright_weight=bright_weight,
            white_weight=white_weight,
            white_core_weight=white_core_weight,
            near_light_weight=near_light_weight,
            halo_weight=halo_weight,
            input_saturation=input_saturation,
            exposure_signal=exposure_signal,
            bright_curve_scale=bright_curve_scale,
            routing_value=routing_value,
        )

    def _prepare_recovery_input(
        self,
        image: Tensor,
        input_white: Tensor,
        input_saturation: Tensor,
        fine_std: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build the fixed RGB context consumed by the learned recovery head."""

        clipped_white_weight = torch.sigmoid(
            (input_white - self.recovery_saturation_threshold) / 0.006
        ) * torch.sigmoid(
            (self.white_saturation_threshold - input_saturation)
            / self.mask_temperature
        )
        lost_structure_weight = torch.sigmoid(
            (self.recovery_flat_std_threshold - fine_std) / 0.003
        )
        recovery_weight = (clipped_white_weight * lost_structure_weight).clamp(
            0.0,
            1.0,
        )
        known_weight = (1.0 - clipped_white_weight).detach()
        context_input = torch.cat((known_weight, image * known_weight), dim=1)
        context_factor = 2
        if min(image.shape[-2:]) >= 2 * context_factor:
            height, width = image.shape[-2:]
            context_input = F.interpolate(
                context_input,
                size=(
                    (height + context_factor - 1) // context_factor,
                    (width + context_factor - 1) // context_factor,
                ),
                mode="area",
            )
            context_statistics = self._local_mean(
                context_input,
                self._scaled_odd_kernel(
                    self.recovery_context_kernel_size,
                    context_factor,
                ),
            )
            context_rgb = context_statistics[:, 1:] / context_statistics[
                :, :1
            ].clamp_min(0.02)
            context_rgb = F.interpolate(
                context_rgb,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        else:
            context_statistics = self._local_mean(
                context_input,
                self.recovery_context_kernel_size,
            )
            context_rgb = context_statistics[:, 1:] / context_statistics[
                :, :1
            ].clamp_min(0.02)
        recovery_input = torch.cat((image, context_rgb, recovery_weight), dim=1)
        return recovery_input, clipped_white_weight

    def prepare_detection_context(self, image: Tensor) -> _PreparedDetectionContext:
        """Precompute parameter-free tensors that full checkpointing repeated."""

        self._validate_image(image)
        with torch.no_grad():
            base = self._prepare_base_image_context(image)
            halo_context_excess = (
                self._reference_local_mean(
                    base.exposure_signal,
                    self.halo_context_inner_kernel_size,
                )
                - self._reference_local_mean(
                    base.exposure_signal,
                    self.halo_context_outer_kernel_size,
                )
            ).clamp(0.0, 1.0)
            logarithm = torch.log(base.input_value.clamp_min(self.chroma_epsilon))
            detail_means = tuple(
                self._local_mean(logarithm, kernel)
                for kernel in self.detail_kernel_sizes
            )
            fine_base = self._local_mean(
                base.input_value,
                self.detail_kernel_sizes[0],
            )
            fine_std = self._local_mean(
                (base.input_value - fine_base).square(),
                self.detail_kernel_sizes[0],
            ).clamp_min(0.0).sqrt()
            recovery_input, clipped_white_weight = self._prepare_recovery_input(
                image,
                base.input_white,
                base.input_saturation,
                fine_std,
            )
        return _PreparedDetectionContext(
            base=base,
            halo_context_excess=halo_context_excess,
            detail_means=detail_means,
            fine_std=fine_std,
            recovery_input=recovery_input,
            clipped_white_weight=clipped_white_weight,
        )

    def _predict_saturation_recovery(
        self,
        image: Tensor,
        input_value: Tensor,
        input_white: Tensor,
        input_saturation: Tensor,
        fine_std: Tensor,
        prepared_input: Tensor | None = None,
        prepared_clipped_weight: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        """Predict clipped-core colour from intact neighbouring pixels."""

        if prepared_input is None:
            recovery_input, clipped_white_weight = self._prepare_recovery_input(
                image,
                input_white,
                input_saturation,
                fine_std,
            )
        else:
            recovery_input = prepared_input
            clipped_white_weight = prepared_clipped_weight
            if clipped_white_weight is None:
                raise ValueError("prepared_clipped_weight is required with prepared_input")
        context_rgb = recovery_input[:, 3:6]
        recovery_weight = recovery_input[:, 6:7]
        recovery_logits = self.recovery_head(recovery_input)
        recovery_rgb = (
            context_rgb
            + self.recovery_residual_scale * torch.tanh(recovery_logits[:, :3])
        ).clamp(0.0, 1.0)
        recovery_confidence = torch.sigmoid(
            -3.0 + 6.0 * torch.tanh(recovery_logits[:, 3:4])
        )
        return {
            "recovery_rgb": recovery_rgb,
            "recovery_confidence": recovery_confidence,
            "recovery_context_rgb": context_rgb,
            "saturation_recovery_weight": recovery_weight,
            "clipped_white_weight": clipped_white_weight,
        }

    def forward_recovery(self, image: Tensor) -> Dict[str, Tensor]:
        """Run only the branch required by synthetic saturation supervision.

        The recovery loss consumes ``recovery_rgb`` and
        ``recovery_confidence`` only.  Avoiding the unrelated encoder, Slot,
        tone and Retinex branches makes the extra training pass substantially
        cheaper without changing recovery-head inputs or gradients.
        """

        self._validate_image(image)
        input_value = luminance(image)
        input_maximum = image.amax(dim=1, keepdim=True)
        input_white = image.amin(dim=1, keepdim=True)
        input_saturation = (
            (input_maximum - input_white)
            / input_maximum.clamp_min(self.chroma_epsilon)
        )
        fine_base = self._local_mean(input_value, self.detail_kernel_sizes[0])
        fine_std = self._local_mean(
            (input_value - fine_base).square(),
            self.detail_kernel_sizes[0],
        ).clamp_min(0.0).sqrt()
        return self._predict_saturation_recovery(
            image,
            input_value,
            input_white,
            input_saturation,
            fine_std,
        )

    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        """Enhance an RGB tensor and return standalone-training diagnostics."""
        return self._forward_impl(image, return_diagnostics=True)

    def enhance_for_detection(
        self,
        image: Tensor,
        *,
        return_guidance: bool = False,
        guidance_sizes: Sequence[tuple[int, int]] | None = None,
        _prepared_context: _PreparedDetectionContext | None = None,
    ) -> Tensor | tuple[Tensor, Tensor | tuple[Tensor, ...]]:
        """Return enhanced RGB and optional Slot-aware detector guidance."""
        return self._forward_impl(
            image,
            return_diagnostics=False,
            return_guidance=return_guidance,
            guidance_sizes=guidance_sizes,
            prepared_context=_prepared_context,
        )

    def _forward_impl(
        self,
        image: Tensor,
        *,
        return_diagnostics: bool,
        return_guidance: bool = False,
        guidance_sizes: Sequence[tuple[int, int]] | None = None,
        prepared_context: _PreparedDetectionContext | None = None,
    ) -> Dict[str, Tensor] | Tensor | tuple[Tensor, Tensor | tuple[Tensor, ...]]:
        self._validate_image(image)
        if prepared_context is not None and return_diagnostics:
            raise ValueError("Prepared detection context does not support diagnostics")
        base = (
            prepared_context.base
            if prepared_context is not None
            else self._prepare_base_image_context(image)
        )
        input_value = base.input_value
        input_maximum = base.input_maximum
        input_white = base.input_white
        illumination = base.illumination
        value_base = base.value_base
        dark_weight = base.dark_weight
        bright_weight = base.bright_weight
        white_weight = base.white_weight
        white_core_weight = base.white_core_weight
        near_light_weight = base.near_light_weight
        halo_weight = base.halo_weight
        input_saturation = base.input_saturation
        exposure_signal = base.exposure_signal
        bright_curve_scale = base.bright_curve_scale
        # White pixels are routed by min(R,G,B), while colored lights remain on
        # luminance routing.  The renderer below still uses soft spatial masks.
        routing_value = base.routing_value
        features = self.encoder(image)
        value_low = F.adaptive_avg_pool2d(routing_value, features.shape[-2:])
        content_logits = self.attention_head(features)
        (
            exposure_centers,
            exposure_bin_widths,
            exposure_prior_sigmas,
        ) = self.exposure_prior(value_low)
        prior_low = self._prior_logits(
            value_low,
            exposure_centers,
            exposure_prior_sigmas,
            self.exposure_prior_strength,
        )
        attention_low = torch.softmax(content_logits + prior_low, dim=1)

        denominator = attention_low.sum(dim=(2, 3)).clamp_min(1e-6)
        slots = torch.einsum("bkhw,bchw->bkc", attention_low, features)
        slots = slots / denominator.unsqueeze(-1)
        usage = attention_low.mean(dim=(2, 3))
        slot_strength = self._slot_strengths(slots, usage, exposure_centers)

        content_full = F.interpolate(
            attention_low.clamp_min(1e-8).log(),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        prior_full = self._prior_logits(
            routing_value,
            exposure_centers,
            exposure_prior_sigmas,
            self.full_resolution_prior_strength,
        )
        attention_full = torch.softmax(content_full + prior_full, dim=1)

        strengths = slot_strength.view(image.shape[0], self.num_slots, 1, 1)
        # Normalize within each directional group. Probability assigned to the
        # neutral/opposite group must not force all gains toward their maxima.
        dark_attention = attention_full[:, :2]
        white_attention = attention_full[:, 3:]
        dark_weights = dark_attention / dark_attention.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        white_weights = white_attention / white_attention.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        dark_strength_map = (dark_weights * strengths[:, :2]).sum(dim=1, keepdim=True)
        highlight_strength_map = (white_weights * strengths[:, 3:]).sum(
            dim=1, keepdim=True
        )
        prediction_features = self.prediction_trunk(features)
        local_logits = F.interpolate(
            self.local_head(prediction_features),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        dark_modulation = 1.0 + self.local_modulation_scale * torch.tanh(
            local_logits[:, 0:1]
        )
        bright_modulation = 1.0 + self.local_modulation_scale * torch.tanh(
            local_logits[:, 1:2]
        )
        white_modulation = 1.0 + self.local_modulation_scale * torch.tanh(
            local_logits[:, 2:3]
        )
        dark_strength_map = (dark_strength_map * dark_modulation).clamp(0.0, 1.0)
        bright_strength_map = (highlight_strength_map * bright_modulation).clamp(
            0.0, 1.0
        )
        white_strength_map = (highlight_strength_map * white_modulation).clamp(
            0.0, 1.0
        )
        halo_logits = F.interpolate(
            self.halo_head(prediction_features),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        halo_modulation = 1.0 + self.local_modulation_scale * torch.tanh(
            halo_logits
        )
        halo_strength_map = (
            highlight_strength_map * halo_modulation
        ).clamp(0.0, 1.0)
        if not return_diagnostics:
            # Detection only consumes the rendered image and compact guidance.
            # Drop H/4 routing tensors as soon as their prediction maps exist;
            # otherwise Python keeps every local reference alive until return.
            del (
                features,
                value_low,
                content_logits,
                prior_low,
                attention_low,
                slots,
                usage,
                slot_strength,
                strengths,
                dark_attention,
                white_attention,
                dark_weights,
                white_weights,
                local_logits,
                halo_logits,
                content_full,
                prior_full,
                routing_value,
                exposure_centers,
                exposure_bin_widths,
                exposure_prior_sigmas,
            )
        # Optical bloom behaves partly as additive veiling light. Estimate the
        # local excess over a much larger neighbourhood, then let the dedicated
        # halo head decide how much of that excess is removed.
        if prepared_context is None:
            halo_inner_context = self._reference_local_mean(
                exposure_signal,
                self.halo_context_inner_kernel_size,
            )
            halo_outer_context = self._reference_local_mean(
                exposure_signal,
                self.halo_context_outer_kernel_size,
            )
            halo_context_excess = (
                halo_inner_context - halo_outer_context
            ).clamp(0.0, 1.0)
        else:
            halo_inner_context = halo_outer_context = None
            halo_context_excess = prepared_context.halo_context_excess
        # All tone curves act on edge-aware low-frequency illumination.  They
        # therefore cannot use a bright texture pixel as evidence for stronger
        # compression than its neighbour.
        dark_candidate = self.dark_floor + (
            self.dark_threshold - self.dark_floor
        ) * (illumination / self.dark_threshold).clamp(0.0, 1.0)
        dark_delta = (
            dark_weight
            * dark_strength_map
            * (dark_candidate - illumination).clamp_min(0.0)
        )
        bright_scale = (
            1.0
            - bright_strength_map * (1.0 - bright_curve_scale)
        ).clamp(self.bright_min_scale, 1.0)
        white_scale = (
            1.0
            - near_light_weight
            * white_strength_map
            * (1.0 - self.white_min_scale)
        ).clamp(self.white_min_scale, 1.0)
        core_scale = (
            1.0
            - white_core_weight
            * white_strength_map
            * (1.0 - self.core_min_scale)
        ).clamp(self.core_min_scale, 1.0)
        halo_scale = (
            1.0
            - halo_weight
            * halo_strength_map
            * (1.0 - self.halo_min_scale)
        ).clamp(self.halo_min_scale, 1.0)
        base_compression_scale = torch.minimum(
            bright_scale,
            torch.minimum(core_scale, white_scale),
        )
        base_compression_delta = (
            (1.0 - dark_weight)
            * input_value
            * (1.0 - base_compression_scale)
        )
        halo_relative_delta = (
            (1.0 - dark_weight)
            * base_compression_scale
            * input_value
            * (1.0 - halo_scale)
        )
        halo_veil_delta = (
            (1.0 - dark_weight)
            * halo_weight
            * halo_strength_map
            * self.halo_context_strength
            * halo_context_excess
        )
        halo_requested_delta = torch.minimum(
            halo_relative_delta + halo_veil_delta,
            input_value * self.halo_maximum_fraction,
        )
        # A halo estimate is allowed to remove veiling light, but not to create
        # a dark moat below the surrounding tone-mapped neighbourhood.  This is
        # a renderer safety bound; the learned halo head still decides the
        # requested removal inside that physically plausible range.
        base_tone_without_halo = (
            input_value + dark_delta - base_compression_delta
        ).clamp(0.0, 1.0)
        halo_uncompensated = base_tone_without_halo - halo_requested_delta
        halo_neighbour_reference = self._reference_local_mean(
            base_tone_without_halo,
            self.halo_context_outer_kernel_size,
        )
        halo_neighbour_floor = (
            self.halo_neighbour_floor_ratio * halo_neighbour_reference
        )
        halo_floor_gap = (halo_neighbour_floor - halo_uncompensated).clamp_min(0.0)
        halo_floor_compensation = torch.minimum(
            halo_requested_delta,
            halo_weight * (1.0 - white_core_weight) * halo_floor_gap,
        )
        halo_total_delta = (
            halo_requested_delta - halo_floor_compensation
        ).clamp_min(0.0)
        compression_delta = base_compression_delta + halo_total_delta
        # Blend competing branches instead of subtracting both at full strength.
        # Dark illumination gets priority; the compression branch takes over
        # continuously as the dark confidence falls.
        base_tone_delta = dark_delta - compression_delta
        tone_value = (input_value + base_tone_delta).clamp(0.0, 1.0)

        # Multi-scale Retinex reconstruction.  The three Laplacian bands carry
        # fine edges, medium texture and coarse local shape separately.  Their
        # gains are predicted by a dedicated head and trained with no-reference
        # reflectance targets.  Pure clipped cores are excluded because they do
        # not contain recoverable texture.
        logarithm = torch.log(input_value.clamp_min(self.chroma_epsilon))
        detail_means = (
            tuple(
                self._local_mean(logarithm, kernel)
                for kernel in self.detail_kernel_sizes
            )
            if prepared_context is None
            else prepared_context.detail_means
        )
        detail_bands = torch.cat(
            (
                logarithm - detail_means[0],
                detail_means[0] - detail_means[1],
                detail_means[1] - detail_means[2],
            ),
            dim=1,
        )
        detail_logits = F.interpolate(
            self.detail_head(prediction_features),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if not return_diagnostics:
            del prediction_features
        detail_gain_maps = self.detail_gain_anchors.to(detail_logits).view(
            1, 3, 1, 1
        ) * torch.exp(
            self.detail_gain_adjustment_scale * torch.tanh(detail_logits[:, :3])
        )
        # The fourth map learns whether local structure is useful or mainly
        # sensor noise. Its high neutral initialization preserves information;
        # training can suppress unreliable regions without a hand-coded denoiser.
        detail_confidence = torch.sigmoid(
            3.0 + 5.0 * torch.tanh(detail_logits[:, 3:4])
        )
        highlight_detail_weight = torch.maximum(bright_weight, white_weight)
        applied_detail_gains = 1.0 + highlight_detail_weight * (
            detail_gain_maps - 1.0
        )
        desired_log_detail = (
            detail_bands * applied_detail_gains
        ).sum(dim=1, keepdim=True)
        tone_logarithm = torch.log(tone_value.clamp_min(self.chroma_epsilon))
        tone_bases = self._local_mean(
            torch.cat((tone_logarithm, tone_value), dim=1),
            self.detail_kernel_sizes[-1],
        )
        tone_log_base, tone_linear_base = tone_bases.chunk(2, dim=1)
        retinex_unnormalized = torch.exp(tone_log_base + desired_log_detail)
        # Log-domain contrast amplification otherwise raises the arithmetic
        # local mean through convexity. Renormalize each 31x31 neighbourhood so
        # the detail branch redistributes contrast without changing the tone
        # curve's low-frequency exposure decision.
        retinex_local_mean = self._local_mean(
            retinex_unnormalized,
            self.detail_kernel_sizes[-1],
        )
        retinex_value = (
            retinex_unnormalized
            * tone_linear_base
            / retinex_local_mean.clamp_min(self.chroma_epsilon)
        ).clamp(0.0, 1.0)
        if prepared_context is None:
            fine_base = self._local_mean(
                input_value,
                self.detail_kernel_sizes[0],
            )
            fine_std = self._local_mean(
                (input_value - fine_base).square(),
                self.detail_kernel_sizes[0],
            ).clamp_min(0.0).sqrt()
        else:
            fine_base = None
            fine_std = prepared_context.fine_std
        reliable_structure = torch.sigmoid(
            (fine_std - self.detail_reliable_std) / 0.003
        )
        reliable_low = torch.sigmoid(
            (input_value - self.detail_minimum_luminance) / 0.015
        )
        reliable_high = torch.sigmoid(
            (self.detail_saturation_ceiling - input_maximum) / 0.010
        )
        detail_reliability = (
            reliable_structure * reliable_low * reliable_high
        ).detach()
        correction_weight = torch.maximum(
            dark_weight, torch.maximum(bright_weight, white_weight)
        )
        raw_detail_residual = (
            correction_weight
            * detail_reliability
            * detail_confidence
            * self.detail_compensation_scale
            * (retinex_value - tone_value)
        )
        # Detail must be zero-mean at the coarsest reconstruction scale. This
        # prevents a contrast branch from silently becoming a second exposure
        # branch and brightening regions far away from a lamp.
        detail_residual = raw_detail_residual - self._local_mean(
            raw_detail_residual,
            self.detail_kernel_sizes[-1],
        )
        if not return_diagnostics:
            # These full-resolution tone/Retinex tensors are diagnostics-only
            # after the residual and reliability maps have been formed.
            del (
                illumination,
                value_base,
                exposure_signal,
                bright_curve_scale,
                dark_weight,
                bright_weight,
                white_weight,
                white_core_weight,
                near_light_weight,
                halo_weight,
                dark_strength_map,
                bright_strength_map,
                white_strength_map,
                halo_strength_map,
                dark_modulation,
                bright_modulation,
                white_modulation,
                halo_modulation,
                halo_inner_context,
                halo_outer_context,
                halo_context_excess,
                dark_candidate,
                dark_delta,
                bright_scale,
                white_scale,
                core_scale,
                halo_scale,
                base_compression_scale,
                base_compression_delta,
                halo_relative_delta,
                halo_veil_delta,
                halo_requested_delta,
                base_tone_without_halo,
                halo_uncompensated,
                halo_neighbour_reference,
                halo_neighbour_floor,
                halo_floor_gap,
                halo_floor_compensation,
                halo_total_delta,
                compression_delta,
                logarithm,
                detail_means,
                detail_bands,
                detail_logits,
                detail_gain_maps,
                highlight_detail_weight,
                applied_detail_gains,
                desired_log_detail,
                tone_logarithm,
                tone_bases,
                tone_log_base,
                retinex_unnormalized,
                retinex_local_mean,
                retinex_value,
                reliable_structure,
                reliable_low,
                reliable_high,
                correction_weight,
                raw_detail_residual,
            )

        recovery = self._predict_saturation_recovery(
            image,
            input_value,
            input_white,
            input_saturation,
            fine_std,
            prepared_input=(
                prepared_context.recovery_input
                if prepared_context is not None
                else None
            ),
            prepared_clipped_weight=(
                prepared_context.clipped_white_weight
                if prepared_context is not None
                else None
            ),
        )
        recovery_rgb = recovery["recovery_rgb"]
        recovery_confidence = recovery["recovery_confidence"]
        saturation_recovery_weight = recovery["saturation_recovery_weight"]
        if return_diagnostics:
            context_rgb = recovery["recovery_context_rgb"]
            clipped_white_weight = recovery["clipped_white_weight"]
        else:
            # The context image and clipped-core mask have already contributed
            # to recovery_rgb/weight and are not detector outputs.
            del recovery
        recovery_luminance = luminance(recovery_rgb)
        recovery_base = self._local_mean(
            recovery_luminance,
            self.recovery_detail_kernel_size,
        )
        recovery_relative_detail = (
            recovery_luminance - recovery_base
        ) / recovery_base.clamp_min(self.detail_minimum_luminance)
        raw_recovery_detail = (
            saturation_recovery_weight
            * recovery_confidence
            * self.recovery_detail_scale
            * tone_linear_base
            * recovery_relative_detail
        )
        recovery_detail_residual = raw_recovery_detail - self._local_mean(
            raw_recovery_detail,
            self.recovery_detail_kernel_size,
        )
        if not return_diagnostics:
            del (
                input_white,
                input_saturation,
                fine_base,
                fine_std,
                tone_linear_base,
                recovery_base,
                recovery_relative_detail,
                raw_recovery_detail,
            )
        proposed_value = tone_value + detail_residual + recovery_detail_residual
        minimum_change = self.direction_minimum_fraction * base_tone_delta
        output_target = torch.where(
            base_tone_delta > 0.0,
            torch.maximum(proposed_value, input_value + minimum_change),
            torch.where(
                base_tone_delta < 0.0,
                torch.minimum(proposed_value, input_value + minimum_change),
                input_value,
            ),
        ).clamp(0.0, 1.0)
        if not return_diagnostics:
            del (
                tone_value,
                detail_residual,
                recovery_detail_residual,
                proposed_value,
                minimum_change,
                base_tone_delta,
            )

        # A single RGB scale preserves hue and channel ratios.  Dark scaling is
        # capped before RGB clipping, so texture is not destroyed by a second
        # saturation event introduced by the enhancer.
        rgb_scale = output_target / input_value.clamp_min(self.chroma_epsilon)
        gamut_scale = 1.0 / input_maximum.clamp_min(self.chroma_epsilon)
        rgb_scale = torch.where(
            rgb_scale > 1.0,
            torch.minimum(rgb_scale, gamut_scale),
            rgb_scale,
        )
        enhanced = (image * rgb_scale).clamp(0.0, 1.0)
        black = input_value <= self.chroma_epsilon
        enhanced = torch.where(black.expand_as(image), output_target.expand_as(image), enhanced)
        if not return_diagnostics:
            del output_target, rgb_scale, gamut_scale, black
        base_output_value = luminance(enhanced)
        recovery_chroma = recovery_rgb - recovery_luminance
        recovery_color_candidate = (
            base_output_value
            + recovery_chroma
            * base_output_value
            / recovery_luminance.clamp_min(self.chroma_epsilon)
        ).clamp(0.0, 1.0)
        if not return_diagnostics:
            del recovery_rgb, recovery_luminance, recovery_chroma
        recovery_color_weight = (
            saturation_recovery_weight
            * recovery_confidence
            * self.recovery_color_scale
        )
        recovery_colored = (
            enhanced * (1.0 - recovery_color_weight)
            + recovery_color_candidate * recovery_color_weight
        ).clamp(0.0, 1.0)
        # RGB clipping can otherwise turn a nominally zero-luminance chroma
        # residual into an exposure change. Re-normalize the colour refinement
        # to the already direction-constrained luminance. A gamut limit may
        # only make a clipped bright pixel darker, never brighter.
        recovery_colored_value = luminance(recovery_colored)
        recovery_luminance_scale = (
            base_output_value
            / recovery_colored_value.clamp_min(self.chroma_epsilon)
        )
        recovery_gamut_scale = 1.0 / recovery_colored.amax(
            dim=1, keepdim=True
        ).clamp_min(self.chroma_epsilon)
        recovery_luminance_scale = torch.where(
            recovery_luminance_scale > 1.0,
            torch.minimum(recovery_luminance_scale, recovery_gamut_scale),
            recovery_luminance_scale,
        )
        enhanced = (recovery_colored * recovery_luminance_scale).clamp(0.0, 1.0)
        if not return_diagnostics:
            del (
                base_output_value,
                recovery_color_candidate,
                recovery_color_weight,
                recovery_colored,
                recovery_colored_value,
                recovery_luminance_scale,
                recovery_gamut_scale,
                image,
                input_maximum,
            )
        if not return_diagnostics:
            if not return_guidance:
                return enhanced
            output_value = luminance(enhanced)
            tone_delta = output_value - input_value
            guidance_parts = (
                attention_full,
                tone_delta,
                tone_delta.abs(),
                input_value,
                detail_reliability * detail_confidence,
                saturation_recovery_weight * recovery_confidence,
            )
            if guidance_sizes is None:
                detector_guidance: Tensor | tuple[Tensor, ...] = torch.cat(
                    guidance_parts,
                    dim=1,
                )
            else:
                # Bilinear interpolation is independent for every channel.
                # Resize before concatenating so detector training never builds
                # a full-resolution ten-channel guidance tensor.
                detector_guidance = tuple(
                    torch.cat(
                        tuple(
                            F.interpolate(
                                part,
                                size=tuple(target_size),
                                mode="bilinear",
                                align_corners=False,
                            )
                            for part in guidance_parts
                        ),
                        dim=1,
                    )
                    for target_size in guidance_sizes
                )
            return enhanced, detector_guidance

        output_value = luminance(enhanced)
        slot_gain = slot_strength * self.slot_directions.to(slot_strength)
        bright_delta = bright_weight * (input_value - output_value).clamp_min(0.0)
        core_delta = white_core_weight * (
            input_value - output_value
        ).clamp_min(0.0)
        halo_delta = halo_total_delta
        tone_delta = output_value - input_value

        correction_strength_map = (
            dark_weight * dark_strength_map
            - torch.maximum(
                bright_weight * bright_strength_map,
                torch.maximum(
                    torch.maximum(
                        white_core_weight * white_strength_map,
                        near_light_weight * white_strength_map,
                    ),
                    halo_weight * halo_strength_map,
                ),
            )
        )
        return {
            "enhanced": enhanced,
            "input_luminance": input_value,
            "output_luminance": output_value,
            "tone_delta": tone_delta,
            "base_tone_delta": base_tone_delta,
            "dark_delta": dark_delta,
            "bright_delta": bright_delta,
            "core_delta": core_delta,
            "halo_delta": halo_delta,
            "halo_relative_delta": halo_relative_delta,
            "halo_veil_delta": halo_veil_delta,
            "halo_requested_delta": halo_requested_delta,
            "halo_total_delta": halo_total_delta,
            "halo_floor_compensation": halo_floor_compensation,
            "illumination_luminance": illumination,
            "bright_curve_scale": bright_curve_scale,
            "detail_residual": detail_residual,
            "detail_gain_maps": detail_gain_maps,
            "detail_confidence": detail_confidence,
            "recovery_rgb": recovery_rgb,
            "recovery_confidence": recovery_confidence,
            "recovery_context_rgb": context_rgb,
            "recovery_detail_residual": recovery_detail_residual,
            "saturation_recovery_weight": saturation_recovery_weight,
            "clipped_white_weight": clipped_white_weight,
            "dark_modulation": dark_modulation,
            "bright_modulation": bright_modulation,
            "white_modulation": white_modulation,
            "halo_modulation": halo_modulation,
            "correction_strength_map": correction_strength_map,
            "bright_scale": bright_scale,
            "white_scale": white_scale,
            "core_scale": core_scale,
            "halo_scale": halo_scale,
            "halo_context_excess": halo_context_excess,
            "white_score": input_white,
            "dark_weight": dark_weight,
            "bright_weight": bright_weight,
            "white_weight": white_weight,
            "white_core_weight": white_core_weight,
            "near_light_weight": near_light_weight,
            "halo_weight": halo_weight,
            "dark_mask": (
                (dark_weight > 0.50) & (base_tone_delta > 0.0)
            ).to(input_value.dtype),
            "bright_mask": (
                (bright_weight > 0.50) & (base_tone_delta < 0.0)
            ).to(input_value.dtype),
            "white_mask": (
                (white_core_weight > 0.25) & (base_tone_delta < 0.0)
            ).to(input_value.dtype),
            "halo_mask": (
                (halo_weight > 0.20) & (base_tone_delta < 0.0)
            ).to(input_value.dtype),
            "attn_map_low": attention_low,
            "attn_map_full": attention_full,
            "slot_usage": usage,
            "slot_gain": slot_gain,
            "exposure_centers": exposure_centers,
            "exposure_bin_widths": exposure_bin_widths,
            "adaptive_prior_ratio": image.new_full(
                (image.shape[0], 1),
                self.exposure_prior.adaptation_ratio,
            ),
            "exposure_prior_sigma": exposure_prior_sigmas,
        }
