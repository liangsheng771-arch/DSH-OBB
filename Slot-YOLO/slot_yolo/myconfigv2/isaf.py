"""Slot-guided Illumination-Structure Adaptive Fusion (ISAF).

ISAF does not treat enhancement as a fixed weighted sum. It decomposes both
backbone features into low-frequency semantic/tone content and high-frequency
texture content, then predicts independent spatial-and-channel gates for the
two frequency bands from detector features and Slot-DCE's internal decisions.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def build_norm(channels: int, norm: str = "bn") -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm == "gn":
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported normalization type: {norm}")


class SlotGuidanceAdapter(nn.Module):
    """Encode Slot assignment, tone, detail and recovery guidance at one scale."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels or max(16, out_channels))
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            build_norm(hidden_channels, norm),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            build_norm(hidden_channels, norm),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            build_norm(out_channels, norm),
            nn.SiLU(inplace=True),
        )

    def forward(self, guidance: Tensor, target_size: tuple[int, int]) -> Tensor:
        if guidance.shape[-2:] != target_size:
            guidance = F.interpolate(
                guidance,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return self.encoder(guidance)


class ZeroMeanDWConv2d(nn.Module):
    """Learn a depthwise high-pass kernel without carrying DC illumination."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.channels = int(channels)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, feature: Tensor) -> Tensor:
        weight = self.weight - self.weight.mean(dim=(2, 3), keepdim=True)
        return F.conv2d(feature, weight, padding=self.padding, groups=self.channels)


class LZSE(nn.Module):
    """Learned zero-DC structure encoder shared by Raw and Enhanced branches."""

    def __init__(self, channels: int, norm: str = "bn") -> None:
        super().__init__()
        self.filter = ZeroMeanDWConv2d(channels)
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = build_norm(channels, norm)

    def forward(self, feature: Tensor) -> Tensor:
        return F.silu(self.norm(self.mix(self.filter(feature))), inplace=True)


class IlluminationGuidedAlignment(nn.Module):
    """Apply a bounded, guidance-conditioned local correction to Enhanced features."""

    def __init__(
        self,
        channels: int,
        guidance_channels: int,
        scale_init: float = 0.0,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        self.guidance_gate = nn.Sequential(
            nn.Conv2d(guidance_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.local_refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            build_norm(channels, norm),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, enhanced: Tensor, guidance: Tensor) -> Tensor:
        correction = self.guidance_gate(guidance) * self.local_refine(enhanced)
        return enhanced + torch.tanh(self.scale) * correction


class ISAFGateGenerator(nn.Module):
    """Predict independent low-frequency and texture gates in space and channels."""

    def __init__(
        self,
        channels: int,
        guidance_channels: int,
        reduction: int = 4,
        norm: str = "bn",
        tone_gate_init: float = 0.20,
        texture_gate_init: float = 0.35,
    ) -> None:
        super().__init__()
        for name, value in (
            ("tone_gate_init", tone_gate_init),
            ("texture_gate_init", texture_gate_init),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")

        spatial_hidden = max(16, min(64, channels // max(int(reduction), 1)))
        self.spatial_predictor = nn.Sequential(
            nn.Conv2d(
                guidance_channels + 7,
                spatial_hidden,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            build_norm(spatial_hidden, norm),
            nn.SiLU(inplace=True),
            nn.Conv2d(spatial_hidden, 2, kernel_size=1, bias=True),
        )

        channel_hidden = max(16, channels // max(int(reduction), 1))
        self.channel_predictor = nn.Sequential(
            # Pool each branch before concatenation. This is exactly equal to
            # GAP(concat(...)) but avoids materializing a 4C full-resolution tensor.
            nn.Identity(),
            nn.Conv2d(4 * channels, channel_hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_hidden, 2 * channels, kernel_size=1, bias=True),
        )
        self.base_logits = nn.Parameter(
            torch.tensor(
                (
                    math.log(tone_gate_init / (1.0 - tone_gate_init)),
                    math.log(texture_gate_init / (1.0 - texture_gate_init)),
                ),
                dtype=torch.float32,
            ).view(1, 2, 1, 1)
        )

        nn.init.normal_(self.spatial_predictor[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.spatial_predictor[-1].bias)
        nn.init.normal_(self.channel_predictor[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.channel_predictor[-1].bias)

    def forward(
        self,
        raw_low: Tensor,
        enhanced_low: Tensor,
        raw_detail: Tensor,
        enhanced_detail: Tensor,
        raw_structure: Tensor,
        enhanced_structure: Tensor,
        guidance: Tensor,
    ) -> tuple[Tensor, Tensor]:
        spatial_evidence = torch.cat(
            (
                raw_low.abs().mean(dim=1, keepdim=True),
                enhanced_low.abs().mean(dim=1, keepdim=True),
                (enhanced_low - raw_low).abs().mean(dim=1, keepdim=True),
                raw_detail.abs().mean(dim=1, keepdim=True),
                enhanced_detail.abs().mean(dim=1, keepdim=True),
                (enhanced_detail - raw_detail).abs().mean(dim=1, keepdim=True),
                (enhanced_structure - raw_structure).abs().mean(dim=1, keepdim=True),
                guidance,
            ),
            dim=1,
        )
        spatial_logits = self.spatial_predictor(spatial_evidence)
        pooled_channel_evidence = torch.cat(
            tuple(
                F.adaptive_avg_pool2d(value, 1)
                for value in (raw_low, enhanced_low, raw_detail, enhanced_detail)
            ),
            dim=1,
        )
        channel_logits = self.channel_predictor(pooled_channel_evidence)
        tone_channel, texture_channel = channel_logits.chunk(2, dim=1)
        base_logits = self.base_logits.to(spatial_logits)
        tone_gate = torch.sigmoid(
            base_logits[:, 0:1] + spatial_logits[:, 0:1] + tone_channel
        )
        texture_gate = torch.sigmoid(
            base_logits[:, 1:2] + spatial_logits[:, 1:2] + texture_channel
        )
        return tone_gate, texture_gate


class TextureRefinement(nn.Module):
    """Refine selected high-frequency content under a bounded residual scale."""

    def __init__(self, channels: int, norm: str = "bn", scale_init: float = 0.0) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            build_norm(channels, norm),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, detail: Tensor) -> Tensor:
        return detail + torch.tanh(self.scale) * self.refine(detail)


class ISAF(nn.Module):
    """Fuse Slot-DCE features through semantic/tone and texture paths."""

    def __init__(
        self,
        channels: int,
        illum_channels: int,
        reduction: int = 4,
        norm: str = "bn",
        tone_gate_init: float = 0.20,
        texture_gate_init: float = 0.35,
        alignment_scale_init: float = 0.0,
        texture_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        # Both streams use the same per-sample normalization in train and eval.
        # A shared BN would mix raw/enhanced running statistics while training
        # each stream with separate batch statistics, biasing their comparison.
        self.structure_extractor = LZSE(channels, norm="gn")
        self.enh_alignment = IlluminationGuidedAlignment(
            channels,
            illum_channels,
            scale_init=alignment_scale_init,
            norm=norm,
        )
        self.gate_generator = ISAFGateGenerator(
            channels,
            illum_channels,
            reduction=reduction,
            norm=norm,
            tone_gate_init=tone_gate_init,
            texture_gate_init=texture_gate_init,
        )
        self.texture_refinement = TextureRefinement(
            channels,
            norm=norm,
            scale_init=texture_scale_init,
        )

    @staticmethod
    def decompose(feature: Tensor) -> tuple[Tensor, Tensor]:
        low = F.avg_pool2d(
            feature,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        return low, feature - low

    def forward(
        self,
        f_raw: Tensor,
        f_enhanced: Tensor,
        f_guidance: Tensor,
        return_aux: bool = False,
    ):
        if f_raw.ndim != 4 or f_enhanced.ndim != 4:
            raise ValueError("ISAF expects raw/enhanced tensors in [B,C,H,W] format")
        if f_raw.shape != f_enhanced.shape:
            raise ValueError(
                "ISAF requires identical raw/enhanced shapes, got "
                f"{tuple(f_raw.shape)} and {tuple(f_enhanced.shape)}"
            )
        if f_guidance.shape[-2:] != f_raw.shape[-2:]:
            f_guidance = F.interpolate(
                f_guidance,
                size=f_raw.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        aligned_enhanced = self.enh_alignment(f_enhanced, f_guidance)
        raw_low, raw_detail = self.decompose(f_raw)
        enhanced_low, enhanced_detail = self.decompose(aligned_enhanced)
        raw_structure = self.structure_extractor(f_raw)
        enhanced_structure = self.structure_extractor(aligned_enhanced)
        tone_gate, texture_gate = self.gate_generator(
            raw_low,
            enhanced_low,
            raw_detail,
            enhanced_detail,
            raw_structure,
            enhanced_structure,
            f_guidance,
        )

        fused_low = torch.lerp(raw_low, enhanced_low, tone_gate)
        fused_detail = torch.lerp(raw_detail, enhanced_detail, texture_gate)
        refined_detail = self.texture_refinement(fused_detail)
        output = fused_low + refined_detail
        if not return_aux:
            return output
        return output, {
            "raw_low": raw_low,
            "enhanced_low": enhanced_low,
            "raw_detail": raw_detail,
            "enhanced_detail": enhanced_detail,
            "raw_structure": raw_structure,
            "enhanced_structure": enhanced_structure,
            "f_guidance": f_guidance,
            "f_enh_aligned": aligned_enhanced,
            "tone_gate": tone_gate,
            "texture_gate": texture_gate,
            "w_raw": 1.0 - tone_gate,
            "w_enh": tone_gate,
        }


__all__ = [
    "ISAF",
    "ISAFGateGenerator",
    "IlluminationGuidedAlignment",
    "LZSE",
    "SlotGuidanceAdapter",
    "TextureRefinement",
    "ZeroMeanDWConv2d",
]
