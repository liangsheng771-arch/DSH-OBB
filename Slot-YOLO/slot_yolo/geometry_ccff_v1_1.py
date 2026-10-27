"""Geometry-aware bidirectional CCFF for the scale-preserving HybridEncoder.

The P5(AIFI) -> P4 -> P3(native CCFF) -> P3(OBB geometry) -> P4 -> P5
order is intentional. This module owns only the selected CCFF parameters;
the existing AIFI, backbone and detection head are constructed elsewhere.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .geometry_ccff import GeometryGuidedP3Fusion, _sample_axis_support
from .rtdetr.hybrid_encoder import CSPRepLayer, ConvNormLayer


def _expect(feature: Tensor, channels: int, height: int, width: int, name: str) -> None:
    if feature.ndim != 4 or feature.shape[1:] != (channels, height, width):
        raise ValueError(
            f"{name} must be [B,{channels},{height},{width}], got {tuple(feature.shape)}"
        )


class SemanticStructureP5P4(nn.Module):
    """Select AIFI P5 semantics using P4 structure before CSPRep fusion."""

    def __init__(self, blocks: int, expansion: float, activation: str) -> None:
        super().__init__()
        self.semantic_p5 = ConvNormLayer(512, 256, 1, 1, act=activation)
        self.structure_p4 = nn.Sequential(
            ConvNormLayer(256, 256, 3, 1, act=activation),
            ConvNormLayer(256, 256, 3, 1, act=activation),
        )
        self.semantic_selector = nn.Sequential(
            ConvNormLayer(512, 256, 1, 1, act=activation),
            nn.Conv2d(256, 256, 1),
            nn.Sigmoid(),
        )
        self.fusion = CSPRepLayer(768, 256, blocks, expansion=expansion, act=activation)

    def forward(self, p5: Tensor, p4: Tensor) -> Tensor:
        semantic = F.interpolate(
            self.semantic_p5(p5), size=p4.shape[-2:], mode="bilinear", align_corners=False,
        )
        structure = self.structure_p4(p4)
        selected = self.semantic_selector(torch.cat((structure, semantic), dim=1)) * semantic
        contrast = structure - selected
        fused = self.fusion(torch.cat((structure, selected, contrast), dim=1))
        _expect(fused, 256, *p4.shape[-2:], "P4_td")
        return fused


class GeometryGuidedP3FusionV11(GeometryGuidedP3Fusion):
    """Reuse the original OBB field, sampler and auxiliary loss with a learned gate."""

    def __init__(self) -> None:
        super().__init__(
            channels=128, coarse_channels=256,
            use_axis_sampling=True, use_shape_routing=True,
            use_geometry_supervision=True, aux_loss_weight=0.2, area_scale=64.0,
        )
        self.geometry_router = nn.Sequential(
            nn.Conv2d(2, 16, 1), nn.SiLU(), nn.Conv2d(16, 1, 1), nn.Sigmoid(),
        )

    def forward(self, native: Tensor, p4_td: Tensor) -> tuple[Tensor, Tensor]:
        expected_coarse = tuple(size // 2 for size in native.shape[-2:])
        if native.shape[0] != p4_td.shape[0] or p4_td.shape[-2:] != expected_coarse:
            raise ValueError("Geometry v1.1 requires P4_td at half the P3_native resolution")

        field = self.field_head(native)
        heat, raw_axis, log_major, log_minor, route = field.split((1, 2, 1, 1, 1), dim=1)
        proposed_major = log_major.float().clamp(math.log(0.5), math.log(32.0)).exp()
        proposed_minor = log_minor.float().clamp(math.log(0.5), math.log(32.0)).exp()
        major = torch.maximum(proposed_major, proposed_minor)
        minor = torch.minimum(proposed_major, proposed_minor)

        # The geometry sampler reads P4_td, not the original P4 lateral input.
        semantic_coarse = self.semantic_reduce(p4_td)
        semantic = F.interpolate(
            semantic_coarse, size=native.shape[-2:], mode="bilinear", align_corners=False,
        )
        along, across = _sample_axis_support(
            semantic_coarse, raw_axis, major, minor, native.shape[-2:],
        )
        correction = self.refine(torch.cat(
            (native, semantic, along, along - across), dim=1,
        ))

        foreground = heat.float().sigmoid() * route.float().sigmoid()
        anisotropy = (major - minor) / (major + minor + 1e-6)
        smallness = self.area_scale / (self.area_scale + major * minor)
        descriptor = torch.cat((anisotropy, smallness), dim=1)
        geometry = self.geometry_router(descriptor).float()
        gate = foreground * (1.0 + geometry)

        self._last_field = field if self.training else None
        output = native + gate.to(correction.dtype) * correction
        _expect(output, 128, *native.shape[-2:], "P3_out")
        return output, gate


class GeometryFeedbackP3P4(nn.Module):
    """Preserve full P3 details and also feed geometry-routed details to P4."""

    def __init__(self, blocks: int, expansion: float, activation: str) -> None:
        super().__init__()
        self.downsample_p3 = nn.Sequential(
            ConvNormLayer(128, 256, 3, 2, act=activation),
            ConvNormLayer(256, 256, 3, 1, act=activation),
        )
        self.fusion = CSPRepLayer(768, 256, blocks, expansion=expansion, act=activation)

    def forward(self, p4_td: Tensor, p3_out: Tensor, gate: Tensor) -> Tensor:
        detail = self.downsample_p3(p3_out)
        _expect(detail, 256, *p4_td.shape[-2:], "D3")
        if gate.shape != (p3_out.shape[0], 1, *p3_out.shape[-2:]):
            raise ValueError("P3 geometry gate must be [B,1,H3,W3]")
        routed_gate = F.avg_pool2d(gate, kernel_size=2, stride=2)
        _expect(routed_gate, 1, *p4_td.shape[-2:], "G4")
        routed_detail = detail * routed_gate.to(detail.dtype)
        output = self.fusion(torch.cat((p4_td, detail, routed_detail), dim=1))
        _expect(output, 256, *p4_td.shape[-2:], "P4_out")
        return output


class SelectiveFeedbackP4P5(nn.Module):
    """Select bottom-up P4 semantics before a learned P5 residual update."""

    def __init__(self, activation: str) -> None:
        super().__init__()
        self.downsample_p4 = nn.Sequential(
            ConvNormLayer(256, 512, 3, 2, act=activation),
            ConvNormLayer(512, 512, 3, 1, act=activation),
        )
        self.semantic_selector = nn.Sequential(
            ConvNormLayer(1024, 512, 1, 1, act=activation),
            nn.Conv2d(512, 512, 1),
            nn.Sigmoid(),
        )
        self.feedback = nn.Sequential(
            ConvNormLayer(512, 512, 3, 1, act=activation),
            nn.Conv2d(512, 512, 1),
        )

    def forward(self, p5_aifi: Tensor, p4_out: Tensor) -> Tensor:
        detail = self.downsample_p4(p4_out)
        _expect(detail, 512, *p5_aifi.shape[-2:], "D4")
        selected = self.semantic_selector(torch.cat((p5_aifi, detail), dim=1)) * detail
        output = p5_aifi + self.feedback(selected)
        _expect(output, 512, *p5_aifi.shape[-2:], "P5_out")
        return output


class GeometryAwareBiCCFF(nn.Module):
    """Fixed 128/256/512 bidirectional CCFF with one OBB geometry field at P3."""

    def __init__(
        self, num_blocks: int = 3, expansion: float = 0.5, activation: str = "silu",
    ) -> None:
        super().__init__()
        self.top_down_p4 = SemanticStructureP5P4(num_blocks, expansion, activation)
        # Retain the project's native P4 -> P3 lateral + CSPRep node.
        self.lateral_p4 = ConvNormLayer(256, 256, 1, 1, act=activation)
        self.native_p3 = CSPRepLayer(384, 128, num_blocks, expansion=expansion, act=activation)
        self.geometry_p3 = GeometryGuidedP3FusionV11()
        self.bottom_up_p4 = GeometryFeedbackP3P4(num_blocks, expansion, activation)
        self.bottom_up_p5 = SelectiveFeedbackP4P5(activation)

    def forward(
        self, p3_lateral: Tensor, p4_lateral: Tensor, p5_aifi: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        p3, p4, p5 = p3_lateral, p4_lateral, p5_aifi
        if any(value.ndim != 4 for value in (p3, p4, p5)):
            raise ValueError("Geometry v1.1 expects three BCHW feature tensors")
        h3, w3 = p3.shape[-2:]
        if h3 % 4 or w3 % 4:
            raise ValueError("Geometry v1.1 requires P3 spatial dimensions divisible by four")
        _expect(p3, 128, h3, w3, "P3_lateral")
        _expect(p4, 256, h3 // 2, w3 // 2, "P4_lateral")
        _expect(p5, 512, h3 // 4, w3 // 4, "P5_AIFI")
        if len({value.shape[0] for value in (p3, p4, p5)}) != 1:
            raise ValueError("Geometry v1.1 requires the same batch size at all scales")
        if len({value.device for value in (p3, p4, p5)}) != 1:
            raise ValueError("Geometry v1.1 requires matching feature devices")

        p4_td = self.top_down_p4(p5, p4)
        p4_for_p3 = self.lateral_p4(p4_td)
        up4 = F.interpolate(p4_for_p3, size=p3.shape[-2:], mode="nearest")
        p3_native = self.native_p3(torch.cat((up4, p3), dim=1))
        _expect(p3_native, 128, h3, w3, "P3_native")
        p3_out, gate = self.geometry_p3(p3_native, p4_td)
        p4_out = self.bottom_up_p4(p4_td, p3_out, gate)
        p5_out = self.bottom_up_p5(p5, p4_out)
        # IGLSFAIFI may retain FP32 P5 residuals under detector AMP while
        # P3/P4 convolutions produce FP16 or BF16. Keep the neck/head pyramid
        # on the bottom-up convolution dtype after the FP32 residual update.
        output_dtype = p4_out.dtype
        return p3_out.to(output_dtype), p4_out, p5_out.to(output_dtype)


__all__ = [
    "SemanticStructureP5P4", "GeometryGuidedP3FusionV11",
    "GeometryFeedbackP3P4", "SelectiveFeedbackP4P5", "GeometryAwareBiCCFF",
]
