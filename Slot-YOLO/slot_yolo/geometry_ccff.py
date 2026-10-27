"""OBB-supervised, instance-shape-routed cross-scale fusion at P3.

The native CCFF produces its ordinary P3 feature first. This module predicts
an orientation/size field from that feature, uses it to sample P4 semantics
along and across the proposed major axis, and adds a foreground-routed
correction. OBB supervision makes the sampling geometry identifiable instead
of asking the detection loss to discover it indirectly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _sample_axis_support(
    semantic: Tensor, axis: Tensor, major: Tensor, minor: Tensor,
    target_hw: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    """Sample symmetric in-object and lateral evidence in FP32 under AMP."""
    batch, channels = semantic.shape[:2]
    height, width = target_hw
    with torch.autocast(device_type=semantic.device.type, enabled=False):
        source = semantic.float()
        direction = F.normalize(axis.float(), dim=1, eps=1e-6)
        theta = 0.5 * torch.atan2(direction[:, 1:2], direction[:, 0:1])
        dx, dy = theta.cos(), theta.sin()
        long_radius = (0.25 * major.float()).clamp(0.5, 4.0)
        side_radius = (0.5 * minor.float()).clamp(0.5, 2.0)
        xx, yy = torch.meshgrid(
            torch.arange(width, device=source.device, dtype=torch.float32) + 0.5,
            torch.arange(height, device=source.device, dtype=torch.float32) + 0.5,
            indexing="xy",
        )
        base_x, base_y = xx[None, None], yy[None, None]
        offsets = (
            (long_radius * dx, long_radius * dy),
            (-long_radius * dx, -long_radius * dy),
            (-side_radius * dy, side_radius * dx),
            (side_radius * dy, -side_radius * dx),
        )
        grids = [torch.stack(
            (2.0 * (base_x + offset_x).squeeze(1) / width - 1.0,
             2.0 * (base_y + offset_y).squeeze(1) / height - 1.0),
            dim=-1,
        ) for offset_x, offset_y in offsets]
        grid = torch.cat(grids, dim=1)
        samples = F.grid_sample(
            source, grid, mode="bilinear", padding_mode="border", align_corners=False,
        ).reshape(batch, channels, 4, height, width)
        along = (samples[:, :, 0] + samples[:, :, 1]) * 0.5
        across = (samples[:, :, 2] + samples[:, :, 3]) * 0.5
    return along.to(semantic.dtype), across.to(semantic.dtype)


def obb_geometry_targets(
    batch: Mapping[str, Tensor], shape: tuple[int, int, int],
    *, area_scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Rasterize class-agnostic OBB center support and major-axis geometry."""
    batch_size, height, width = shape
    boxes = batch["bboxes"].detach().float()
    image_indices = batch["batch_idx"].detach().long().flatten()
    if boxes.ndim != 2 or boxes.shape[-1] != 5 or image_indices.numel() != boxes.shape[0]:
        raise ValueError("Geometry CCFF expects OBB targets as Nx5 xywhr and N batch indices")
    device = boxes.device
    heat = torch.zeros(batch_size, 1, height, width, device=device, dtype=torch.float32)
    axis = torch.zeros(batch_size, 2, height, width, device=device, dtype=torch.float32)
    log_major = torch.zeros_like(heat)
    log_minor = torch.zeros_like(heat)
    importance = torch.zeros_like(heat)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32) + 0.5,
        torch.arange(width, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    for image_index in range(batch_size):
        current = boxes[image_indices == image_index]
        if current.numel() == 0:
            continue
        centers_x = current[:, 0] * width
        centers_y = current[:, 1] * height
        box_w = current[:, 2] * width
        box_h = current[:, 3] * height
        valid = (box_w > 0) & (box_h > 0)
        if not valid.any():
            continue
        centers_x, centers_y = centers_x[valid], centers_y[valid]
        box_w, box_h = box_w[valid], box_h[valid]
        angles = current[valid, 4] + torch.where(
            box_h > box_w, math.pi / 2, 0.0,
        )
        major = torch.maximum(box_w, box_h)
        minor = torch.minimum(box_w, box_h)
        delta_x = xx[None] - centers_x[:, None, None]
        delta_y = yy[None] - centers_y[:, None, None]
        cosine, sine = angles.cos()[:, None, None], angles.sin()[:, None, None]
        longitudinal = delta_x * cosine + delta_y * sine
        transverse = -delta_x * sine + delta_y * cosine
        sigma_major = (major / 4).clamp_min(0.75)[:, None, None]
        sigma_minor = (minor / 4).clamp_min(0.5)[:, None, None]
        gaussian = torch.exp(-0.5 * (
            (longitudinal / sigma_major).square()
            + (transverse / sigma_minor).square()
        ))
        strength, owner = gaussian.max(dim=0)
        anisotropy = (major - minor) / (major + minor + 1e-6)
        size_weight = area_scale / (area_scale + major * minor)
        heat[image_index, 0] = strength
        axis[image_index, 0] = (2 * angles).cos()[owner]
        axis[image_index, 1] = (2 * angles).sin()[owner]
        log_major[image_index, 0] = major.clamp_min(0.5).log()[owner]
        log_minor[image_index, 0] = minor.clamp_min(0.5).log()[owner]
        importance[image_index, 0] = strength * anisotropy[owner] * size_weight[owner]
    return heat, axis, log_major, log_minor, importance


class GeometryGuidedP3Fusion(nn.Module):
    """Preserve native P3 and route aligned P4 semantics by predicted OBB shape."""

    def __init__(
        self, channels: int = 128, coarse_channels: int = 256, *,
        use_axis_sampling: bool = True,
        use_shape_routing: bool = True,
        use_geometry_supervision: bool = True,
        aux_loss_weight: float = 0.2,
        area_scale: float = 64.0,
    ) -> None:
        super().__init__()
        self.use_axis_sampling = use_axis_sampling
        self.use_shape_routing = use_shape_routing
        self.use_geometry_supervision = use_geometry_supervision
        self.aux_loss_weight = aux_loss_weight if use_geometry_supervision else 0.0
        self.area_scale = area_scale
        self.field_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, 6, 1),
        )
        nn.init.normal_(self.field_head[-1].weight, std=0.01)
        with torch.no_grad():
            self.field_head[-1].bias.copy_(torch.tensor(
                [-2.0, 1.0, 0.0, math.log(6.0), math.log(2.0), 0.0]
            ))
        self.semantic_reduce = nn.Conv2d(coarse_channels, channels, 1, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(4 * channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        nn.init.zeros_(self.refine[-1].weight)
        self._last_field: Tensor | None = None

    def forward(self, native: Tensor, coarse: Tensor) -> Tensor:
        expected_coarse = tuple((size + 1) // 2 for size in native.shape[-2:])
        if native.shape[0] != coarse.shape[0] or coarse.shape[-2:] != expected_coarse:
            raise ValueError("Geometry CCFF needs matching P3/P4 batch and pyramid sizes")
        field = self.field_head(native)
        heat, raw_axis, log_major, log_minor, route = field.split((1, 2, 1, 1, 1), dim=1)
        proposed_major = log_major.float().clamp(math.log(0.5), math.log(32.0)).exp()
        proposed_minor = log_minor.float().clamp(math.log(0.5), math.log(32.0)).exp()
        major = torch.maximum(proposed_major, proposed_minor)
        minor = torch.minimum(proposed_major, proposed_minor)
        semantic_coarse = self.semantic_reduce(coarse)
        semantic = F.interpolate(
            semantic_coarse, size=native.shape[-2:], mode="bilinear", align_corners=False,
        )
        if self.use_axis_sampling:
            along, across = _sample_axis_support(
                semantic_coarse, raw_axis, major, minor, native.shape[-2:],
            )
            contrast = along - across
        else:
            along, contrast = semantic, torch.zeros_like(semantic)
        correction = self.refine(torch.cat((native, semantic, along, contrast), dim=1))
        gate = heat.float().sigmoid() * route.float().sigmoid()
        if self.use_shape_routing:
            anisotropy = (major - minor) / (major + minor + 1e-6)
            smallness = self.area_scale / (self.area_scale + major * minor)
            gate = gate * anisotropy * smallness
        if self.training:
            self._last_field = field
        else:
            self._last_field = None
        return native + gate.to(correction.dtype) * correction

    def auxiliary_loss(self, batch: Mapping[str, Tensor]) -> Tensor:
        if self._last_field is None:
            raise RuntimeError("Geometry CCFF auxiliary loss requires a preceding training forward")
        field = self._last_field.float()
        if not self.use_geometry_supervision:
            return field.sum() * 0.0
        heat, axis_raw, log_major, log_minor, _ = field.split((1, 2, 1, 1, 1), dim=1)
        target_heat, target_axis, target_major, target_minor, importance = obb_geometry_targets(
            batch, (field.shape[0], *field.shape[-2:]), area_scale=self.area_scale,
        )
        # Thin OBBs occupy only a few P3 cells. A spatial mean would let the
        # background dominate and collapse the predicted foreground gate.
        positive_mass = target_heat.sum()
        negative_mass = (1.0 - target_heat).sum().clamp_min(1.0)
        heat_loss = (
            -(target_heat * F.logsigmoid(heat)).sum()
            / positive_mass.clamp_min(1.0)
            - ((1.0 - target_heat) * F.logsigmoid(-heat)).sum()
            / negative_mass
        )
        denominator = importance.sum().clamp_min(1.0)
        axis = F.normalize(axis_raw, dim=1, eps=1e-6)
        axis_loss = ((1.0 - (axis * target_axis).sum(dim=1, keepdim=True))
                     * importance).sum() / denominator
        size_loss = (
            F.smooth_l1_loss(log_major, target_major, reduction="none")
            + F.smooth_l1_loss(log_minor, target_minor, reduction="none")
        ).mul(importance).sum() / denominator
        return self.aux_loss_weight * (heat_loss + axis_loss + 0.25 * size_loss)

    def clear_cache(self) -> None:
        self._last_field = None


__all__ = ["GeometryGuidedP3Fusion", "obb_geometry_targets", "_sample_axis_support"]
