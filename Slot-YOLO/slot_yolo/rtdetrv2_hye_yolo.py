"""Scale-preserving native AIFI/CCFF neck for YOLOv8s feature widths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import json

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rtdetrv2_hye_config import (
    NECK_NAME,
    normalize_rtdetrv2_hye_yolo_config,
)
from .rtdetr.hybrid_encoder import (
    CSPRepLayer,
    ConvNormLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


def _expected_downsample(size: tuple[int, int]) -> tuple[int, int]:
    return tuple((value + 1) // 2 for value in size)


def _check_pyramid(
    features: Sequence[Tensor],
    channels: Sequence[int],
    *,
    context: str,
    reference: Sequence[Tensor] | None = None,
) -> None:
    if len(features) != 3:
        raise ValueError(f"{context}: expected exactly three C3/C4/C5 tensors")
    for index, (feature, expected_channels) in enumerate(zip(features, channels)):
        level = index + 3
        if not isinstance(feature, Tensor) or feature.ndim != 4:
            raise ValueError(f"{context}: C{level} must be a BCHW tensor")
        if feature.shape[1] != expected_channels:
            raise ValueError(
                f"{context}: C{level} expected {expected_channels} channels, got {feature.shape[1]}"
            )
        if feature.shape[0] != features[0].shape[0]:
            raise ValueError(f"{context}: C3/C4/C5 batch sizes must match")
        if feature.device != features[0].device or feature.dtype != features[0].dtype:
            raise ValueError(f"{context}: C3/C4/C5 devices and dtypes must match")
        if index:
            expected_size = _expected_downsample(tuple(features[index - 1].shape[-2:]))
            actual_size = tuple(feature.shape[-2:])
            if actual_size != expected_size:
                raise ValueError(
                    f"{context}: C{level} spatial size must be {expected_size} after a "
                    f"3x3 stride-2 downsample, got {actual_size}"
                )
        if reference is not None and (
            feature.shape[0] != reference[index].shape[0]
            or feature.shape[-2:] != reference[index].shape[-2:]
        ):
            raise ValueError(
                f"{context}: P{level} batch/spatial shape must match input C{level}"
            )


class RTDETRV2_hyE_yolo(nn.Module):
    """P5-512 native AIFI plus scale-width native CCFF for YOLOv8s."""

    neck_name = NECK_NAME
    architecture_version = 1

    def __init__(
        self,
        in_channels: Sequence[int] = (128, 256, 512),
        feat_strides: Sequence[int] = (8, 16, 32),
        *,
        aifi: Mapping[str, Any] | None = None,
        ccff: Mapping[str, Any] | None = None,
        use_ig_lsf_aifi: bool = False,
        use_wavelet_ccff: bool = False,
        ig_lsf_aifi: Mapping[str, Any] | None = None,
        use_sga_ccff: bool = False,
        sga_ccff: Mapping[str, Any] | None = None,
        use_selective_ccff: bool = False,
        selective_ccff: Mapping[str, Any] | None = None,
        use_geometry_ccff: bool = False,
        geometry_ccff: Mapping[str, Any] | None = None,
        use_geometry_ccff_v1_1: bool = False,
    ) -> None:
        super().__init__()
        config = normalize_rtdetrv2_hye_yolo_config(
            {
                "in_channels": list(in_channels),
                "feat_strides": list(feat_strides),
                "aifi": dict(aifi or {}),
                "ccff": dict(ccff or {}),
                "use_ig_lsf_aifi": use_ig_lsf_aifi,
                "use_wavelet_ccff": use_wavelet_ccff,
                "ig_lsf_aifi": dict(ig_lsf_aifi or {}),
                "use_sga_ccff": use_sga_ccff,
                "sga_ccff": dict(sga_ccff or {}),
                "use_selective_ccff": use_selective_ccff,
                "selective_ccff": dict(selective_ccff or {}),
                "use_geometry_ccff": use_geometry_ccff,
                "geometry_ccff": dict(geometry_ccff or {}),
                "use_geometry_ccff_v1_1": use_geometry_ccff_v1_1,
            }
        )
        self.architecture_config = config
        self.use_ig_lsf_aifi = config["use_ig_lsf_aifi"]
        self.use_wavelet_ccff = config["use_wavelet_ccff"]
        self.use_sga_ccff = config["use_sga_ccff"]
        self.use_selective_ccff = config["use_selective_ccff"]
        self.use_geometry_ccff = config["use_geometry_ccff"]
        self.use_geometry_ccff_v1_1 = config["use_geometry_ccff_v1_1"]
        self.required_guidance_stages = (2,) if self.use_ig_lsf_aifi else ()
        self.architecture_version = 2 if self.use_ig_lsf_aifi else 1
        if self.use_geometry_ccff:
            self.architecture_version = 5
        if self.use_geometry_ccff_v1_1:
            self.architecture_version = 6
        if self.architecture_version >= 2:
            metadata = json.dumps(config, sort_keys=True, allow_nan=False).encode("utf-8")
            self.register_buffer("_architecture_metadata", torch.tensor(list(metadata), dtype=torch.uint8))
        self.input_channels = tuple(config["in_channels"])
        self.output_channels = self.input_channels
        self.feat_strides = tuple(config["feat_strides"])
        aifi_config = config["aifi"]
        ccff_config = config["ccff"]
        d_model = self.input_channels[-1]

        layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=aifi_config["nhead"],
            dim_feedforward=aifi_config["dim_feedforward"],
            dropout=aifi_config["dropout"],
            activation=aifi_config["activation"],
            normalize_before=aifi_config["normalize_before"],
        )
        self.aifi = TransformerEncoder(layer, aifi_config["num_encoder_layers"])
        self.pe_temperature = float(aifi_config["pe_temperature"])
        if self.use_ig_lsf_aifi:
            from .ig_lsf_aifi import IGLSFAIFI

            # Preserve the native encoder's initialization and the RNG stream
            # seen by CCFF across AIFI ablations. Only one encoder is registered.
            with torch.random.fork_rng(devices=[]):
                improved = IGLSFAIFI(d_model, aifi=aifi_config, settings=config["ig_lsf_aifi"])
            improved.native_aifi = self.aifi
            self.aifi = improved

        if self.use_geometry_ccff_v1_1:
            from .geometry_ccff_v1_1 import GeometryAwareBiCCFF

            self.ccff = GeometryAwareBiCCFF(**ccff_config)
            return  # The new module owns only the selected bidirectional CCFF.

        blocks = ccff_config["num_blocks"]
        expansion = ccff_config["expansion"]
        activation = ccff_config["activation"]
        self.lateral_p5 = ConvNormLayer(512, 512, 1, 1, act=activation)
        self.fpn_p4 = CSPRepLayer(768, 256, blocks, expansion=expansion, act=activation)
        self.lateral_p4 = ConvNormLayer(256, 256, 1, 1, act=activation)
        self.fpn_p3 = CSPRepLayer(384, 128, blocks, expansion=expansion, act=activation)
        self.downsample_p3 = ConvNormLayer(128, 128, 3, 2, padding=1, act=activation)
        self.pan_p4 = CSPRepLayer(384, 256, blocks, expansion=expansion, act=activation)
        self.downsample_p4 = ConvNormLayer(256, 256, 3, 2, padding=1, act=activation)
        self.pan_p5 = CSPRepLayer(768, 512, blocks, expansion=expansion, act=activation)

        if self.use_geometry_ccff:
            from .geometry_ccff import GeometryGuidedP3Fusion

            self.geometry_p3 = GeometryGuidedP3Fusion(**config["geometry_ccff"])

    @staticmethod
    def build_2d_sincos_position_embedding(
        height: int,
        width: int,
        embed_dim: int = 512,
        temperature: float = 10000.0,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Build H-major/W-minor positions matching BCHW flatten(2)."""

        if embed_dim % 4:
            raise ValueError("2D sin/cos position embedding width must be divisible by 4")
        if height <= 0 or width <= 0:
            raise ValueError("position embedding height and width must be positive")
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, device=device, dtype=torch.float32) / pos_dim
        omega = temperature ** (-omega)
        x = x.reshape(-1, 1) * omega.reshape(1, -1)
        y = y.reshape(-1, 1) * omega.reshape(1, -1)
        position = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        return position.unsqueeze(0).to(dtype=dtype)

    def validate_architecture(self, settings: Mapping[str, Any] | None = None) -> None:
        improved = getattr(self, "use_ig_lsf_aifi", False)
        version = (6 if getattr(self, "use_geometry_ccff_v1_1", False) else
                   5 if getattr(self, "use_geometry_ccff", False) else
                   2 if improved else 1)
        if getattr(self, "architecture_version", None) != version:
            raise ValueError(
                "RTDETRV2_hyE_yolo checkpoint architecture version is incompatible; "
                "use matching weights or train with resume=false"
            )
        if settings is None:
            return
        expected = normalize_rtdetrv2_hye_yolo_config(
            settings,
            actual_in_channels=self.input_channels,
            actual_feat_strides=self.feat_strides,
        )
        if expected != normalize_rtdetrv2_hye_yolo_config(self.architecture_config):
            raise ValueError(
                "Checkpoint RTDETRV2_hyE_yolo settings differ from the current config; "
                "changing neck construction requires train.resume=false"
            )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if getattr(self, "architecture_version", 1) >= 2:
            value = state_dict.get(prefix + "_architecture_metadata")
            if not isinstance(value, Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
                raise RuntimeError("Missing/invalid improved neck architecture metadata; use matching ablation weights")
            saved = json.loads(bytes(value.detach().cpu().tolist()).decode("utf-8"))
            if normalize_rtdetrv2_hye_yolo_config(saved) != normalize_rtdetrv2_hye_yolo_config(self.architecture_config):
                raise RuntimeError("Neck checkpoint architecture/config mismatch; use matching ablation weights")
            # Older IG/wavelet metadata predates the disabled SGA fields. Its
            # normalized architecture is identical, but serialized length differs.
            state_dict[prefix + "_architecture_metadata"] = self._architecture_metadata.detach().clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    @staticmethod
    def _require_spatial_match(value: Tensor, reference: Tensor, operation: str) -> None:
        if value.shape[-2:] != reference.shape[-2:]:
            raise ValueError(
                f"RTDETRV2_hyE_yolo {operation} produced spatial size "
                f"{tuple(value.shape[-2:])}, expected {tuple(reference.shape[-2:])}"
            )

    def forward(self, c3: Tensor, c4: Tensor, c5: Tensor, *, guidance=None) -> tuple[Tensor, Tensor, Tensor]:
        self.validate_architecture()
        inputs = (c3, c4, c5)
        _check_pyramid(inputs, self.input_channels, context="RTDETRV2_hyE_yolo input")

        if getattr(self, "use_ig_lsf_aifi", False):
            if not isinstance(guidance, (tuple, list)) or len(guidance) != 3 or guidance[2] is None:
                raise ValueError("IGLSFAIFI neck requires raw SlotDCE guidance at P5 (guidance[2])")
            a5 = self.aifi(c5, guidance[2])
        else:
            height, width = c5.shape[-2:]
            spatial_tokens = c5.flatten(2).transpose(1, 2)
            position = self.build_2d_sincos_position_embedding(
                height, width, c5.shape[1], self.pe_temperature,
                device=c5.device, dtype=c5.dtype,
            )
            encoded_tokens = self.aifi(spatial_tokens, pos_embed=position)
            a5 = encoded_tokens.transpose(1, 2).reshape_as(c5).contiguous()

        if getattr(self, "use_geometry_ccff_v1_1", False):
            outputs = self.ccff(c3, c4, a5)
            _check_pyramid(outputs, self.output_channels, context=type(self.ccff).__name__ + " output", reference=inputs)
            return outputs

        l5 = self.lateral_p5(a5)
        up5 = F.interpolate(l5, size=c4.shape[-2:], mode="nearest")
        t4 = self.fpn_p4(torch.cat((up5, c4), dim=1))
        l4 = self.lateral_p4(t4)
        up4 = F.interpolate(l4, size=c3.shape[-2:], mode="nearest")
        t3 = self.fpn_p3(torch.cat((up4, c3), dim=1))

        p3 = self.geometry_p3(t3, l4) if getattr(self, "use_geometry_ccff", False) else t3
        # Keep P4/P5 exactly on the native PAN path. The geometry expert only
        # specializes P3, where thin OCM boxes have a few feature pixels.
        down3 = self.downsample_p3(t3)
        self._require_spatial_match(down3, l4, "P3 downsample")
        p4 = self.pan_p4(torch.cat((down3, l4), dim=1))
        down4 = self.downsample_p4(p4)
        self._require_spatial_match(down4, l5, "P4 downsample")
        p5 = self.pan_p5(torch.cat((down4, l5), dim=1))

        outputs = (p3, p4, p5)
        _check_pyramid(
            outputs,
            self.output_channels,
            context="RTDETRV2_hyE_yolo output",
            reference=inputs,
        )
        return outputs


def build_rtdetrv2_hye_yolo(
    input_channels: Sequence[int],
    output_channels: Sequence[int],
    settings: Mapping[str, Any] | None,
    *,
    input_strides: Sequence[int] | None = None,
) -> RTDETRV2_hyE_yolo:
    """Construct the new neck only after its switch has been selected."""

    config = normalize_rtdetrv2_hye_yolo_config(
        settings,
        actual_in_channels=input_channels,
        actual_feat_strides=input_strides or (8, 16, 32),
    )
    if tuple(int(value) for value in output_channels) != tuple(config["in_channels"]):
        raise ValueError(
            "RTDETRV2_hyE_yolo output channels must match the YOLOv8s Detect/OBB "
            f"contract {config['in_channels']}, got {list(output_channels)}"
        )
    return RTDETRV2_hyE_yolo(**config)


__all__ = ["RTDETRV2_hyE_yolo", "build_rtdetrv2_hye_yolo"]
