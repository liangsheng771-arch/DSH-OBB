"""IG-LSF-AIFI v1 for the scale-preserving, 512-channel P5 path.

Independently usable and optionally selected by RTDETRV2_hyE_yolo. Pass raw,
already aligned SlotDCE guidance at forward time. The native 512-wide encoder
runs exactly once; CCFF is selected separately.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rtdetrv2_hye_config import normalize_rtdetrv2_hye_yolo_config
from .rtdetr.hybrid_encoder import TransformerEncoder, TransformerEncoderLayer
from .rtdetrv2_hye_yolo import RTDETRV2_hyE_yolo


_DEFAULTS = {
    "implementation_version": 1,
    "aux_channels": 128,
    "norm_groups": 8,
    "local_kernel_sizes": [3, 5],
    "spectral": {
        "band_edges": [0.25, 0.55],
        "band_temperature": 0.08,
        "eta": 0.5,
        "preserve_dc": True,
        "controller_hidden": 32,
        "controller_init_std": 0.001,
        "eps": 1e-6,
    },
    "router": {"feature_channels": 32, "guidance_channels": 16, "prior": [0.1, 0.1, 0.8]},
    "residual": {"scale_init": 0.05, "scale_max": 0.5},
}


def _merge(defaults: dict, supplied: Mapping | None, name: str) -> dict:
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, Mapping):
        raise TypeError(f"{name} must be a mapping")
    unknown = set(supplied).difference(defaults)
    if unknown:
        raise ValueError(f"Unsupported {name} settings: {sorted(unknown)}")
    return {
        key: _merge(value, supplied.get(key), f"{name}.{key}")
        if isinstance(value, dict) else deepcopy(supplied.get(key, value))
        for key, value in defaults.items()
    }


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def normalize_ig_lsf_aifi_config(settings: Mapping[str, Any] | None = None) -> dict:
    """Validate v1's fixed topology and its explicitly supported scalar settings."""
    config = _merge(_DEFAULTS, settings, "ig_lsf_aifi")
    for key in ("implementation_version", "aux_channels", "norm_groups"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"ig_lsf_aifi.{key} must be a positive integer")
    if config["aux_channels"] % config["norm_groups"]:
        raise ValueError("ig_lsf_aifi.norm_groups must divide aux_channels")
    for key in ("implementation_version", "aux_channels", "norm_groups"):
        if config[key] != _DEFAULTS[key]:
            raise ValueError(f"IGLSFAIFI v1 requires {key}={_DEFAULTS[key]}")
    kernels = config["local_kernel_sizes"]
    if not isinstance(kernels, (list, tuple)) or any(
        type(k) is not int or k <= 0 or k % 2 != 1 for k in kernels
    ):
        raise ValueError("local_kernel_sizes must contain positive odd integers")
    if list(kernels) != [3, 5]:
        raise ValueError("IGLSFAIFI v1 requires local_kernel_sizes=[3, 5]")
    config["local_kernel_sizes"] = list(kernels)

    spectral = config["spectral"]
    edges = spectral["band_edges"]
    if not isinstance(edges, (list, tuple)) or len(edges) != 2:
        raise ValueError("spectral.band_edges must contain two edges for three bands")
    edges = [_number(v, "spectral.band_edges") for v in edges]
    if not 0 < edges[0] < edges[1] < 1:
        raise ValueError("spectral.band_edges require 0 < r1 < r2 < 1")
    spectral["band_edges"] = edges
    for key in ("band_temperature", "controller_init_std", "eps"):
        spectral[key] = _number(spectral[key], f"spectral.{key}", positive=True)
    spectral["eta"] = _number(spectral["eta"], "spectral.eta")
    if not 0 < spectral["eta"] < 1:
        raise ValueError("spectral.eta must lie in (0, 1)")
    if spectral["preserve_dc"] is not True:
        raise ValueError("IGLSFAIFI v1 requires spectral.preserve_dc=true")
    if type(spectral["controller_hidden"]) is not int or spectral["controller_hidden"] != 32:
        raise ValueError("IGLSFAIFI v1 requires spectral.controller_hidden=32")

    router = config["router"]
    for key in ("feature_channels", "guidance_channels"):
        if type(router[key]) is not int or router[key] != _DEFAULTS["router"][key]:
            raise ValueError(f"IGLSFAIFI v1 requires router.{key}={_DEFAULTS['router'][key]}")
    prior = router["prior"]
    if not isinstance(prior, (list, tuple)) or len(prior) != 3:
        raise ValueError("router.prior must contain local/spectral/skip probabilities")
    prior = [_number(v, "router.prior", positive=True) for v in prior]
    if not math.isclose(sum(prior), 1.0, abs_tol=1e-7, rel_tol=0.0):
        raise ValueError("router.prior must sum to one")
    router["prior"] = prior
    residual = config["residual"]
    residual["scale_init"] = _number(residual["scale_init"], "residual.scale_init")
    residual["scale_max"] = _number(residual["scale_max"], "residual.scale_max", positive=True)
    if abs(residual["scale_init"]) >= residual["scale_max"]:
        raise ValueError("residual requires abs(scale_init) < scale_max")
    return config


class _LocalCorrection(nn.Module):
    def __init__(self, channels: int, groups: int, kernels: list[int]) -> None:
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, kernels[0], padding=kernels[0] // 2,
                             groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, kernels[1], padding=kernels[1] // 2,
                             groups=channels, bias=False)
        self.norm = nn.GroupNorm(groups, channels, eps=1e-5, affine=True)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, u: Tensor) -> Tensor:
        v = (self.dw3(u) + self.dw5(u)) / math.sqrt(2.0)
        return self.proj(F.silu(self.norm(v), inplace=False))


class _BandController(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(13, eps=1e-5)
        self.fc1 = nn.Linear(13, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, 3, bias=True)

    def forward(self, q: Tensor) -> Tensor:
        # Float views preserve gradients to half parameters without changing storage.
        with torch.autocast(device_type=q.device.type, enabled=False):
            q = F.layer_norm(q.float(), self.norm.normalized_shape,
                             self.norm.weight.float(), self.norm.bias.float(), self.norm.eps)
            q = F.silu(F.linear(q, self.fc1.weight.float(), self.fc1.bias.float()))
            return F.linear(q, self.fc2.weight.float(), self.fc2.bias.float()).tanh()


class _SharedSpectralCorrection(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.band_edges = tuple(config["band_edges"])
        self.temperature = config["band_temperature"]
        self.eta = config["eta"]
        self.eps = config["eps"]
        self.controller = _BandController(config["controller_hidden"])

    def frequency_templates(self, height: int, width: int, device: torch.device):
        """Small per-call FP32 radial templates; no cache or learnable frequency grid."""
        fy = torch.fft.fftfreq(height, device=device, dtype=torch.float32)
        fx = torch.fft.rfftfreq(width, device=device, dtype=torch.float32)
        radius = (fy[:, None].square() + fx[None, :].square()).sqrt() / math.sqrt(0.5)
        s1 = torch.sigmoid((radius - self.band_edges[0]) / self.temperature)
        s2 = torch.sigmoid((radius - self.band_edges[1]) / self.temperature)
        phi = torch.stack((1 - s1, s1 - s2, s2))
        dc_mask = torch.ones_like(radius)
        dc_mask[0, 0] = 0
        multiplicity = torch.full((width // 2 + 1,), 2.0, device=device, dtype=torch.float32)
        multiplicity[0] = 1
        if width % 2 == 0:
            multiplicity[-1] = 1
        return phi, dc_mask, multiplicity

    def delta_gain(self, band_coeff: Tensor, phi: Tensor, dc_mask: Tensor) -> Tensor:
        return self.eta * torch.einsum("bk,khw->bhw", band_coeff, phi).unsqueeze(1) * dc_mask[None, None]

    def forward(self, u: Tensor, guidance: Tensor, *, return_diagnostics: bool = False):
        with torch.autocast(device_type=u.device.type, enabled=False):
            height, width = u.shape[-2:]
            z = torch.fft.rfft2(u.float(), dim=(-2, -1), norm="ortho")
            phi, dc_mask, multiplicity = self.frequency_templates(height, width, u.device)
            magnitude_descriptor = torch.log1p(z.abs()).mean(dim=1)
            q_band = phi * multiplicity[None, None, :] * dc_mask[None]
            band_descriptor = torch.einsum("bhw,khw->bk", magnitude_descriptor, q_band)
            band_descriptor = band_descriptor / (q_band.sum(dim=(-2, -1))[None] + self.eps)
            normalized = band_descriptor / (band_descriptor.sum(dim=-1, keepdim=True) + self.eps)
            pooled_guidance = guidance.float().mean(dim=(-2, -1))
            coeff = self.controller(torch.cat((normalized, pooled_guidance), dim=-1))
            delta = self.delta_gain(coeff, phi, dc_mask)
            correction = torch.fft.irfft2(z * delta, s=(height, width), dim=(-2, -1), norm="ortho")
            if not return_diagnostics:
                return correction.to(u.dtype)
            with torch.no_grad():
                gain = 1 + delta.detach()
                diagnostics = {
                    "band_descriptor_raw": band_descriptor.detach(),
                    "band_descriptor_normalized": normalized.detach(),
                    "band_coeff": coeff.detach(),
                    "spectral_gain_min": gain.amin(dim=(1, 2, 3)),
                    "spectral_gain_mean": gain.mean(dim=(1, 2, 3)),
                    "spectral_gain_max": gain.amax(dim=(1, 2, 3)),
                }
            return correction.to(u.dtype), diagnostics


class _SpatialRouter(nn.Module):
    def __init__(self, channels: int, config: dict) -> None:
        super().__init__()
        self.feature = nn.Conv2d(channels, config["feature_channels"], 1, bias=True)
        self.guidance = nn.Conv2d(10, config["guidance_channels"], 1, bias=True)
        hidden = config["feature_channels"] + config["guidance_channels"]
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.logits = nn.Conv2d(hidden, 3, 1, bias=True)

    def forward(self, u: Tensor, guidance: Tensor) -> Tensor:
        ur = F.silu(self.feature(u), inplace=False)
        gr = F.silu(self.guidance(guidance.to(dtype=u.dtype)), inplace=False)
        features = F.silu(self.depthwise(torch.cat((ur, gr), dim=1)), inplace=False)
        return self.logits(features).float().softmax(dim=1)


class IGLSFAIFI(nn.Module):
    """Illumination-guided local/shared-spectral correction around native AIFI-512.

    ``forward(x, guidance, *, return_diagnostics=False)`` accepts Bx512xHxW
    features and Bx10xHxW raw SlotDCE guidance. No resizing, clamping or detaching
    is performed on the guidance. This standalone class is not a full neck.
    """

    implementation_version = 1

    def __init__(self, in_channels: int = 512, *, aifi: Mapping[str, Any] | None = None,
                 settings: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if type(in_channels) is not int or in_channels != 512:
            raise ValueError("IGLSFAIFI v1 requires the native 512-channel P5; no input projection to 256")
        self.in_channels = in_channels
        config = normalize_ig_lsf_aifi_config(settings)
        # AIFI defaults/validation belong to the existing neck, not this algorithm.
        aifi_config = normalize_rtdetrv2_hye_yolo_config({"aifi": {} if aifi is None else aifi})["aifi"]
        if aifi_config["num_encoder_layers"] != 1:
            raise ValueError("IGLSFAIFI v1 requires one native AIFI encoder layer")
        self.architecture_config = {
            "in_channels": in_channels, "aifi": aifi_config, "ig_lsf_aifi": config,
        }
        self.pe_temperature = aifi_config["pe_temperature"]
        encoder_args = {key: value for key, value in aifi_config.items()
                        if key not in {"pe_temperature", "num_encoder_layers"}}
        layer = TransformerEncoderLayer(d_model=in_channels, **encoder_args)
        self.native_aifi = TransformerEncoder(layer, aifi_config["num_encoder_layers"])

        channels, groups = config["aux_channels"], config["norm_groups"]
        self.aux_proj = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, bias=False),
            nn.GroupNorm(groups, channels, eps=1e-5, affine=True),
            nn.SiLU(inplace=False),
        )
        self.local = _LocalCorrection(channels, groups, config["local_kernel_sizes"])
        self.spectral = _SharedSpectralCorrection(config["spectral"])
        self.router = _SpatialRouter(channels, config["router"])
        self.out_proj = nn.Conv2d(channels, in_channels, 1, bias=False)
        residual = config["residual"]
        self.gamma_max = residual["scale_max"]
        self.gamma_logit = nn.Parameter(torch.tensor(math.atanh(residual["scale_init"] / self.gamma_max)))
        self._initialize_auxiliary(config)
        # Repository EMA expects tensor-only state_dicts. An integral metadata
        # buffer survives .half(), is not averaged by EMA and is never trainable.
        metadata = json.dumps(self.architecture_config, sort_keys=True, allow_nan=False).encode("utf-8")
        self.register_buffer("_architecture_metadata", torch.tensor(list(metadata), dtype=torch.uint8))

    def _initialize_auxiliary(self, config: dict) -> None:
        # Deliberately exclude native_aifi from traversal and preserve its RNG/init.
        for root in (self.aux_proj, self.local, self.spectral, self.router, self.out_proj):
            for module in root.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
        controller = self.spectral.controller
        nn.init.xavier_uniform_(controller.fc1.weight)
        nn.init.zeros_(controller.fc1.bias)
        nn.init.normal_(controller.fc2.weight, std=config["spectral"]["controller_init_std"])
        nn.init.zeros_(controller.fc2.bias)
        nn.init.zeros_(self.router.logits.weight)
        with torch.no_grad():
            self.router.logits.bias.copy_(torch.tensor(config["router"]["prior"]).log())

    @staticmethod
    def configuration_from_state_dict(state: Mapping, prefix: str = "") -> dict:
        """Read construction settings before a strict standalone checkpoint load."""
        value = state[prefix + "_architecture_metadata"]
        if not isinstance(value, Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise RuntimeError("Invalid IGLSFAIFI checkpoint architecture metadata")
        return json.loads(bytes(value.detach().cpu().tolist()).decode("utf-8"))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "_architecture_metadata"
        if key not in state_dict:
            raise RuntimeError("Missing IGLSFAIFI checkpoint _architecture_metadata; not a v1 module checkpoint")
        if self.configuration_from_state_dict(state_dict, prefix) != self.architecture_config:
            raise RuntimeError("IGLSFAIFI checkpoint architecture/config mismatch; rebuild with saved settings")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def no_weight_decay(self) -> set[str]:
        """Names relative to this module, consumed by SlotYOLOTrainer."""
        return {"gamma_logit"}

    def optimizer_parameter_groups(self, weight_decay: float = 5e-4) -> list[dict]:
        """Standalone optimizer groups, using repository rules plus scalar gamma exemption.

        Both standalone and detector training consume no_weight_decay().
        """
        from .trainer import collect_unique_optimizer_parameters

        decay = _number(weight_decay, "weight_decay")
        if decay < 0:
            raise ValueError("weight_decay must be non-negative")
        groups, _ = collect_unique_optimizer_parameters(self)
        return [
            {"params": list(groups[0].values()), "weight_decay": decay},
            {"params": list(groups[1].values()) + list(groups[2].values()), "weight_decay": 0.0},
        ]

    @staticmethod
    def _check_inputs(x: Tensor, guidance: Tensor) -> None:
        if not isinstance(x, Tensor) or x.ndim != 4 or x.shape[1] != 512:
            raise ValueError("IGLSFAIFI x must have shape [B,512,H,W]")
        if min(x.shape[0], x.shape[2], x.shape[3]) <= 0 or not x.is_floating_point():
            raise ValueError("IGLSFAIFI x requires positive B/H/W and floating dtype")
        if (not isinstance(guidance, Tensor) or guidance.ndim != 4
                or guidance.shape[:2] != (x.shape[0], 10)
                or guidance.shape[-2:] != x.shape[-2:]
                or guidance.device != x.device or not guidance.is_floating_point()):
            raise ValueError("IGLSFAIFI guidance must be raw floating [B,10,H,W] SlotDCE guidance "
                             "with matching batch, H/W and device (not the 16-channel ISAF embedding)")

    def native_forward(self, x: Tensor) -> Tensor:
        """Unmodified native encoder with the existing row-major 512-wide positions."""
        tokens = x.flatten(2).transpose(1, 2)
        position = RTDETRV2_hyE_yolo.build_2d_sincos_position_embedding(
            x.shape[-2], x.shape[-1], self.in_channels, self.pe_temperature,
            device=tokens.device, dtype=tokens.dtype,
        )
        return self.native_aifi(tokens, pos_embed=position).transpose(1, 2).reshape_as(x).contiguous()

    @staticmethod
    def fuse_corrections(local: Tensor, spectral: Tensor, weights: Tensor) -> Tensor:
        weights = weights.to(dtype=local.dtype)
        return weights[:, 0:1] * local + weights[:, 1:2] * spectral

    def forward(self, x: Tensor, guidance: Tensor, *, return_diagnostics: bool = False):
        self._check_inputs(x, guidance)
        t = self.native_forward(x)
        u = self.aux_proj(x)
        local = self.local(u)
        spectral_result = self.spectral(u, guidance, return_diagnostics=return_diagnostics)
        if return_diagnostics:
            spectral, diagnostics = spectral_result
        else:
            spectral = spectral_result
        weights = self.router(u, guidance)
        delta = self.fuse_corrections(local, spectral, weights)
        gamma = self.gamma_max * self.gamma_logit.float().tanh()
        y = t + gamma.to(t.dtype) * self.out_proj(delta).to(t.dtype)
        if not return_diagnostics:
            return y
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            def norm(value: Tensor) -> Tensor:
                return torch.linalg.vector_norm(value.detach().float().flatten(1), dim=1)

            routed = weights.detach().float()
            denominator = norm(u) + self.spectral.eps
            diagnostics.update({
                "router_weights": weights.detach(),
                "router_mean": weights.detach().mean(dim=(-2, -1)),
                "gamma": gamma.detach(),
                "local_delta_ratio": norm(routed[:, 0:1] * local.detach().float()) / denominator,
                "spectral_delta_ratio": norm(routed[:, 1:2] * spectral.detach().float()) / denominator,
                "final_delta_ratio": norm(y.detach().float() - t.detach().float()) / (norm(t) + self.spectral.eps),
            })
        return y, diagnostics


__all__ = ["IGLSFAIFI", "normalize_ig_lsf_aifi_config"]
