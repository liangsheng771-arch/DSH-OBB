"""Parameter and FLOPs reporting for the Slot-YOLO feature pipeline."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model

from .myconfigv2.feature_pipeline import ConfiguredFeatureStage


def _parameter_count(module: nn.Module) -> int:
    """Count unique learnable parameters in a module."""

    return sum(parameter.numel() for parameter in module.parameters())


def _image_size(imgsz: int | list[int] | tuple[int, ...]) -> tuple[int, int]:
    """Normalize a training image size to ``(height, width)``."""

    if isinstance(imgsz, (list, tuple)):
        if len(imgsz) == 1:
            return int(imgsz[0]), int(imgsz[0])
        return int(imgsz[0]), int(imgsz[1])
    return int(imgsz), int(imgsz)


def _profile_flops(module: nn.Module, inputs: tuple[Any, ...]) -> float | None:
    """Profile a module and return multiply-adds as GFLOPs.

    ``thop`` reports MACs, so the result is multiplied by two to follow the
    FLOPs convention used by Ultralytics.  Profiling is optional and never
    interrupts training when the dependency or a custom operator is missing.
    """

    # THOP's default module hooks miss the original AIFI's functional attention
    # matmuls. Use operator-level counting for the new encoder and its full model.
    if getattr(module, "neck_name", None) in {
        "efficient_hybrid_encoder",
        "rtdetrv2_hye_yolo",
    }:
        training_states = {id(child): child.training for child in module.modules()}
        try:
            from torch.utils.flop_counter import FlopCounterMode

            module.eval()
            with torch.no_grad(), FlopCounterMode(display=False) as counter:
                module(*inputs)
            return float(counter.get_total_flops() / 1e9)
        except Exception as error:
            LOGGER.debug("Encoder FLOPs profiling skipped: %s", error)
            return None
        finally:
            for child in module.modules():
                child.training = training_states[id(child)]

    try:
        import thop
    except ImportError:
        return None

    training_states = {id(child): child.training for child in module.modules()}
    try:
        module.eval()
        with torch.no_grad():
            macs, _ = thop.profile(module, inputs=inputs, verbose=False)
        return float(macs * 2.0 / 1e9)
    except Exception as error:  # pragma: no cover - depends on thop/operator versions
        LOGGER.debug("FLOPs profiling skipped for %s: %s", type(module).__name__, error)
        return None
    finally:
        for child in module.modules():
            child.training = training_states[id(child)]


class _DetectionEnhancer(nn.Module):
    """Expose Slot-DCE's detector path to a FLOPs profiler."""

    def __init__(
        self,
        enhancer: nn.Module,
        guidance_sizes: tuple[tuple[int, int], ...] | None,
    ) -> None:
        super().__init__()
        self.enhancer = enhancer
        self.guidance_sizes = guidance_sizes

    def forward(self, image: Tensor):
        output = self.enhancer.enhance_for_detection(
            image,
            return_guidance=self.guidance_sizes is not None,
            guidance_sizes=self.guidance_sizes,
        )
        return output[0] if isinstance(output, tuple) else output


def _format_complexity(parameters: int, flops: float | None) -> str:
    flops_text = "unavailable" if flops is None else f"{flops:.3f} GFLOPs"
    return f"{parameters:,} ({parameters / 1e6:.3f} M), {flops_text}"


def log_model_complexity(
    model: nn.Module,
    imgsz: int | list[int] | tuple[int, ...] = 640,
) -> dict[str, dict[str, int | float | None]]:
    """Log total and module-level parameter/FLOPs complexity for Slot-YOLO."""

    model = unwrap_model(model)
    height, width = _image_size(imgsz)
    parameter = next(model.parameters())
    device = parameter.device
    dtype = (
        parameter.dtype
        if parameter.is_floating_point() and device.type == "cuda"
        else torch.float32
    )

    total_inputs = (torch.zeros((1, 3, height, width), device=device, dtype=dtype),)
    total_flops = _profile_flops(model, total_inputs)

    enhancer = model.enhancer
    slotdce_isaf_enabled = bool(getattr(model, "use_slotdce_isaf", True))
    guidance_indices = tuple(model._runtime_guidance_stage_indices())
    guidance_sizes = tuple(
        (
            max((height + stride - 1) // stride, 1),
            max((width + stride - 1) // stride, 1),
        )
        for stride in (8, 16, 32)
    )
    detector_guidance_sizes = tuple(guidance_sizes[index] for index in guidance_indices)
    enhancer_flops = (
        _profile_flops(
            _DetectionEnhancer(
                enhancer,
                detector_guidance_sizes if guidance_indices else None,
            ),
            total_inputs,
        )
        if slotdce_isaf_enabled
        else 0.0
    )

    sgdc_enabled = bool(getattr(model, "sgdc_enabled", False))
    channels = (
        int(model.sgdc.c3_channels),
        int(model.sgdc.c4_channels),
        int(model.sgdc.c5_channels),
    ) if sgdc_enabled else (0, 0, 0)
    stage_shapes = tuple(
        (
            max((height + stride - 1) // stride, 1),
            max((width + stride - 1) // stride, 1),
        )
        for stride in (8, 16, 32)
    )

    isaf_parameters = 0
    isaf_flops = 0.0
    isaf_profiled = False
    for stage_index, stage in enumerate(
        (model.fuse_p3, model.fuse_p4, model.fuse_p5)
    ):
        if not isinstance(stage, ConfiguredFeatureStage):
            continue
        channels_stage = int(stage.fusion.structure_extractor.filter.channels)
        stage_height, stage_width = stage_shapes[stage_index]
        guidance_channels = int(stage.guidance_adapter.encoder[0].in_channels)
        stage_inputs = (
            torch.zeros((1, channels_stage, stage_height, stage_width), device=device, dtype=dtype),
            torch.zeros((1, channels_stage, stage_height, stage_width), device=device, dtype=dtype),
            torch.zeros((1, guidance_channels, stage_height, stage_width), device=device, dtype=dtype),
        )
        isaf_parameters += _parameter_count(stage)
        stage_flops = _profile_flops(stage, stage_inputs)
        if stage_flops is None:
            isaf_flops = None
        elif isaf_flops is not None:
            isaf_flops += stage_flops
        isaf_profiled = True

    sgdc_parameters = 0
    sgdc_flops: float | None = 0.0
    if sgdc_enabled:
        sgdc = model.sgdc
        sgdc_parameters = _parameter_count(sgdc)
        c3, c4, c5 = channels
        (h3, w3), (h4, w4), (h5, w5) = stage_shapes
        sgdc_inputs = (
            torch.zeros((1, c3, h3, w3), device=device, dtype=dtype),
            torch.zeros((1, c4, h4, w4), device=device, dtype=dtype),
            torch.zeros((1, c5, h5, w5), device=device, dtype=dtype),
        )
        sgdc_flops = _profile_flops(sgdc, sgdc_inputs)

    report = {
        "total": {
            "parameters": _parameter_count(model),
            "flops": total_flops,
        },
        "slot_dce": {
            "parameters": _parameter_count(enhancer),
            "flops": enhancer_flops,
        },
        "isaf": {
            "parameters": isaf_parameters,
            "flops": isaf_flops if isaf_profiled else 0.0,
        },
        "sgdc": {
            "parameters": sgdc_parameters,
            "flops": sgdc_flops if sgdc_enabled else 0.0,
        },
    }

    if getattr(model, "use_efficient_hybrid_encoder", False):
        hybrid = model.efficient_hybrid_neck
        hybrid_inputs = tuple(
            torch.zeros((1, channels, h, w), device=device, dtype=dtype)
            for channels, (h, w) in zip(hybrid.input_channels, stage_shapes)
        )
        if getattr(hybrid, "required_guidance_stages", ()):
            hybrid_inputs += (tuple(torch.zeros((1, 10, h, w), device=device, dtype=dtype)
                                    for h, w in stage_shapes),)
        report["efficient_hybrid_encoder"] = {
            "parameters": _parameter_count(hybrid),
            "flops": _profile_flops(hybrid, hybrid_inputs),
        }
    if getattr(model, "use_rtdetrv2_hye_yolo", False):
        neck = model.rtdetrv2_hye_yolo_neck
        neck_inputs = tuple(
            torch.zeros((1, channels, h, w), device=device, dtype=dtype)
            for channels, (h, w) in zip(neck.input_channels, stage_shapes)
        )
        report["rtdetrv2_hye_yolo"] = {
            "parameters": _parameter_count(neck),
            "flops": _profile_flops(neck, neck_inputs),
        }

    LOGGER.info("Slot-YOLO complexity @ input=%sx%s:", height, width)
    LOGGER.info(
        "  Slot-DCE+ISAF status: %s",
        "enabled" if slotdce_isaf_enabled else "disabled",
    )
    LOGGER.info("  Neck status: %s", getattr(model, "neck_description", "Neck:YOLO-PAN"))
    if getattr(model, "use_ig_fs_ccff", False):
        LOGGER.info("  FLOPs scope: convolution/matmul counts; FFT/IFFT, Haar and elementwise costs are excluded.")
    LOGGER.info("  SGDC status: %s", "enabled" if sgdc_enabled else "disabled")
    LOGGER.info("  Total   : %s", _format_complexity(report["total"]["parameters"], report["total"]["flops"]))
    LOGGER.info("  Slot-DCE: %s", _format_complexity(report["slot_dce"]["parameters"], report["slot_dce"]["flops"]))
    LOGGER.info("  ISAF    : %s", _format_complexity(report["isaf"]["parameters"], report["isaf"]["flops"]))
    LOGGER.info("  SGDC    : %s", _format_complexity(report["sgdc"]["parameters"], report["sgdc"]["flops"]))
    if "efficient_hybrid_encoder" in report:
        row = report["efficient_hybrid_encoder"]
        encoder = model.efficient_hybrid_neck.encoder
        report["ig_aifi"] = {"parameters": _parameter_count(encoder.encoder) if getattr(model, "use_ig_aifi", False) else 0}
        report["ig_fs_ccff"] = {"parameters": _parameter_count(encoder.ccff) if getattr(model, "use_ig_fs_ccff", False) else 0}
        report["total"]["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        report["p5_semantic_refine"] = {"parameters": _parameter_count(model.efficient_hybrid_neck.p5_semantic_refine)
                                        if getattr(model, "use_p5_dual_branch", False) else 0}
        LOGGER.info("  EHE neck: %s", _format_complexity(row["parameters"], row["flops"]))
        LOGGER.info("  IG-AIFI parameters: %s; IG-FS-CCFF parameters: %s; trainable total: %s",
                    report["ig_aifi"]["parameters"], report["ig_fs_ccff"]["parameters"], report["total"]["trainable_parameters"])
        LOGGER.info("  P5 semantic refinement parameters: %s", report["p5_semantic_refine"]["parameters"])
        LOGGER.info("  EHE/total FLOPs include attention matmuls; use compare_necks.py for the same counting method in both modes.")
    if "rtdetrv2_hye_yolo" in report:
        row = report["rtdetrv2_hye_yolo"]
        LOGGER.info("  RTDETRV2_hyE_yolo neck: %s", _format_complexity(row["parameters"], row["flops"]))
    return report


__all__ = ["log_model_complexity"]
