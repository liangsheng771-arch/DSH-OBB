"""Evaluate the current SlotDCE highlight, halo and prior behavior."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

import torch
from torch import Tensor
from torch.nn import functional as F

from config_utils import build_model, choose_device, load_config, load_model_checkpoint
from datasets import find_images
from image_utils import load_rgb_tensor, luminance, white_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config", default="config/slotDCE_v9_8_detector_lite.yaml"
    )
    parser.add_argument("--output", default="outputs/evaluate")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="Resize the longest side for faster diagnostics; 0 keeps original size.",
    )
    return parser.parse_args()


def entropy(value: Tensor, bins: int = 256) -> float:
    histogram = torch.histc(value.float(), bins=bins, min=0.0, max=1.0)
    probability = histogram / histogram.sum().clamp_min(1.0)
    probability = probability[probability > 0]
    return float(-(probability * probability.log2()).sum())


def gradient_correlation(source: Tensor, output: Tensor) -> float:
    source_gradients = torch.cat(
        (
            (source[..., 1:] - source[..., :-1]).flatten(),
            (source[..., 1:, :] - source[..., :-1, :]).flatten(),
        )
    )
    output_gradients = torch.cat(
        (
            (output[..., 1:] - output[..., :-1]).flatten(),
            (output[..., 1:, :] - output[..., :-1, :]).flatten(),
        )
    )
    return float(
        torch.nn.functional.cosine_similarity(
            source_gradients.unsqueeze(0),
            output_gradients.unsqueeze(0),
        )[0]
    )


def region_mean(value: Tensor, mask: Tensor) -> float:
    return float(value[mask].mean()) if bool(mask.any()) else 0.0


def direction_ratio(condition: Tensor, mask: Tensor) -> float:
    """Treat an absent region as vacuously passing the direction check."""

    return float(condition[mask].float().mean()) if bool(mask.any()) else 1.0


def relative_detail(value: Tensor, kernel_size: int = 9) -> Tensor:
    padding = kernel_size // 2
    mode = "reflect" if min(value.shape[-2:]) > padding else "replicate"
    base = F.avg_pool2d(F.pad(value, (padding,) * 4, mode=mode), kernel_size, 1)
    return (value - base) / base.clamp_min(0.04)


def detail_metrics(source: Tensor, output: Tensor, mask: Tensor) -> tuple[float, float]:
    source_detail = relative_detail(source)[mask]
    output_detail = relative_detail(output)[mask]
    if source_detail.numel() == 0:
        return 0.0, 0.0
    correlation = F.cosine_similarity(
        source_detail.unsqueeze(0), output_detail.unsqueeze(0), eps=1.0e-8
    )[0]
    amplitude = output_detail.abs().mean() / source_detail.abs().mean().clamp_min(1.0e-8)
    return float(correlation), float(amplitude)


def evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = choose_device(args.device)
    model = build_model(config["model"]).to(device)
    load_model_checkpoint(model, args.checkpoint, device)
    model.eval()
    paths = find_images(args.input)
    if not paths:
        raise RuntimeError("No evaluation images found.")

    rows: List[Dict[str, object]] = []
    with torch.inference_mode():
        for path in paths:
            image = load_rgb_tensor(path).unsqueeze(0).to(device)
            if args.max_side > 0 and max(image.shape[-2:]) > args.max_side:
                scale = args.max_side / max(image.shape[-2:])
                image = F.interpolate(
                    image,
                    size=(
                        max(1, round(image.shape[-2] * scale)),
                        max(1, round(image.shape[-1] * scale)),
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            outputs = model(image)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            enhanced = outputs["enhanced"]
            source = luminance(image)
            output = luminance(enhanced)
            delta = output - source
            dark = outputs["dark_mask"].bool()
            bright = outputs["bright_mask"].bool()
            core = outputs["white_mask"].bool()
            halo = outputs["halo_mask"].bool()
            normal = ~(dark | bright | core | halo)
            source_white = white_score(image)
            output_white = white_score(enhanced)
            input_chroma = image - source
            output_chroma = enhanced - output
            reliable = image.amax(dim=1, keepdim=True) < float(
                config["model"]["detail_saturation_ceiling"]
            )
            all_detail = detail_metrics(source, output, reliable)
            bright_detail = detail_metrics(source, output, reliable & bright)
            core_detail = detail_metrics(source, output, reliable & core)
            halo_detail = detail_metrics(source, output, reliable & halo)
            recovery_region = outputs["saturation_recovery_weight"] > 0.10
            row: Dict[str, object] = {
                "path": str(path),
                "input_luminance": float(source.mean()),
                "output_luminance": float(output.mean()),
                "dark_input": region_mean(source, dark),
                "dark_output": region_mean(output, dark),
                "dark_positive_ratio": direction_ratio(delta > 0, dark),
                "bright_input": region_mean(source, bright),
                "bright_output": region_mean(output, bright),
                "bright_negative_ratio": direction_ratio(delta < 0, bright),
                "core_ratio": float(core.float().mean()),
                "core_white_input": region_mean(source_white, core),
                "core_white_output": region_mean(output_white, core),
                "core_negative_ratio": direction_ratio(
                    (output_white - source_white) < 0, core
                ),
                "halo_ratio": float(halo.float().mean()),
                "halo_input": region_mean(source, halo),
                "halo_output": region_mean(output, halo),
                "halo_negative_ratio": direction_ratio(delta < 0, halo),
                "halo_context_excess": region_mean(
                    outputs["halo_context_excess"], halo
                ),
                "halo_relative_delta": region_mean(
                    outputs["halo_relative_delta"], halo
                ),
                "halo_veil_delta": region_mean(outputs["halo_veil_delta"], halo),
                "halo_requested_delta": region_mean(
                    outputs["halo_requested_delta"], halo
                ),
                "halo_floor_compensation": region_mean(
                    outputs["halo_floor_compensation"], halo
                ),
                "halo_total_delta": region_mean(outputs["halo_total_delta"], halo),
                "normal_rgb_mae": region_mean(
                    (enhanced - image).abs().mean(dim=1, keepdim=True), normal
                ),
                "input_entropy": entropy(source),
                "output_entropy": entropy(output),
                "gradient_correlation": gradient_correlation(source, output),
                "detail_correlation_all": all_detail[0],
                "detail_amplitude_all": all_detail[1],
                "detail_correlation_bright": bright_detail[0],
                "detail_amplitude_bright": bright_detail[1],
                "detail_correlation_core": core_detail[0],
                "detail_amplitude_core": core_detail[1],
                "detail_correlation_halo": halo_detail[0],
                "detail_amplitude_halo": halo_detail[1],
                "chroma_mae": float((output_chroma - input_chroma).abs().mean()),
                "detail_residual_abs": float(outputs["detail_residual"].abs().mean()),
                "detail_confidence": float(outputs["detail_confidence"].mean()),
                "detail_gain_fine": float(outputs["detail_gain_maps"][:, 0].mean()),
                "detail_gain_medium": float(outputs["detail_gain_maps"][:, 1].mean()),
                "detail_gain_coarse": float(outputs["detail_gain_maps"][:, 2].mean()),
                "clipped_white_ratio": float(
                    (outputs["clipped_white_weight"] > 0.50).float().mean()
                ),
                "recovery_region_ratio": float(recovery_region.float().mean()),
                "recovery_confidence": region_mean(
                    outputs["recovery_confidence"], recovery_region
                ),
                "recovery_context_rgb_mae": region_mean(
                    (outputs["recovery_rgb"] - outputs["recovery_context_rgb"])
                    .abs()
                    .mean(dim=1, keepdim=True),
                    recovery_region,
                ),
                "recovery_detail_residual_abs": region_mean(
                    outputs["recovery_detail_residual"].abs(), recovery_region
                ),
                "inference_ms": elapsed_ms,
            }
            for name, low, high in (
                ("y030_045", 0.30, 0.45),
                ("y045_060", 0.45, 0.60),
                ("y060_075", 0.60, 0.75),
                ("y075_100", 0.75, 1.01),
            ):
                band = (source >= low) & (source < high)
                band_input = region_mean(source, band)
                band_output = region_mean(output, band)
                row[f"{name}_input"] = band_input
                row[f"{name}_output"] = band_output
                row[f"{name}_scale"] = band_output / max(band_input, 1.0e-8)
            for slot_index in range(model.num_slots):
                row[f"slot_{slot_index + 1}_usage"] = float(
                    outputs["attn_map_full"][:, slot_index].mean()
                )
                row[f"slot_{slot_index + 1}_gain"] = float(
                    outputs["slot_gain"][0, slot_index]
                )
            rows.append(row)

    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_image.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric = [key for key, value in rows[0].items() if isinstance(value, float)]
    summary = {
        "image_count": len(rows),
        **{key: mean(float(row[key]) for row in rows) for key in numeric},
        "direction_pass": all(
            float(row["dark_positive_ratio"]) > 0.999
            and float(row["bright_negative_ratio"]) > 0.999
            and float(row["core_negative_ratio"]) > 0.999
            and float(row["halo_negative_ratio"]) > 0.999
            for row in rows
        ),
        "note": "No-reference diagnostics; detector mAP and human review remain required.",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    evaluate(parse_args())
