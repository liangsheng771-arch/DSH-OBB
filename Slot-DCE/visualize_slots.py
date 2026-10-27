"""Render current SlotDCE Slot and recovery diagnostics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from config_utils import build_model, choose_device, load_config, load_model_checkpoint
from datasets import find_images
from image_utils import load_rgb_tensor, luminance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config", default="config/slotDCE_v9_8_detector_lite.yaml"
    )
    parser.add_argument("--output", default="outputs/slots")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def rgb(tensor: Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def scalar(tensor: Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def show(axis: plt.Axes, value: np.ndarray, title: str, **kwargs: object) -> None:
    axis.imshow(value, **kwargs)
    axis.set_title(title)
    axis.axis("off")


def save_figure(source: Path, destination: Path, image: Tensor, outputs: dict) -> None:
    panels = 39 + outputs["attn_map_full"].shape[1]
    columns = 4
    rows = math.ceil(panels / columns)
    figure, grid = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    axes = list(np.asarray(grid).flat)
    cursor = 0
    show(axes[cursor], rgb(image[0]), "Input"); cursor += 1
    show(axes[cursor], rgb(outputs["enhanced"][0]), "Enhanced"); cursor += 1
    show(axes[cursor], scalar(outputs["input_luminance"][0, 0]), "Input Y", cmap="gray", vmin=0, vmax=1); cursor += 1
    show(axes[cursor], scalar(outputs["output_luminance"][0, 0]), "Output Y", cmap="gray", vmin=0, vmax=1); cursor += 1
    show(axes[cursor], scalar(outputs["tone_delta"][0, 0]), "Tone delta", cmap="coolwarm", vmin=-0.4, vmax=0.4); cursor += 1
    show(axes[cursor], scalar(outputs["correction_strength_map"][0, 0]), "Signed correction strength", cmap="coolwarm", vmin=-1, vmax=1); cursor += 1
    show(axes[cursor], scalar(outputs["dark_delta"][0, 0]), "Dark lift", cmap="magma", vmin=0, vmax=0.4); cursor += 1
    show(axes[cursor], scalar(outputs["bright_delta"][0, 0]), "Ordinary-bright compression", cmap="magma", vmin=0, vmax=0.3); cursor += 1
    show(axes[cursor], scalar(outputs["core_delta"][0, 0]), "White-core compression", cmap="magma", vmin=0, vmax=0.5); cursor += 1
    show(axes[cursor], scalar(outputs["halo_delta"][0, 0]), "Halo compression", cmap="magma", vmin=0, vmax=0.4); cursor += 1
    show(axes[cursor], scalar(outputs["halo_relative_delta"][0, 0]), "Halo multiplicative delta", cmap="magma", vmin=0, vmax=0.2); cursor += 1
    show(axes[cursor], scalar(outputs["halo_veil_delta"][0, 0]), "Additive veil removal", cmap="magma", vmin=0, vmax=0.2); cursor += 1
    show(axes[cursor], scalar(outputs["halo_requested_delta"][0, 0]), "Requested halo removal", cmap="magma", vmin=0, vmax=0.25); cursor += 1
    show(axes[cursor], scalar(outputs["halo_floor_compensation"][0, 0]), "Anti-ring compensation", cmap="magma", vmin=0, vmax=0.1); cursor += 1
    show(axes[cursor], scalar(outputs["halo_context_excess"][0, 0]), "Neighbourhood light excess", cmap="magma", vmin=0, vmax=0.3); cursor += 1
    show(axes[cursor], scalar(outputs["base_tone_delta"][0, 0]), "Base tone delta", cmap="coolwarm", vmin=-0.4, vmax=0.4); cursor += 1
    show(axes[cursor], scalar(outputs["detail_residual"][0, 0]), "Detail residual", cmap="coolwarm", vmin=-0.1, vmax=0.1); cursor += 1
    show(axes[cursor], scalar(outputs["detail_confidence"][0, 0]), "Learned detail confidence", cmap="viridis", vmin=0, vmax=1); cursor += 1
    show(axes[cursor], scalar(outputs["detail_gain_maps"][0, 0]), "Fine Retinex gain", cmap="viridis", vmin=0.8, vmax=1.7); cursor += 1
    show(axes[cursor], scalar(outputs["detail_gain_maps"][0, 1]), "Medium Retinex gain", cmap="viridis", vmin=0.8, vmax=1.6); cursor += 1
    show(axes[cursor], scalar(outputs["detail_gain_maps"][0, 2]), "Coarse Retinex gain", cmap="viridis", vmin=0.8, vmax=1.5); cursor += 1
    show(axes[cursor], scalar(outputs["dark_weight"][0, 0]), "Dark mask", cmap="magma", vmin=0, vmax=1); cursor += 1
    show(axes[cursor], scalar(outputs["bright_weight"][0, 0]), "Bright mask", cmap="magma", vmin=0, vmax=1); cursor += 1
    show(axes[cursor], scalar(outputs["white_core_weight"][0, 0]), "White-core mask", cmap="magma", vmin=0, vmax=1); cursor += 1
    show(axes[cursor], scalar(outputs["halo_weight"][0, 0]), "Halo-ring mask", cmap="magma", vmin=0, vmax=1); cursor += 1
    show(axes[cursor], scalar(outputs["core_scale"][0, 0]), "White-core scale", cmap="viridis", vmin=.3, vmax=1); cursor += 1
    show(axes[cursor], scalar(outputs["halo_scale"][0, 0]), "Halo scale", cmap="viridis", vmin=.6, vmax=1); cursor += 1
    show(axes[cursor], scalar(outputs["dark_modulation"][0, 0]), "Dark local modulation", cmap="viridis", vmin=.75, vmax=1.25); cursor += 1
    show(axes[cursor], scalar(outputs["bright_modulation"][0, 0]), "Bright local modulation", cmap="viridis", vmin=.70, vmax=1.30); cursor += 1
    show(axes[cursor], scalar(outputs["white_modulation"][0, 0]), "White local modulation", cmap="viridis", vmin=.75, vmax=1.25); cursor += 1
    show(axes[cursor], scalar(outputs["halo_modulation"][0, 0]), "Learned halo modulation", cmap="viridis", vmin=.70, vmax=1.30); cursor += 1
    show(axes[cursor], scalar(outputs["clipped_white_weight"][0, 0]), "Clipped-white probability", cmap="magma", vmin=0, vmax=1); cursor += 1
    show(axes[cursor], scalar(outputs["saturation_recovery_weight"][0, 0]), "Context-recovery gate", cmap="magma", vmin=0, vmax=1); cursor += 1
    show(axes[cursor], scalar(outputs["recovery_confidence"][0, 0]), "Learned recovery confidence", cmap="viridis", vmin=0, vmax=1); cursor += 1
    show(axes[cursor], rgb(outputs["recovery_context_rgb"][0]), "Known-neighbour colour prior"); cursor += 1
    show(axes[cursor], rgb(outputs["recovery_rgb"][0]), "Learned context reconstruction"); cursor += 1
    show(axes[cursor], scalar(outputs["recovery_detail_residual"][0, 0]), "Recovered detail residual", cmap="coolwarm", vmin=-0.08, vmax=0.08); cursor += 1
    for slot in range(outputs["attn_map_full"].shape[1]):
        show(
            axes[cursor],
            scalar(outputs["attn_map_full"][0, slot]),
            f"S{slot + 1} attention",
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        cursor += 1
    histogram = axes[cursor]
    histogram.hist(scalar(outputs["input_luminance"][0, 0]).ravel(), bins=100, range=(0, 1), alpha=.55, label="input")
    histogram.hist(scalar(outputs["output_luminance"][0, 0]).ravel(), bins=100, range=(0, 1), alpha=.55, label="output")
    histogram.legend(); histogram.set_title("Luminance histogram"); cursor += 1
    text = axes[cursor]; text.axis("off")
    lines = [source.name, "", "signed correction fraction:"]
    for index, (gain, usage) in enumerate(zip(outputs["slot_gain"][0], outputs["slot_usage"][0])):
        lines.append(f"S{index + 1}: gain={float(gain):+.4f}, usage={float(usage):.4f}")
    text.text(.02, .98, "\n".join(lines), va="top", family="monospace"); cursor += 1
    for axis in axes[cursor:]:
        axis.axis("off")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def visualize(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = choose_device(args.device)
    model = build_model(config["model"]).to(device)
    load_model_checkpoint(model, args.checkpoint, device)
    model.eval()
    root = Path(args.input).expanduser()
    paths = find_images(root)
    if args.max_images > 0:
        paths = paths[: args.max_images]
    output = Path(args.output).expanduser()
    with torch.inference_mode():
        for source in paths:
            image = load_rgb_tensor(source).unsqueeze(0).to(device)
            outputs = model(image)
            relative = Path(source.name) if root.is_file() else source.relative_to(root.resolve())
            save_figure(source, output / relative.parent / f"{relative.stem}_slots.png", image, outputs)


if __name__ == "__main__":
    visualize(parse_args())
