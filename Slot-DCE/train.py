"""Train the current Detector-Lite SlotDCE model."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config_utils import (
    ARCHITECTURE,
    build_loss,
    build_model,
    choose_device,
    load_config,
    load_model_checkpoint,
    torch_load,
)
from datasets import UndergroundDataset
from image_utils import luminance, save_tensor_grid
from local_filters import local_mean

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]


SAMPLING_MODES = ("full_resize", "random", "white", "mixed")
LOSS_GROUPS = ("tone", "texture", "recovery", "smoothness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/slotDCE_v9_8_detector_lite.yaml"
    )
    parser.add_argument("--data", help="Override training.train_dir.")
    parser.add_argument("--output", help="Override training.output_dir.")
    parser.add_argument("--resume", help="Resume a checkpoint of the configured version.")
    parser.add_argument(
        "--weights",
        help="Initialize current-model weights; reset optimizer and epoch.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--ablate-loss",
        action="append",
        choices=LOSS_GROUPS,
        help="Disable one loss group; repeat the option to disable several.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_grad_scaler(enabled: bool, init_scale: float, growth_interval: int) -> Any:
    kwargs = {
        "enabled": enabled,
        "init_scale": init_scale,
        "growth_interval": growth_interval,
    }
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", **kwargs)
        except TypeError:
            return torch.amp.GradScaler(**kwargs)
    return torch.cuda.amp.GradScaler(**kwargs)


def save_checkpoint(state: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(state), temporary)
    temporary.replace(destination)


def append_csv(path: Path, row: Mapping[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def format_slots(values: Tensor) -> str:
    numbers = values.detach().float().cpu().tolist()
    return "[" + ", ".join(
        f"S{index + 1}={value:+.3f}"
        for index, value in enumerate(numbers)
    ) + "]"


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    weight = mask.to(value.dtype)
    numerator = (value * weight).sum()
    return numerator / weight.sum().clamp_min(1.0)


def masked_slot_mean(attention: Tensor, mask: Tensor) -> Tensor:
    weight = mask.to(attention.dtype)
    return (attention * weight).sum(dim=(0, 2, 3)) / weight.sum().clamp_min(1.0)


def relative_detail(value: Tensor, kernel_size: int = 9) -> Tensor:
    base = local_mean(value, kernel_size)
    return (value - base) / base.clamp_min(0.04)


def detail_retention(source: Tensor, output: Tensor, mask: Tensor) -> tuple[float, float]:
    weight = mask.to(source.dtype)
    source_detail = relative_detail(source) * weight
    output_detail = relative_detail(output) * weight
    source_flat = source_detail.flatten()
    output_flat = output_detail.flatten()
    correlation = F.cosine_similarity(
        source_flat.unsqueeze(0), output_flat.unsqueeze(0), eps=1.0e-8
    )[0]
    amplitude = output_flat.abs().sum() / source_flat.abs().sum().clamp_min(1.0e-8)
    return float(correlation), float(amplitude)


def format_crops(modes: Sequence[str]) -> str:
    counts = Counter(modes)
    return "{" + ",".join(
        f"{mode}={counts[mode]}" for mode in SAMPLING_MODES if counts[mode]
    ) + "}"


def simulate_saturation(
    images: Tensor,
    settings: Mapping[str, Any],
) -> tuple[Tensor, Tensor, Tensor]:
    """Create local white clipping using the unpaired input as its own target."""

    fraction = float(settings.get("batch_fraction", 0.25))
    count = max(1, min(images.shape[0], round(images.shape[0] * fraction)))
    indices = torch.randperm(images.shape[0], device=images.device)[:count]
    target = images[indices]
    low, high = (float(value) for value in settings.get("gain_range", (1.60, 2.40)))
    peak_gains = torch.empty(count, 1, 1, 1, device=images.device).uniform_(low, high)
    height, width = target.shape[-2:]
    yy = torch.arange(height, device=images.device, dtype=target.dtype).view(1, 1, height, 1)
    xx = torch.arange(width, device=images.device, dtype=target.dtype).view(1, 1, 1, width)
    target_max = target.amax(dim=1, keepdim=True)
    target_min = target.amin(dim=1, keepdim=True)
    target_value = luminance(target)
    saturation = (target_max - target_min) / target_max.clamp_min(1.0e-6)
    clip_threshold = float(settings.get("clip_threshold", 0.985))
    source_ceiling = float(settings.get("source_ceiling", 0.97))
    minimum_source_luminance = float(
        settings.get("minimum_source_luminance", 0.45)
    )
    maximum_source_saturation = float(
        settings.get("maximum_source_saturation", 0.45)
    )
    recoverable = (
        (target_max <= source_ceiling)
        & (target_value >= minimum_source_luminance)
        & (saturation <= maximum_source_saturation)
    )

    # Select the optical-veil centre from a pixel for which the unclipped target
    # still contains real structure. Random centres in dark coal frequently made
    # no clipped pixels at all and silently starved the recovery branch.
    random_scores = torch.rand_like(target_value).masked_fill(~recoverable, -1.0)
    flat_centres = random_scores.flatten(1).argmax(dim=1)
    has_recoverable = recoverable.flatten(1).any(dim=1).view(count, 1, 1, 1)
    center_y = torch.div(flat_centres, width, rounding_mode="floor").to(
        target.dtype
    ).view(count, 1, 1, 1)
    center_x = (flat_centres % width).to(target.dtype).view(count, 1, 1, 1)
    radius_low, radius_high = (
        float(value) for value in settings.get("radius_range", (0.12, 0.30))
    )
    radius = torch.empty(count, 1, 1, 1, device=images.device).uniform_(
        radius_low,
        radius_high,
    ) * min(height, width)
    exposure_blob = torch.exp(
        -((yy - center_y).square() + (xx - center_x).square())
        / (2.0 * radius.square().clamp_min(1.0))
    )
    gain_map = 1.0 + (peak_gains - 1.0) * exposure_blob
    veil_low, veil_high = (
        float(value) for value in settings.get("veil_range", (0.10, 0.30))
    )
    random_veil = torch.empty(
        count, 1, 1, 1, device=images.device, dtype=target.dtype
    ).uniform_(veil_low, veil_high)
    centre_minimum = target_min.flatten(1).gather(
        1, flat_centres[:, None]
    ).view(count, 1, 1, 1)
    required_veil = (
        clip_threshold + 0.01 - centre_minimum * peak_gains
    ).clamp_min(0.0)
    peak_veil = torch.maximum(random_veil, required_veil).clamp_max(0.75)
    synthetic = (target * gain_map + peak_veil * exposure_blob).clamp(0.0, 1.0)

    mask = (
        (synthetic.amin(dim=1, keepdim=True) >= clip_threshold)
        & recoverable
        & has_recoverable
    )
    return target, synthetic, mask.to(images.dtype)


def save_preview(
    images: Tensor,
    outputs: Mapping[str, Tensor],
    destination: Path,
    epoch: int,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    count = min(4, images.shape[0])
    delta = outputs["tone_delta"][:count]
    delta_view = (0.5 + delta / 0.8).clamp(0.0, 1.0).repeat(1, 3, 1, 1)
    detail = outputs["detail_residual"][:count]
    detail_view = (0.5 + detail / 0.20).clamp(0.0, 1.0).repeat(1, 3, 1, 1)
    rows = torch.cat(
        (
            images[:count].detach().cpu(),
            outputs["enhanced"][:count].detach().cpu(),
            delta_view.detach().cpu(),
            detail_view.detach().cpu(),
        ),
        dim=0,
    )
    save_tensor_grid(
        rows,
        destination / f"epoch_{epoch:04d}_input_output_delta.png",
        nrow=count,
    )
    slots = outputs["attn_map_full"][0].detach().cpu().unsqueeze(1)
    save_tensor_grid(
        slots,
        destination / f"epoch_{epoch:04d}_slots.png",
        nrow=slots.shape[0],
    )


def _configure_training(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config)
    training = config["training"]
    overrides = {
        "train_dir": args.data,
        "output_dir": args.output,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }
    for key, value in overrides.items():
        if value is not None:
            training[key] = value
    if args.no_amp:
        training["mixed_precision"] = False
    configured_loss_weights = config["loss"].setdefault("weights", {})
    for name in getattr(args, "ablate_loss", None) or ():
        if name not in configured_loss_weights:
            raise ValueError(
                f"Loss group {name!r} is not used by this config; choose from "
                f"{sorted(configured_loss_weights)}."
            )
        configured_loss_weights[name] = 0.0
    if int(training["epochs"]) <= 0 or int(training["batch_size"]) <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    return config


def train(args: argparse.Namespace) -> None:
    config = _configure_training(args)
    training = config["training"]
    seed = int(training.get("seed", 2026))
    seed_everything(seed)
    device = choose_device(args.device)

    output_root = Path(training["output_dir"]).expanduser()
    checkpoint_dir = output_root / "checkpoints"
    preview_dir = output_root / "previews"
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)

    sampling = training["exposure_sampling"]
    probabilities = {
        "full_resize": float(sampling["full_resize_probability"]),
        "random": float(sampling["random_crop_probability"]),
        "white": float(sampling["white_crop_probability"]),
        "mixed": float(sampling["mixed_crop_probability"]),
    }
    dataset = UndergroundDataset(
        training["train_dir"],
        image_size=int(sampling["crop_size"]),
        scale_range=training.get("scale_range", (0.9, 1.1)),
        rotation_degrees=float(training.get("rotation_degrees", 0.0)),
        horizontal_flip_probability=float(
            training.get("horizontal_flip_probability", 0.5)
        ),
        sampling_probabilities=probabilities,
        crop_size=int(sampling["crop_size"]),
        exposure_thumbnail_size=int(sampling.get("thumbnail_size", 96)),
        crop_dark_threshold=float(sampling.get("dark_threshold", 0.30)),
        crop_white_threshold=float(sampling.get("white_threshold", 0.70)),
        crop_min_pixels=int(sampling.get("min_region_pixels", 1)),
        cache_images=bool(training.get("cache_images", False)),
        cache_short_side=int(training.get("cache_short_side", 384)),
        cache_workers=int(training.get("cache_workers", 8)),
    )
    workers = int(training.get("num_workers", 4))
    loader_kwargs: Dict[str, Any] = {
        "batch_size": int(training["batch_size"]),
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": bool(training.get("pin_memory", True)) and device.type == "cuda",
        "drop_last": bool(training.get("drop_last", True)),
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(seed),
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(training.get("prefetch_factor", 2))
    loader = DataLoader(dataset, **loader_kwargs)
    if len(loader) == 0:
        raise RuntimeError("No training batches: reduce batch_size or disable drop_last.")

    model = build_model(config["model"]).to(device)
    architecture = ARCHITECTURE
    resume = args.resume or training.get("resume")
    if args.weights and resume:
        raise ValueError("Use either --weights or --resume, not both.")
    if args.weights:
        initial = load_model_checkpoint(model, args.weights, device)
        print(
            f"Initialized {architecture} model parameters from "
            f"{initial.get('architecture', 'checkpoint')} "
            f"epoch {initial.get('epoch', 'n/a')}; optimizer and schedule are reset."
        )
    criterion = build_loss(config["loss"], config["model"]).to(device)
    channels_last = bool(training.get("channels_last", True)) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(training.get("cudnn_benchmark", True))
        torch.backends.cuda.matmul.allow_tf32 = bool(training.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(training.get("allow_tf32", True))

    optimizer_kwargs = {
        "lr": float(training["learning_rate"]),
        "weight_decay": float(training.get("weight_decay", 1e-6)),
    }
    if bool(training.get("fused_optimizer", True)) and device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = AdamW(model.parameters(), **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        optimizer = AdamW(model.parameters(), **optimizer_kwargs)
    epochs = int(training["epochs"])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(training.get("min_lr", 5e-6)),
    )

    amp_enabled = device.type == "cuda" and bool(training.get("mixed_precision", True))
    amp_name = str(training.get("amp_dtype", "bfloat16")).lower()
    amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
    scaler = create_grad_scaler(
        amp_enabled and amp_dtype == torch.float16,
        float(training.get("amp_init_scale", 1024.0)),
        int(training.get("amp_growth_interval", 4000)),
    )
    start_epoch = 1
    global_step = 0
    best_loss = float("inf")
    if resume:
        checkpoint = torch_load(resume, device)
        if checkpoint.get("architecture") != architecture:
            raise ValueError(f"Resume requires an exact {architecture} checkpoint.")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_loss = float(checkpoint.get("best_loss", best_loss))

    writer = (
        SummaryWriter(output_root / "tensorboard")
        if SummaryWriter is not None
        else None
    )
    log_interval = int(training.get("log_interval", 100))
    tensorboard_interval = int(training.get("tensorboard_interval", 200))
    strict_checks = bool(training.get("strict_gradient_checks", True))
    validation_interval = max(
        1,
        int(training.get("validation_interval", log_interval)),
    )
    print(
        f"{architecture}: device={device}, images={len(dataset)}, batches={len(loader)}, "
        f"AMP={amp_enabled}({amp_name}), parameters="
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sums = {
            name: torch.zeros((), device=device)
            for name in ("total", *criterion.loss_names)
        }
        sample_count = 0
        crop_counts: Counter[str] = Counter()
        slot_gain_sum = torch.zeros(model.num_slots, device=device)
        prior_center_sum = torch.zeros_like(slot_gain_sum)
        prior_width_sum = torch.zeros_like(slot_gain_sum)
        prior_ratio_sum = torch.zeros((), device=device)
        prior_sigma_sum = torch.zeros_like(slot_gain_sum)
        slot_usage_sum = torch.zeros_like(slot_gain_sum)
        dark_attention_sum = torch.zeros_like(slot_gain_sum)
        bright_attention_sum = torch.zeros_like(slot_gain_sum)
        white_attention_sum = torch.zeros_like(slot_gain_sum)
        halo_attention_sum = torch.zeros_like(slot_gain_sum)
        dark_pixels = torch.zeros((), device=device)
        bright_pixels = torch.zeros((), device=device)
        white_pixels = torch.zeros((), device=device)
        halo_pixels = torch.zeros((), device=device)
        dark_input_sum = torch.zeros((), device=device)
        dark_output_sum = torch.zeros((), device=device)
        bright_input_sum = torch.zeros((), device=device)
        bright_output_sum = torch.zeros((), device=device)
        white_input_sum = torch.zeros((), device=device)
        white_output_sum = torch.zeros((), device=device)
        halo_input_sum = torch.zeros((), device=device)
        halo_output_sum = torch.zeros((), device=device)
        halo_requested_sum = torch.zeros((), device=device)
        halo_floor_compensation_sum = torch.zeros((), device=device)
        halo_effective_sum = torch.zeros((), device=device)
        detail_gain_sum = torch.zeros(3, device=device)
        detail_confidence_sum = torch.zeros((), device=device)
        synthetic_recovery_pixels = torch.zeros((), device=device)
        synthetic_recovery_total = torch.zeros((), device=device)
        recovery_confidence_sum = torch.zeros((), device=device)
        preview_data = None

        for batch_index, (images, crop_modes) in enumerate(loader, start=1):
            report_batch = batch_index % log_interval == 0 or batch_index == len(loader)
            images = images.to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            recovery_settings = training.get("saturation_simulation", {})
            recovery_enabled = bool(
                recovery_settings.get("enabled", True)
                and criterion.weights["recovery"] > 0.0
            )
            if recovery_enabled:
                recovery_target, recovery_input, recovery_mask = simulate_saturation(
                    images,
                    recovery_settings,
                )
            else:
                recovery_target = recovery_input = recovery_mask = None
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                outputs = model(images)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                recovery_outputs = (
                    model.forward_recovery(recovery_input)
                    if recovery_input is not None
                    else None
                )
            # Keep numerically sensitive local statistics and logarithms in
            # FP32. Gradients still flow through the
            # cast into the BF16/FP16 model forward graph.
            loss_outputs = {
                key: outputs[key].float()
                if outputs[key].is_floating_point()
                else outputs[key]
                for key in criterion.required_output_names
            }
            recovery_loss_outputs = (
                {
                    key: recovery_outputs[key].float()
                    if recovery_outputs[key].is_floating_point()
                    else recovery_outputs[key]
                    for key in criterion.required_recovery_output_names
                }
                if recovery_outputs is not None
                else None
            )
            with torch.autocast(device_type=device.type, enabled=False):
                losses = criterion(
                    images.float(),
                    loss_outputs,
                    recovery_target=(
                        recovery_target.float()
                        if recovery_target is not None
                        else None
                    ),
                    recovery_outputs=recovery_loss_outputs,
                    recovery_mask=(
                        recovery_mask.float()
                        if recovery_mask is not None
                        else None
                    ),
                )
            if not bool(torch.isfinite(losses["total"])):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, batch {batch_index}."
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            try:
                gradient_norm = clip_grad_norm_(
                    model.parameters(),
                    float(training.get("gradient_clip", 5.0)),
                    error_if_nonfinite=True,
                )
            except RuntimeError as error:
                # The fused norm check is cheap in the normal path.  Only scan
                # individual parameter tensors when it actually reports a
                # failure, avoiding dozens of GPU synchronizations per batch.
                nonfinite_gradients = [
                    name
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                ]
                loss_text = ", ".join(
                    f"{name}={float(value.detach()):.6g}"
                    for name, value in losses.items()
                )
                raise FloatingPointError(
                    "Non-finite gradients after FP32 loss computation at "
                    f"epoch {epoch}, batch {batch_index}; first parameters="
                    f"{nonfinite_gradients[:8]}; losses: {loss_text}"
                ) from error
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.shape[0]
            sample_count += batch_size
            global_step += 1
            crop_counts.update(crop_modes)
            for name in loss_sums:
                loss_sums[name].add_(losses[name].detach(), alpha=batch_size)

            with torch.no_grad():
                input_value = outputs["input_luminance"].detach().float()
                output_value = outputs["output_luminance"].detach().float()
                dark_mask = outputs["dark_mask"].detach().bool()
                bright_mask = outputs["bright_mask"].detach().bool()
                white_mask = outputs["white_mask"].detach().bool()
                halo_mask = outputs["halo_mask"].detach().bool()
                dark_count = dark_mask.sum()
                bright_count = bright_mask.sum()
                white_count = white_mask.sum()
                halo_count = halo_mask.sum()
                dark_pixels += dark_count
                bright_pixels += bright_count
                white_pixels += white_count
                halo_pixels += halo_count
                dark_input_sum += (input_value * dark_mask).sum()
                dark_output_sum += (output_value * dark_mask).sum()
                bright_input_sum += (input_value * bright_mask).sum()
                bright_output_sum += (output_value * bright_mask).sum()
                white_input_sum += (
                    outputs["white_score"].detach().float() * white_mask
                ).sum()
                white_output_sum += (
                    outputs["enhanced"]
                    .detach()
                    .float()
                    .amin(dim=1, keepdim=True)
                    * white_mask
                ).sum()
                halo_input_sum += (input_value * halo_mask).sum()
                halo_output_sum += (output_value * halo_mask).sum()
                halo_requested_sum += (
                    outputs["halo_requested_delta"].detach().float() * halo_mask
                ).sum()
                halo_floor_compensation_sum += (
                    outputs["halo_floor_compensation"].detach().float()
                    * halo_mask
                ).sum()
                halo_effective_sum += (
                    outputs["halo_total_delta"].detach().float() * halo_mask
                ).sum()
                detail_gain_sum += (
                    outputs["detail_gain_maps"].detach().float().mean(dim=(2, 3)).sum(dim=0)
                )
                detail_confidence_sum += (
                    outputs["detail_confidence"].detach().float().mean()
                    * batch_size
                )
                if recovery_mask is not None and recovery_outputs is not None:
                    recovery_count = recovery_mask.sum()
                    synthetic_recovery_pixels += recovery_count
                    synthetic_recovery_total += recovery_mask.numel()
                    recovery_confidence_sum += (
                        recovery_outputs["recovery_confidence"].detach().float()
                        * recovery_mask
                    ).sum()
                attention = outputs["attn_map_full"].detach().float()
                slot_gain_sum += outputs["slot_gain"].detach().float().sum(dim=0)
                prior_center_sum += outputs["exposure_centers"].detach().float().sum(dim=0)
                prior_width_sum += outputs["exposure_bin_widths"].detach().float().sum(dim=0)
                prior_ratio_sum += outputs[
                    "adaptive_prior_ratio"
                ].detach().float().sum()
                prior_sigma_sum += outputs["exposure_prior_sigma"].detach().float().sum(dim=0)
                slot_usage_sum += attention.mean(dim=(2, 3)).sum(dim=0)
                dark_attention_sum += (attention * dark_mask).sum(dim=(0, 2, 3))
                bright_attention_sum += (attention * bright_mask).sum(dim=(0, 2, 3))
                white_attention_sum += (attention * white_mask).sum(dim=(0, 2, 3))
                halo_attention_sum += (attention * halo_mask).sum(dim=(0, 2, 3))

                validate_directions = strict_checks and (
                    batch_index % validation_interval == 0
                    or batch_index == len(loader)
                )
                if report_batch or validate_directions:
                    violation_counts = torch.stack(
                        (
                            ((output_value + 1e-6 < input_value) & dark_mask).sum(),
                            ((output_value > input_value + 1e-6) & bright_mask).sum(),
                            ((output_value > input_value + 1e-6) & white_mask).sum(),
                            ((output_value > input_value + 1e-6) & halo_mask).sum(),
                        )
                    )
                    if validate_directions and bool(violation_counts.any()):
                        counts = tuple(int(value) for value in violation_counts.cpu())
                        raise RuntimeError(
                            "Direction guarantee violated: "
                            f"dark={counts[0]}, bright={counts[1]}, "
                            f"core={counts[2]}, halo={counts[3]}."
                        )
                    if report_batch:
                        (
                            dark_violation,
                            bright_violation,
                            white_violation,
                            halo_violation,
                        ) = tuple(int(value) for value in violation_counts.cpu())

                if preview_data is None:
                    preview_data = (
                        images[:4].detach().cpu(),
                        {key: value[:4].detach().cpu() for key, value in outputs.items()},
                    )

            if writer is not None and global_step % tensorboard_interval == 0:
                for name, value in losses.items():
                    writer.add_scalar(f"loss/{name}", value, global_step)
                writer.add_scalar("optimization/gradient_norm", gradient_norm, global_step)
                writer.add_scalar(
                    "exposure/dark_output",
                    masked_mean(output_value, dark_mask),
                    global_step,
                )
                writer.add_scalar(
                    "exposure/bright_output",
                    masked_mean(output_value, bright_mask),
                    global_step,
                )
                writer.add_scalar(
                    "exposure/core_output",
                    masked_mean(outputs["enhanced"].float().amin(dim=1, keepdim=True), white_mask),
                    global_step,
                )
                writer.add_scalar(
                    "exposure/halo_output",
                    masked_mean(output_value, halo_mask),
                    global_step,
                )

            if report_batch:
                attention = outputs["attn_map_full"].detach().float()
                dark_attention = masked_slot_mean(attention, dark_mask)
                bright_attention = masked_slot_mean(attention, bright_mask)
                white_attention = masked_slot_mean(attention, white_mask)
                halo_attention = masked_slot_mean(attention, halo_mask)
                correction = outputs["correction_strength_map"].detach().float()
                delta = outputs["tone_delta"].detach().float()
                pixel_count = input_value.numel()
                print(
                    f"epoch {epoch:04d}/{epochs:04d} batch "
                    f"{batch_index:04d}/{len(loader):04d} "
                    f"loss={float(losses['total'].detach()):.6f} crop={format_crops(crop_modes)} "
                    f"dark_ratio={float(dark_mask.sum()) / pixel_count:.4f} "
                    f"bright_ratio={float(bright_mask.sum()) / pixel_count:.4f} "
                    f"core_ratio={float(white_mask.sum()) / pixel_count:.4f} "
                    f"halo_ratio={float(halo_mask.sum()) / pixel_count:.4f}"
                )
                print(
                    "  losses="
                    + " ".join(
                        f"{name}={float(losses[name].detach()):.5f}"
                        for name in criterion.loss_names
                    )
                )
                print(f"  slot_gain={format_slots(outputs['slot_gain'].mean(dim=0))} grad={float(gradient_norm):.3e}")
                print(
                    "  adaptive_prior="
                    f"centers={format_slots(outputs['exposure_centers'].mean(dim=0))} "
                    f"widths={format_slots(outputs['exposure_bin_widths'].mean(dim=0))} "
                    f"sigma={format_slots(outputs['exposure_prior_sigma'].mean(dim=0))} "
                    f"ratio={float(outputs['adaptive_prior_ratio'].mean()):.3f}"
                )
                print(f"  slot_usage={format_slots(attention.mean(dim=(0, 2, 3)))}")
                print(f"  dark_attention={format_slots(dark_attention)}")
                print(f"  bright_attention={format_slots(bright_attention)}")
                print(f"  core_attention={format_slots(white_attention)}")
                print(f"  halo_attention={format_slots(halo_attention)}")
                print(
                    "  correction_strength="
                    f"[min={float(correction.min()):+.4f}, "
                    f"max={float(correction.max()):+.4f}, "
                    f"mean={float(correction.mean()):+.4f}] "
                    f"tone_delta=[dark={float(masked_mean(delta, dark_mask)):+.4f}, "
                    f"bright={float(masked_mean(delta, bright_mask)):+.4f}, "
                    f"core={float(masked_mean(delta, white_mask)):+.4f}, "
                    f"halo={float(masked_mean(delta, halo_mask)):+.4f}]"
                )
                print(
                    "  luminance dark="
                    f"{float(masked_mean(input_value, dark_mask)):.4f}->"
                    f"{float(masked_mean(output_value, dark_mask)):.4f} "
                    "bright="
                    f"{float(masked_mean(input_value, bright_mask)):.4f}->"
                    f"{float(masked_mean(output_value, bright_mask)):.4f} "
                    "core_white_score="
                    f"{float(masked_mean(outputs['white_score'].detach().float(), white_mask)):.4f}->"
                    f"{float(masked_mean(outputs['enhanced'].detach().float().amin(dim=1, keepdim=True), white_mask)):.4f} "
                    "halo="
                    f"{float(masked_mean(input_value, halo_mask)):.4f}->"
                    f"{float(masked_mean(output_value, halo_mask)):.4f} "
                    "violations=(dark,bright,core,halo)="
                    f"({dark_violation},{bright_violation},{white_violation},{halo_violation})"
                )
                shoulder_bands = (
                    ("Y.30-.45", (input_value >= 0.30) & (input_value < 0.45)),
                    ("Y.45-.60", (input_value >= 0.45) & (input_value < 0.60)),
                    ("Y.60-.75", (input_value >= 0.60) & (input_value < 0.75)),
                    ("Y>.75", input_value >= 0.75),
                )
                print(
                    "  learned_tone_curve="
                    + " ".join(
                        f"{name}:{float(masked_mean(input_value, mask)):.3f}->"
                        f"{float(masked_mean(output_value, mask)):.3f}"
                        for name, mask in shoulder_bands
                    )
                    + " scales="
                    + f"bright_abs={float(masked_mean(outputs['bright_scale'].detach().float(), bright_mask)):.3f} "
                    + f"near_white_abs={float(masked_mean(outputs['white_scale'].detach().float(), white_mask)):.3f} "
                    + f"core_abs={float(masked_mean(outputs['core_scale'].detach().float(), white_mask)):.3f} "
                    + f"halo_relative={float(masked_mean(outputs['halo_scale'].detach().float(), halo_mask)):.3f}"
                )
                reliable = images.detach().float().amax(dim=1, keepdim=True) < float(
                    config["model"]["detail_saturation_ceiling"]
                )
                all_detail = detail_retention(input_value, output_value, reliable)
                bright_detail = detail_retention(
                    input_value, output_value, reliable & bright_mask
                )
                white_detail = detail_retention(
                    input_value, output_value, reliable & white_mask
                )
                halo_detail = detail_retention(
                    input_value, output_value, reliable & halo_mask
                )
                print(
                    "  detail_retention(correlation/amplitude)="
                    f"all={all_detail[0]:.4f}/{all_detail[1]:.3f} "
                    f"bright={bright_detail[0]:.4f}/{bright_detail[1]:.3f} "
                    f"core={white_detail[0]:.4f}/{white_detail[1]:.3f} "
                    f"halo={halo_detail[0]:.4f}/{halo_detail[1]:.3f} "
                    f"residual_abs={float(outputs['detail_residual'].detach().float().abs().mean()):.5f}"
                )
                print(
                    "  local_modulation="
                    f"[dark={float(masked_mean(outputs['dark_modulation'].detach().float(), dark_mask)):.3f}, "
                    f"bright={float(masked_mean(outputs['bright_modulation'].detach().float(), bright_mask)):.3f}, "
                    f"core={float(masked_mean(outputs['white_modulation'].detach().float(), white_mask)):.3f}, "
                    f"halo={float(masked_mean(outputs['halo_modulation'].detach().float(), halo_mask)):.3f}]"
                )
                print(
                    "  halo_neighbourhood="
                    f"excess={float(masked_mean(outputs['halo_context_excess'].detach().float(), halo_mask)):.4f} "
                    f"relative_delta={float(masked_mean(outputs['halo_relative_delta'].detach().float(), halo_mask)):.4f} "
                    f"veil_delta={float(masked_mean(outputs['halo_veil_delta'].detach().float(), halo_mask)):.4f} "
                    f"requested={float(masked_mean(outputs['halo_requested_delta'].detach().float(), halo_mask)):.4f} "
                    f"floor_comp={float(masked_mean(outputs['halo_floor_compensation'].detach().float(), halo_mask)):.4f} "
                    f"effective={float(masked_mean(outputs['halo_total_delta'].detach().float(), halo_mask)):.4f}"
                )
                detail_gains = outputs["detail_gain_maps"].detach().float()
                print(
                    "  retinex_detail_gain="
                    f"[fine={float(masked_mean(detail_gains[:, 0:1], reliable & bright_mask)):.3f}, "
                    f"medium={float(masked_mean(detail_gains[:, 1:2], reliable & bright_mask)):.3f}, "
                    f"coarse={float(masked_mean(detail_gains[:, 2:3], reliable & bright_mask)):.3f}]"
                )
                if recovery_mask is not None and recovery_outputs is not None:
                    print(
                        "  saturation_recovery="
                        f"mask_ratio={float(recovery_mask.mean()):.4f} "
                        f"confidence={float(masked_mean(recovery_outputs['recovery_confidence'].detach().float(), recovery_mask.bool())):.3f} "
                        f"rgb_l1={float(masked_mean((recovery_outputs['recovery_rgb'].detach().float() - recovery_target.float()).abs().mean(dim=1, keepdim=True), recovery_mask.bool())):.4f}"
                    )

        scheduler.step()
        averages = {
            name: float((value / sample_count).cpu())
            for name, value in loss_sums.items()
        }
        slot_gain_average = slot_gain_sum / sample_count
        prior_center_average = prior_center_sum / sample_count
        prior_width_average = prior_width_sum / sample_count
        prior_sigma_average = prior_sigma_sum / sample_count
        slot_usage_average = slot_usage_sum / sample_count
        dark_attention_average = dark_attention_sum / dark_pixels.clamp_min(1.0)
        bright_attention_average = bright_attention_sum / bright_pixels.clamp_min(1.0)
        white_attention_average = white_attention_sum / white_pixels.clamp_min(1.0)
        halo_attention_average = halo_attention_sum / halo_pixels.clamp_min(1.0)
        row: Dict[str, float] = {
            "epoch": float(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **averages,
            "dark_input_mean": float(
                (dark_input_sum / dark_pixels.clamp_min(1.0)).cpu()
            ),
            "dark_output_mean": float(
                (dark_output_sum / dark_pixels.clamp_min(1.0)).cpu()
            ),
            "bright_input_mean": float(
                (bright_input_sum / bright_pixels.clamp_min(1.0)).cpu()
            ),
            "bright_output_mean": float(
                (bright_output_sum / bright_pixels.clamp_min(1.0)).cpu()
            ),
            "core_input_mean": float(
                (white_input_sum / white_pixels.clamp_min(1.0)).cpu()
            ),
            "core_output_mean": float(
                (white_output_sum / white_pixels.clamp_min(1.0)).cpu()
            ),
            "halo_input_mean": float(
                (halo_input_sum / halo_pixels.clamp_min(1.0)).cpu()
            ),
            "halo_output_mean": float(
                (halo_output_sum / halo_pixels.clamp_min(1.0)).cpu()
            ),
            "halo_requested_mean": float(
                (halo_requested_sum / halo_pixels.clamp_min(1.0)).cpu()
            ),
            "halo_floor_compensation_mean": float(
                (
                    halo_floor_compensation_sum / halo_pixels.clamp_min(1.0)
                ).cpu()
            ),
            "halo_effective_mean": float(
                (halo_effective_sum / halo_pixels.clamp_min(1.0)).cpu()
            ),
            "detail_gain_fine": float(detail_gain_sum[0] / sample_count),
            "detail_gain_medium": float(detail_gain_sum[1] / sample_count),
            "detail_gain_coarse": float(detail_gain_sum[2] / sample_count),
            "detail_confidence": float(detail_confidence_sum / sample_count),
            "synthetic_recovery_ratio": float(
                synthetic_recovery_pixels
                / synthetic_recovery_total.clamp_min(1.0)
            ),
            "recovery_confidence": float(
                recovery_confidence_sum
                / synthetic_recovery_pixels.clamp_min(1.0)
            ),
            "adaptive_prior_ratio": float(prior_ratio_sum / sample_count),
            "adaptive_prior_sigma_mean": float(prior_sigma_average.mean()),
        }
        for mode in SAMPLING_MODES:
            row[f"crop_{mode}"] = float(crop_counts[mode])
        for index in range(model.num_slots):
            row[f"slot_{index + 1}_gain"] = float(slot_gain_average[index])
            row[f"slot_{index + 1}_prior_center"] = float(prior_center_average[index])
            row[f"slot_{index + 1}_prior_width"] = float(prior_width_average[index])
            row[f"slot_{index + 1}_prior_sigma"] = float(prior_sigma_average[index])
            row[f"slot_{index + 1}_usage"] = float(slot_usage_average[index])
            row[f"slot_{index + 1}_dark_attention"] = float(
                dark_attention_average[index]
            )
            row[f"slot_{index + 1}_bright_attention"] = float(
                bright_attention_average[index]
            )
            row[f"slot_{index + 1}_core_attention"] = float(
                white_attention_average[index]
            )
            row[f"slot_{index + 1}_halo_attention"] = float(
                halo_attention_average[index]
            )
        append_csv(output_root / "metrics_losses.csv", row)
        if writer is not None:
            for name, value in row.items():
                writer.add_scalar(f"epoch/{name}", value, epoch)

        state = {
            "architecture": architecture,
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": min(best_loss, averages["total"]),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
        }
        save_checkpoint(state, checkpoint_dir / "latest.pth")
        if averages["total"] < best_loss:
            best_loss = averages["total"]
            state["best_loss"] = best_loss
            save_checkpoint(state, checkpoint_dir / "best_loss.pth")
        if epoch % int(training.get("checkpoint_interval", 5)) == 0:
            save_checkpoint(state, checkpoint_dir / f"epoch_{epoch:04d}.pth")
        if (
            preview_data is not None
            and epoch % int(training.get("visualize_interval", 5)) == 0
        ):
            save_preview(*preview_data, preview_dir, epoch)
        print(
            f"epoch {epoch:04d} complete: loss={averages['total']:.6f}, "
            f"dark={row['dark_input_mean']:.4f}->{row['dark_output_mean']:.4f}, "
            f"bright={row['bright_input_mean']:.4f}->{row['bright_output_mean']:.4f}, "
            f"core={row['core_input_mean']:.4f}->{row['core_output_mean']:.4f}, "
            f"halo={row['halo_input_mean']:.4f}->{row['halo_output_mean']:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    train(parse_args())
