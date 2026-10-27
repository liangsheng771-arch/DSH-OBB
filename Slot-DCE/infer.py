"""Run current SlotDCE illumination correction."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from config_utils import build_model, choose_device, load_config, load_model_checkpoint
from datasets import find_images
from image_utils import save_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config", default="config/slotDCE_v9_8_detector_lite.yaml"
    )
    parser.add_argument("--output", default="outputs/infer")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-side", type=int)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def limited_size(width: int, height: int, max_side: int) -> Tuple[int, int]:
    if max_side <= 0 or max(width, height) <= max_side:
        return width, height
    scale = max_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def load_tensor(path: Path, max_side: int) -> Tensor:
    with Image.open(path) as source:
        target = limited_size(source.width, source.height, max_side)
        source.draft("RGB", target)
        image = source.convert("RGB")
    if image.size != target:
        image = image.resize(target, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div(255.0)


class InferenceDataset(Dataset[Tuple[Tensor, Path]]):
    def __init__(self, paths: Sequence[Path], max_side: int) -> None:
        self.paths = list(paths)
        self.max_side = max_side

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Path]:
        path = self.paths[index]
        return load_tensor(path, self.max_side), path


def collate_same_size(
    batch: Sequence[Tuple[Tensor, Path]],
) -> Tuple[Tensor, List[Path]]:
    images, paths = zip(*batch)
    return torch.stack(list(images)), list(paths)


def group_by_resolution(paths: Sequence[Path], max_side: int) -> List[List[Path]]:
    groups: DefaultDict[Tuple[int, int], List[Path]] = defaultdict(list)
    for path in paths:
        with Image.open(path) as image:
            width, height = limited_size(image.width, image.height, max_side)
        groups[(height, width)].append(path)
    return list(groups.values())


def relative_output(source: Path, input_root: Path) -> Path:
    relative = Path(source.name) if input_root.is_file() else source.relative_to(
        input_root.resolve()
    )
    return relative.with_suffix("")


def infer(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    inference = config.get("inference", {})
    device = choose_device(args.device)
    model = build_model(config["model"]).to(device)
    checkpoint = load_model_checkpoint(model, args.checkpoint, device)
    model.eval()

    channels_last = bool(inference.get("channels_last", True)) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
    max_side = (
        int(args.max_side)
        if args.max_side is not None
        else int(inference.get("max_side", 1280))
    )
    if max_side < 0:
        raise ValueError("max_side must be zero or positive.")
    batch_size = int(args.batch_size or inference.get("batch_size", 1))
    amp_enabled = device.type == "cuda" and bool(inference.get("mixed_precision", True))
    input_root = Path(args.input).expanduser()
    paths = find_images(input_root)
    if not paths:
        raise RuntimeError("No supported input images found.")
    output_root = Path(args.output).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        f"{checkpoint.get('architecture', 'SlotDCE')} inference: "
        f"checkpoint_epoch={checkpoint.get('epoch', 'n/a')}, "
        f"images={len(paths)}, device={device}, AMP={amp_enabled}, "
        f"max_side={max_side or 'original'}"
    )
    elapsed = 0.0
    processed = 0
    with torch.inference_mode():
        for group in group_by_resolution(paths, max_side):
            loader = DataLoader(
                InferenceDataset(group, max_side),
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(inference.get("num_workers", 0)),
                pin_memory=device.type == "cuda",
                collate_fn=collate_same_size,
            )
            for images, batch_paths in loader:
                images = images.to(device, non_blocking=True)
                if channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    outputs = model(images)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed += time.perf_counter() - start
                processed += images.shape[0]
                originals = images.detach().float().cpu()
                enhanced = outputs["enhanced"].detach().float().cpu()
                for index, source in enumerate(batch_paths):
                    stem = relative_output(source, input_root)
                    save_comparison(
                        originals[index],
                        enhanced[index],
                        output_root / stem.parent / f"{stem.name}_comparison.png",
                    )
    seconds = elapsed / max(processed, 1)
    print(
        f"Processed {processed} image(s), model-only {seconds * 1000:.2f} ms/image, "
        f"{1.0 / seconds if seconds else float('inf'):.2f} FPS."
    )
    print(f"Comparisons: {output_root.resolve()}")


if __name__ == "__main__":
    infer(parse_args())
