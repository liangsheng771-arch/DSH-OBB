"""Small RGB loading and image-grid helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def luminance(image: Tensor) -> Tensor:
    """Return ITU-R BT.601 luminance for an RGB tensor."""

    coefficients = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (image * coefficients).sum(dim=1, keepdim=True)


def white_score(image: Tensor) -> Tensor:
    """Return min(R,G,B), which is high for low-saturation white pixels."""

    return image.amin(dim=1, keepdim=True)


def load_rgb_tensor(path: Union[str, Path]) -> Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB")
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div(255.0)


def save_rgb_tensor(tensor: Tensor, path: Union[str, Path]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray(np.round(array * 255).astype(np.uint8)).save(destination)


def save_tensor_grid(tensors: Tensor, path: Union[str, Path], nrow: int) -> None:
    if tensors.ndim != 4 or tensors.shape[1] not in (1, 3) or nrow <= 0:
        raise ValueError("Expected [B, 1|3, H, W] tensors and a positive nrow.")
    values = tensors.detach().float().cpu().clamp(0, 1)
    if values.shape[1] == 1:
        values = values.repeat(1, 3, 1, 1)
    rows = int(np.ceil(values.shape[0] / nrow))
    height, width = values.shape[-2:]
    canvas = torch.zeros(3, rows * height, nrow * width)
    for index, value in enumerate(values):
        row, column = divmod(index, nrow)
        canvas[:, row * height:(row + 1) * height, column * width:(column + 1) * width] = value
    save_rgb_tensor(canvas, path)


def save_comparison(input_image: Tensor, enhanced: Tensor, path: Union[str, Path]) -> None:
    source = input_image.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    output = enhanced.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined = np.concatenate((source, output), axis=1)
    Image.fromarray(np.round(combined * 255).astype(np.uint8)).save(destination)
