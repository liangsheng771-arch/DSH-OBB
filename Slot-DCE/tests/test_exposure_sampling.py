"""Tests for mixed exposure-aware sampling and safe fallbacks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from datasets import UndergroundDataset


def save_rgb(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def dataset(root: Path, mode: str) -> UndergroundDataset:
    probabilities = {name: float(name == mode) for name in ("full_resize", "random", "white", "mixed")}
    return UndergroundDataset(
        root,
        image_size=64,
        crop_size=64,
        scale_range=(1.0, 1.0),
        horizontal_flip_probability=0.0,
        sampling_probabilities=probabilities,
        exposure_thumbnail_size=64,
        crop_dark_threshold=0.30,
        crop_white_threshold=0.70,
        crop_min_pixels=1,
    )


def test_full_resize_keeps_shape(tmp_path: Path) -> None:
    save_rgb(tmp_path / "frame.png", np.full((80, 160, 3), 96, np.uint8))
    tensor, mode = dataset(tmp_path, "full_resize")[0]
    assert tensor.shape == (3, 64, 64)
    assert mode == "full_resize"


def test_white_and_mixed_modes_find_regions(tmp_path: Path) -> None:
    image = np.zeros((96, 192, 3), np.uint8)
    image[:, 96:] = 255
    save_rgb(tmp_path / "mixed.png", image)
    assert dataset(tmp_path, "white")[0][1] == "white"
    assert dataset(tmp_path, "mixed")[0][1] == "mixed"


def test_missing_regions_fall_back_to_random(tmp_path: Path) -> None:
    save_rgb(tmp_path / "dark.png", np.full((96, 192, 3), 64, np.uint8))
    white_tensor, white_mode = dataset(tmp_path, "white")[0]
    mixed_tensor, mixed_mode = dataset(tmp_path, "mixed")[0]
    assert white_tensor.shape == mixed_tensor.shape == (3, 64, 64)
    assert white_mode == mixed_mode == "random"
