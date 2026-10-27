"""Mine-image dataset with mixed exposure-aware training crops."""

from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageStat
from torch import Tensor
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SAMPLING_MODES = ("full_resize", "random", "white", "mixed")


def find_images(path: Union[str, Path]) -> List[Path]:
    root = Path(path).expanduser()
    if root.is_file():
        if root.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {root.suffix}")
        return [root.resolve()]
    if not root.is_dir():
        raise FileNotFoundError(f"Image path does not exist: {root}")
    return sorted(
        item.resolve()
        for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


class UndergroundDataset(Dataset[Tuple[Tensor, str]]):
    """Return exposure-aware training crops from unpaired RGB images."""

    def __init__(
        self,
        root: Union[str, Path],
        image_size: int = 384,
        *,
        crop_size: int,
        sampling_probabilities: Mapping[str, float],
        scale_range: Sequence[float] = (0.9, 1.1),
        rotation_degrees: float = 0.0,
        horizontal_flip_probability: float = 0.5,
        exposure_thumbnail_size: int = 96,
        crop_dark_threshold: float = 0.30,
        crop_white_threshold: float = 0.70,
        crop_min_pixels: int = 1,
        crop_jitter_fraction: float = 0.20,
        cache_images: bool = False,
        cache_short_side: int = 384,
        cache_workers: int = 8,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        if len(scale_range) != 2 or not 0 < scale_range[0] <= scale_range[1]:
            raise ValueError("scale_range must contain two ordered positive values.")
        if rotation_degrees < 0.0:
            raise ValueError("rotation_degrees must be non-negative.")
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must lie in [0, 1].")
        if exposure_thumbnail_size <= 0 or crop_min_pixels <= 0:
            raise ValueError("thumbnail size and min pixels must be positive.")
        if not 0.0 <= crop_dark_threshold < crop_white_threshold <= 1.0:
            raise ValueError("Invalid exposure crop thresholds.")
        if not 0.0 <= crop_jitter_fraction < 0.5:
            raise ValueError("crop_jitter_fraction must lie in [0, 0.5).")
        if cache_short_side <= 0 or cache_workers <= 0:
            raise ValueError("Cache settings must be positive.")

        self.image_paths = find_images(root)
        if not self.image_paths:
            raise RuntimeError(f"No supported images found under {Path(root)}")
        self.image_size = int(image_size)
        self.crop_size = int(crop_size)
        if self.crop_size <= 0:
            raise ValueError("crop_size must be positive.")
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.rotation_degrees = float(rotation_degrees)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.exposure_thumbnail_size = int(exposure_thumbnail_size)
        self.crop_dark_threshold = float(crop_dark_threshold)
        self.crop_white_threshold = float(crop_white_threshold)
        self.crop_min_pixels = int(crop_min_pixels)
        self.crop_jitter_fraction = float(crop_jitter_fraction)
        self.cache_short_side = int(cache_short_side)

        self.sampling_probabilities = self._validate_probabilities(
            sampling_probabilities
        )
        self.cached_images: Optional[List[np.ndarray]] = None
        if cache_images:
            self.cached_images = self._build_cache(cache_workers)

    @staticmethod
    def _validate_probabilities(
        probabilities: Mapping[str, float],
    ) -> Tuple[Tuple[str, float], ...]:
        unknown = set(probabilities) - set(SAMPLING_MODES)
        if unknown:
            raise ValueError(f"Unknown sampling mode(s): {sorted(unknown)}")
        values = tuple(
            (mode, float(probabilities.get(mode, 0.0)))
            for mode in SAMPLING_MODES
        )
        if any(value < 0.0 for _, value in values):
            raise ValueError("Sampling probabilities must be non-negative.")
        if abs(sum(value for _, value in values) - 1.0) > 1e-6:
            raise ValueError("Sampling probabilities must sum to 1.0.")
        return values

    def __len__(self) -> int:
        return len(self.image_paths)

    def _decode_for_cache(self, path: Path) -> np.ndarray:
        with Image.open(path) as source:
            image = source.convert("RGB")
        short_side = min(image.size)
        if short_side > self.cache_short_side:
            scale = self.cache_short_side / short_side
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.BILINEAR,
            )
        return np.asarray(image, dtype=np.uint8).copy()

    def _build_cache(self, workers: int) -> List[np.ndarray]:
        count = min(workers, len(self.image_paths), os.cpu_count() or 1)
        print(f"Caching {len(self.image_paths)} image(s) with {count} thread(s)...")
        with ThreadPoolExecutor(max_workers=count) as executor:
            images = list(executor.map(self._decode_for_cache, self.image_paths))
        size = sum(image.nbytes for image in images) / 1024**3
        print(f"RAM image cache ready: {size:.2f} GiB.")
        return images

    def _select_mode(self) -> str:
        draw = random.random()
        cumulative = 0.0
        for mode, probability in self.sampling_probabilities:
            cumulative += probability
            if draw < cumulative:
                return mode
        return self.sampling_probabilities[-1][0]

    def _full_resize(self, image: Image.Image) -> Image.Image:
        scale = min(
            self.image_size / max(image.width, 1),
            self.image_size / max(image.height, 1),
        )
        size = (
            max(1, min(self.image_size, round(image.width * scale))),
            max(1, min(self.image_size, round(image.height * scale))),
        )
        resized = image.resize(size, Image.Resampling.BILINEAR)
        fill = tuple(round(value) for value in ImageStat.Stat(resized).mean[:3])
        canvas = Image.new("RGB", (self.image_size, self.image_size), fill)
        canvas.paste(
            resized,
            ((self.image_size - resized.width) // 2, (self.image_size - resized.height) // 2),
        )
        return canvas

    def _ensure_crop_size(self, image: Image.Image, crop_size: int) -> Image.Image:
        if image.width >= crop_size and image.height >= crop_size:
            return image
        scale = max(crop_size / image.width, crop_size / image.height)
        return image.resize(
            (
                max(crop_size, round(image.width * scale)),
                max(crop_size, round(image.height * scale)),
            ),
            Image.Resampling.BILINEAR,
        )

    def _exposure_masks(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        scale = min(1.0, self.exposure_thumbnail_size / max(image.size))
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        value = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        white = rgb.min(axis=-1) > self.crop_white_threshold
        masks = np.stack(
            ((value < self.crop_dark_threshold) & ~white, white)
        ).astype(np.float32)
        pooled = torch.nn.functional.adaptive_max_pool2d(
            torch.from_numpy(masks).unsqueeze(0),
            output_size=(size[1], size[0]),
        )[0]
        return pooled[0].numpy() > 0, pooled[1].numpy() > 0

    @staticmethod
    def _window_sums(mask: np.ndarray, height: int, width: int) -> np.ndarray:
        integral = np.pad(mask.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return (
            integral[height:, width:]
            - integral[:-height, width:]
            - integral[height:, :-width]
            + integral[:-height, :-width]
        )

    def _white_origin(
        self, image: Image.Image, crop_size: int
    ) -> Optional[tuple[int, int]]:
        _, white = self._exposure_masks(image)
        if int(white.sum()) < self.crop_min_pixels:
            return None
        row, column = random.choice(np.argwhere(white).tolist())
        center_y = (row + random.random()) * image.height / white.shape[0]
        center_x = (column + random.random()) * image.width / white.shape[1]
        jitter = self.crop_jitter_fraction * crop_size
        top = round(center_y - crop_size / 2 + random.uniform(-jitter, jitter))
        left = round(center_x - crop_size / 2 + random.uniform(-jitter, jitter))
        return (
            min(max(top, 0), image.height - crop_size),
            min(max(left, 0), image.width - crop_size),
        )

    def _mixed_origin(
        self, image: Image.Image, crop_size: int
    ) -> Optional[tuple[int, int]]:
        dark, white = self._exposure_masks(image)
        if int(dark.sum()) < self.crop_min_pixels or int(white.sum()) < self.crop_min_pixels:
            return None
        height = min(dark.shape[0], max(1, round(crop_size * dark.shape[0] / image.height)))
        width = min(dark.shape[1], max(1, round(crop_size * dark.shape[1] / image.width)))
        valid = (
            (self._window_sums(dark, height, width) >= self.crop_min_pixels)
            & (self._window_sums(white, height, width) >= self.crop_min_pixels)
        )
        candidates = np.argwhere(valid)
        if not len(candidates):
            return None
        row, column = candidates[random.randrange(len(candidates))]
        row_range = dark.shape[0] - height
        column_range = dark.shape[1] - width
        top = 0 if row_range == 0 else round(row / row_range * (image.height - crop_size))
        left = 0 if column_range == 0 else round(column / column_range * (image.width - crop_size))
        return top, left

    def _crop(self, image: Image.Image, crop_size: int, requested: str) -> tuple[Image.Image, str]:
        if requested == "white":
            origin = self._white_origin(image, crop_size)
        elif requested == "mixed":
            origin = self._mixed_origin(image, crop_size)
        else:
            origin = None
        if origin is None:
            top = random.randint(0, image.height - crop_size)
            left = random.randint(0, image.width - crop_size)
            actual = "random"
        else:
            top, left = origin
            actual = requested
        return image.crop((left, top, left + crop_size, top + crop_size)), actual

    def _augment(self, image: Image.Image) -> tuple[Image.Image, str]:
        requested = self._select_mode()
        if requested == "full_resize":
            result = self._full_resize(image)
            actual = "full_resize"
        else:
            scale = random.uniform(*self.scale_range)
            crop_size = max(1, round(self.crop_size / scale))
            image = self._ensure_crop_size(image, crop_size)
            if self.rotation_degrees:
                angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
                fill = tuple(round(value) for value in ImageStat.Stat(image).mean[:3])
                image = image.rotate(
                    angle,
                    resample=Image.Resampling.BILINEAR,
                    fillcolor=fill,
                )
            result, actual = self._crop(image, crop_size, requested)
            if result.size != (self.image_size, self.image_size):
                result = result.resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
        if random.random() < self.horizontal_flip_probability:
            result = result.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return result, actual

    def __getitem__(self, index: int) -> Tuple[Tensor, str]:
        if self.cached_images is None:
            with Image.open(self.image_paths[index]) as source:
                image = source.convert("RGB")
        else:
            image = Image.fromarray(self.cached_images[index], mode="RGB")
        image, mode = self._augment(image)
        array = np.asarray(image, dtype=np.float32).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).div(255.0)
        return tensor, mode
