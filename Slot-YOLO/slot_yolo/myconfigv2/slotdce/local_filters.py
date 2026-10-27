"""Fast separable neighbourhood filters used by the SlotDCE renderer."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def _padding_mode(value: Tensor, padding: int) -> str:
    """Match the previous reflect/replicate boundary rule."""

    height, width = value.shape[-2:]
    return "reflect" if height > padding and width > padding else "replicate"


def local_mean(value: Tensor, kernel_size: int) -> Tensor:
    """Return a stride-one square box mean using two one-dimensional passes.

    A square box filter is separable.  Computing a 1xK horizontal mean followed
    by a Kx1 vertical mean is mathematically equivalent to the former KxK
    ``avg_pool2d`` call, while reducing its window work from K^2 to 2K.
    """

    if kernel_size == 1:
        return value
    padding = kernel_size // 2
    mode = _padding_mode(value, padding)
    horizontal = F.avg_pool2d(
        F.pad(value, (padding, padding, 0, 0), mode=mode),
        kernel_size=(1, kernel_size),
        stride=1,
    )
    return F.avg_pool2d(
        F.pad(horizontal, (0, 0, padding, padding), mode=mode),
        kernel_size=(kernel_size, 1),
        stride=1,
    )


def local_max(value: Tensor, kernel_size: int) -> Tensor:
    """Return a stride-one square maximum using exact separable max pooling."""

    if kernel_size == 1:
        return value
    padding = kernel_size // 2
    horizontal = F.max_pool2d(
        value,
        kernel_size=(1, kernel_size),
        stride=1,
        padding=(0, padding),
    )
    return F.max_pool2d(
        horizontal,
        kernel_size=(kernel_size, 1),
        stride=1,
        padding=(padding, 0),
    )

