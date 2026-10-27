"""Resize illumination evidence by coverage when reducing spatial resolution."""

from torch.nn import functional as F


def resize_guidance(value, size):
    size = tuple(size)
    if value.shape[-2:] == size:
        return value
    if all(target <= source for target, source in zip(size, value.shape[-2:])):
        return F.interpolate(value, size=size, mode="area")
    return F.interpolate(value, size=size, mode="bilinear", align_corners=False)
