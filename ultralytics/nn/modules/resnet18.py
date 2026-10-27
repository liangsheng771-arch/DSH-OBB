"""ResNet-18-vd stages for the Slot-YOLO RT-DETR backbone.

Architecture follows RT-DETR's PResNet(depth=18, variant='d'): a three-convolution
stem, [2, 2, 2, 2] basic blocks, ReLU and average-pool projection shortcuts.
Reference: https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/src/nn/backbone/presnet.py
"""

import torch
from torch import nn

from .conv import Conv


class ResNet18BasicBlock(nn.Module):
    """Two 3x3 convolutions with the ResNet-vd projection shortcut."""

    def __init__(self, c1: int, c2: int, stride: int = 1, projection: bool = False):
        super().__init__()
        self.conv1 = Conv(c1, c2, 3, stride, act=nn.ReLU())
        self.conv2 = Conv(c2, c2, 3, 1, act=False)
        if projection or stride != 1 or c1 != c2:
            self.shortcut = nn.Sequential(
                *([nn.AvgPool2d(2, 2, ceil_mode=True)] if stride == 2 else []),
                Conv(c1, c2, 1, 1, act=False),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


class ResNet18Layer(nn.Module):
    """One parsed stem/stage with explicit channel counts and no width scaling."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 2):
        super().__init__()
        self.is_first = is_first
        if is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2 // 2, 3, 2, act=nn.ReLU()),
                Conv(c2 // 2, c2 // 2, 3, 1, act=nn.ReLU()),
                Conv(c2 // 2, c2, 3, 1, act=nn.ReLU()),
                nn.MaxPool2d(3, 2, 1),
            )
        else:
            self.layer = nn.Sequential(
                ResNet18BasicBlock(c1, c2, s, projection=True),
                *(ResNet18BasicBlock(c2, c2) for _ in range(n - 1)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
