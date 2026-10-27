"""Small tensor-only adapter for inserting SlotDCE before a detector."""

from __future__ import annotations

from torch import Tensor, nn

from .slot_dce import SlotDCE


class SlotDCEDetectionAdapter(nn.Module):
    """Expose SlotDCE as a differentiable RGB preprocessing layer."""

    def __init__(self, enhancer: SlotDCE) -> None:
        super().__init__()
        self.enhancer = enhancer

    def forward(self, images: Tensor) -> Tensor:
        # Detection needs only the enhanced RGB tensor.  Avoid constructing the
        # standalone training diagnostics (masks, luminance maps and Slot data).
        return self.enhancer.enhance_for_detection(images)
