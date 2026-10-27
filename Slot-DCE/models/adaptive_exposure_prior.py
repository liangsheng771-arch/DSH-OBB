"""Learned monotonic exposure bins for ordered Slots."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AdaptiveExposurePrior(nn.Module):
    """Predict positive exposure-bin widths from an image luminance histogram.

    Centers are midpoints of cumulative bins, so they are strictly ordered and
    bounded without sorting or a center-specific anchor. A uniform width floor
    and bounded logits prevent bin collapse. Zero output-layer initialization
    gives a stable equal-width partition at the first step.
    """

    def __init__(
        self,
        num_slots: int = 5,
        histogram_bins: int = 16,
        hidden_channels: int = 16,
        adaptation_ratio: float = 0.25,
        logit_limit: float = 1.0,
        width_to_sigma_ratio: float = 0.60,
    ) -> None:
        super().__init__()
        if num_slots < 2:
            raise ValueError("Adaptive exposure routing requires at least two Slots.")
        if histogram_bins < num_slots:
            raise ValueError("histogram_bins must be at least num_slots.")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")
        if not 0.0 <= adaptation_ratio < 1.0:
            raise ValueError("adaptation_ratio must lie in [0, 1).")
        if logit_limit <= 0.0:
            raise ValueError("logit_limit must be positive.")
        if width_to_sigma_ratio <= 0.0:
            raise ValueError("width_to_sigma_ratio must be positive.")

        self.num_slots = int(num_slots)
        self.histogram_bins = int(histogram_bins)
        self.adaptation_ratio = float(adaptation_ratio)
        self.logit_limit = float(logit_limit)
        self.width_to_sigma_ratio = float(width_to_sigma_ratio)
        self.register_buffer(
            "histogram_centers",
            torch.linspace(0.0, 1.0, histogram_bins, dtype=torch.float32),
        )
        descriptor_channels = histogram_bins + 2
        self.predictor = nn.Sequential(
            nn.Linear(descriptor_channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, num_slots),
        )

    def reset_output(self) -> None:
        """Initialize the predicted widths to the uniform partition."""

        output = self.predictor[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def _soft_histogram(self, value: Tensor) -> Tensor:
        samples = value.detach().float().flatten(2).squeeze(1)
        centers = self.histogram_centers.to(samples).view(1, -1, 1)
        bin_width = 1.0 / max(self.histogram_bins - 1, 1)
        membership = F.relu(1.0 - (samples.unsqueeze(1) - centers).abs() / bin_width)
        histogram = membership.mean(dim=2)
        return histogram / histogram.sum(dim=1, keepdim=True).clamp_min(1.0e-6)

    def forward(
        self,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if value.ndim != 4 or value.shape[1] != 1:
            raise ValueError("AdaptiveExposurePrior expects [B,1,H,W] values.")

        histogram = self._soft_histogram(value)
        samples = value.detach().float().flatten(2)
        mean = samples.mean(dim=2)
        std = samples.var(dim=2, unbiased=False).clamp_min(0.0).sqrt()
        descriptor = torch.cat((histogram, mean, std), dim=1)
        raw_logits = self.predictor(descriptor.to(self.predictor[0].weight.dtype))
        bounded_logits = self.logit_limit * torch.tanh(raw_logits)
        adaptive_distribution = torch.softmax(bounded_logits, dim=1)
        uniform_width = (1.0 - self.adaptation_ratio) / self.num_slots
        widths = uniform_width + self.adaptation_ratio * adaptive_distribution
        right_edges = widths.cumsum(dim=1)
        left_edges = right_edges - widths
        centers = left_edges + 0.5 * widths
        sigmas = self.width_to_sigma_ratio * widths
        return (
            centers.to(value.dtype),
            widths.to(value.dtype),
            sigmas.to(value.dtype),
        )
