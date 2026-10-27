"""EMA for a backbone whose convolution Parameters have multiple module aliases."""

import torch

from ultralytics.utils.torch_utils import ModelEMA, unwrap_model


def classification_parameter_ids(model):
    """Locate both actual Detect classification branches, independent of indices."""
    model = unwrap_model(model)
    head = model.model[-1]
    return {id(parameter) for name in ("cv3", "one2one_cv3")
            for branch in (getattr(head, name, None),) if branch is not None
            for parameter in branch.parameters()}


class UniqueModelEMA(ModelEMA):
    """Apply one EMA update per actual tensor, preserving independent BN buffers."""

    @classmethod
    def from_existing(cls, ema):
        # Preserve the already restored EMA weights, decay schedule and step count.
        result = cls.__new__(cls)
        result.__dict__.update(ema.__dict__)
        return result

    @torch.no_grad()
    def update(self, model):
        if not self.enabled:
            return
        self.updates += 1
        decay = self.decay(self.updates)
        source = unwrap_model(model).state_dict()
        # keep_vars returns the original Parameters/buffers, so identity detects
        # the shared raw/enhanced convs without conflating distinct BN tensors.
        seen = set()
        for name, value in self.ema.state_dict(keep_vars=True).items():
            if id(value) in seen:
                continue
            seen.add(id(value))
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source[name].detach(), alpha=1.0 - decay)
