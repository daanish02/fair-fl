import numpy as np
import torch
from pydantic import BaseModel, Field

from fairfl.core.registry import register_strategy
from fairfl.strategies.base import Strategy


class MedianParams(BaseModel):
    sigma: float = Field(0.0, ge=0.0, description="Std of symmetric Gaussian noise added to each update before the "
                                                  "median (the paper's correction; 0 = plain coordinate-wise median).")


@register_strategy("median")
class Median(Strategy):
    """Chen et al., NeurIPS 2020 (Algorithm 2 medianSGD with noise-perturbed correction), applied to model updates."""

    Params = MedianParams

    def aggregate(self, rnd, gparams, results):
        deltas = np.stack([(r.params - gparams).cpu().numpy() for r in results])
        if self.p.sigma > 0:
            deltas = deltas + self.rng.normal(0.0, self.p.sigma, size=deltas.shape).astype(deltas.dtype)
        return gparams + torch.from_numpy(np.median(deltas, axis=0).astype(np.float32)).to(gparams.device)
