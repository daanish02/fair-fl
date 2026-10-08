from __future__ import annotations

import numpy as np
import torch
from pydantic import BaseModel, Field

from fairfl.registry import register_strategy
from fairfl.strategies.base import Strategy


class MedianParams(BaseModel):
    sigma: float = Field(0.0, ge=0.0, description="Std of symmetric Gaussian noise added to each update before the "
                                                  "median (the paper's correction; 0 = plain coordinate-wise median).")


@register_strategy("median")
class Median(Strategy):
    """Chen et al., NeurIPS 2020 (Algorithm 2 medianSGD with noise-perturbed correction), applied to model updates."""

    Params = MedianParams

    def aggregate(self, rnd, global_state, results):
        out = {}
        for key, g in global_state.items():
            if not g.is_floating_point():
                out[key] = results[0].state[key]
                continue
            deltas = np.stack([(r.state[key] - g).cpu().numpy() for r in results])
            if self.p.sigma > 0:
                deltas = deltas + self.rng.normal(0.0, self.p.sigma, size=deltas.shape).astype(deltas.dtype)
            out[key] = g + torch.from_numpy(np.median(deltas, axis=0)).to(g.dtype).to(g.device)
        return out
