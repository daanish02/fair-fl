from __future__ import annotations

import torch
from pydantic import BaseModel, Field

from fairfl.registry import register_strategy
from fairfl.strategies.base import Strategy

# BatchNorm running stats are floating-point buffers, not gradients: a client's reported delta for
# these reflects local forward-pass statistics, not a step to vote on the sign of. Sign-voting them
# would move them by a fixed +/-step regardless of magnitude, corrupting the running averages.
_BN_BUFFER_SUFFIXES = ("running_mean", "running_var")


class SignSGDParams(BaseModel):
    step: float = Field(0.001, gt=0.0, description="Server step delta: x <- x - delta * sign(sum_i sign(g_i + b xi_i)).")
    b: float = Field(0.0, ge=0.0, description="Std of Gaussian noise added to each client's gradient before its sign "
                                              "(Noisy signSGD, Chen et al. Algorithm 3).")


@register_strategy("signsgd")
class SignSGD(Strategy):
    """signSGD with majority vote (Bernstein et al. 2018), noisy variant of Chen et al., NeurIPS 2020 (Algorithm 3).

    Run with one full-batch local step per round (train.local_epochs 1, train.batch_size >= client size); the client
    gradient is recovered as g_i = -(w_i - w) / lr, so the client lr only rescales it.
    """

    Params = SignSGDParams

    def aggregate(self, rnd, global_state, results, round_lr):
        lr = round_lr  # gradient recovery must use the lr clients actually trained with (cosine-decayed per round)
        out = {}
        for key, g in global_state.items():
            if not g.is_floating_point():
                out[key] = results[0].state[key]
                continue
            if key.endswith(_BN_BUFFER_SUFFIXES):
                weights = [r.num_samples for r in results]
                out[key] = sum(r.state[key] * w for r, w in zip(results, weights)) / sum(weights)
                continue
            vote = torch.zeros_like(g)
            for r in results:
                grad = -(r.state[key] - g) / lr
                if self.p.b > 0:
                    noise = torch.from_numpy(self.rng.standard_normal(grad.shape).astype("float32")).to(grad.device)
                    grad = grad + self.p.b * noise
                vote += torch.sign(grad)
            out[key] = g - self.p.step * torch.sign(vote)
        return out
