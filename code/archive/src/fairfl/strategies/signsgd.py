import torch
from pydantic import BaseModel, Field

from fairfl.core.registry import register_strategy
from fairfl.strategies.base import Strategy


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

    def aggregate(self, rnd, gparams, results):
        lr = self.train_cfg.lr
        vote = torch.zeros_like(gparams)
        for r in results:
            g = -(r.params - gparams) / lr
            if self.p.b > 0:
                noise = torch.from_numpy(self.rng.standard_normal(g.numel()).astype("float32")).to(g.device)
                g = g + self.p.b * noise
            vote += torch.sign(g)
        return gparams - self.p.step * torch.sign(vote)
