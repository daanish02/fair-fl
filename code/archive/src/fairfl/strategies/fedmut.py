import random

import torch
from pydantic import BaseModel, Field

from fairfl.core.params import weighted_mean
from fairfl.core.registry import register_strategy
from fairfl.core.types import FitIns
from fairfl.strategies.base import Strategy


class FedMutParams(BaseModel):
    radius: float = Field(4.0, gt=0.0, description="Mutation range alpha.")
    mut_acc_rate: float = Field(0.3, ge=0.0, le=1.0, description="Initial acceleration (asymmetry) of mutation signs.")
    mut_bound: int = Field(50, ge=1, description="Rounds over which the acceleration decays to 0.")
    weighted: bool = Field(False, description="Weight by sample count (official code averages uniformly).")


def mutation_signs(num_layers: int, m: int, ctrl_rate: float, rnd: random.Random) -> list[list[float]]:
    """Per layer, m/2 pairs of opposite-ish signs (1, -1 + ctrl_rate), shuffled across clients (official code)."""
    out = []
    for _ in range(num_layers):
        ctrl = []
        for _ in range(m // 2):
            pair = [1.0, -1.0 + ctrl_rate]
            ctrl += pair if rnd.random() > 0.5 else pair[::-1]
        rnd.shuffle(ctrl)
        out.append(ctrl)
    return out


@register_strategy("fedmut")
class FedMut(Strategy):
    """Hu et al., AAAI 2024: each client starts from a distinct mutation w_glob + alpha * sign * (w_glob - w_old).

    Port of `mutation_spread` in github.com/HMHelloWorld/FedMut (Algorithm/Training_FedMut.py).
    """

    Params = FedMutParams

    def initialize(self, gparams, num_clients, client_sizes, layer_sizes):
        super().initialize(gparams, num_clients, client_sizes, layer_sizes)
        self.pending: list[torch.Tensor] | None = None
        self.pyrng = random.Random(int(self.rng.integers(2**31)))

    def configure_round(self, rnd, gparams, ids, pre_eval):
        chosen = self.sample_clients(rnd, ids)
        models = self.pending if self.pending is not None and len(self.pending) == len(chosen) else [gparams] * len(chosen)
        return {cid: FitIns(params=w) for cid, w in zip(chosen, models)}

    def aggregate(self, rnd, gparams, results):
        weights = [r.num_samples if self.p.weighted else 1.0 for r in results]
        w_glob = weighted_mean([r.params for r in results], weights)
        delta = w_glob - gparams
        m = len(results)
        ctrl_rate = self.p.mut_acc_rate * (1.0 - min(rnd / self.p.mut_bound, 1.0))
        signs = mutation_signs(len(self.layer_sizes), m, ctrl_rate, self.pyrng)
        bounds = torch.tensor([0] + self.layer_sizes).cumsum(0).tolist()
        models = []
        for j in range(m):
            w = w_glob.clone()
            if not (j == m - 1 and m % 2 == 1):
                for k in range(len(self.layer_sizes)):
                    a, b = bounds[k], bounds[k + 1]
                    w[a:b] += delta[a:b] * signs[k][j] * self.p.radius
            models.append(w)
        self.pending = models
        return w_glob
