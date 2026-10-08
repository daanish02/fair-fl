from __future__ import annotations

import random

from pydantic import BaseModel, Field

from fairfl.registry import register_strategy
from fairfl.strategies.base import FitIns, Strategy, weighted_mean


class FedMutParams(BaseModel):
    radius: float = Field(4.0, gt=0.0, description="Mutation range alpha.")
    mut_acc_rate: float = Field(0.3, ge=0.0, le=1.0, description="Initial acceleration (asymmetry) of mutation signs.")
    mut_bound: int = Field(50, ge=1, description="Rounds over which the acceleration decays to 0.")
    weighted: bool = Field(False, description="Weight by sample count (official code averages uniformly).")


def _mutation_signs(keys: list[str], m: int, ctrl_rate: float, rnd: random.Random) -> dict[str, list[float]]:
    """Per layer key, m/2 pairs of opposite-ish signs (1, -1 + ctrl_rate), shuffled across clients."""
    out = {}
    for key in keys:
        ctrl = []
        for _ in range(m // 2):
            pair = [1.0, -1.0 + ctrl_rate]
            ctrl += pair if rnd.random() > 0.5 else pair[::-1]
        rnd.shuffle(ctrl)
        out[key] = ctrl
    return out


@register_strategy("fedmut")
class FedMut(Strategy):
    """Hu et al., AAAI 2024: each client starts from a distinct mutation w_glob + alpha * sign * (w_glob - w_old).

    Port of `mutation_spread` in github.com/HMHelloWorld/FedMut (Algorithm/Training_FedMut.py).
    """

    Params = FedMutParams

    def initialize(self, global_state, num_clients, client_sizes):
        super().initialize(global_state, num_clients, client_sizes)
        self.pending: list[dict] | None = None
        self.pyrng = random.Random(int(self.rng.integers(2**31)))

    def configure_round(self, rnd, global_state, ids, pre_eval, round_lr):
        chosen = self.sample_clients(ids)
        states = self.pending if self.pending is not None and len(self.pending) == len(chosen) \
            else [global_state] * len(chosen)
        return {cid: FitIns(state=w, config={"lr": round_lr}) for cid, w in zip(chosen, states)}

    def aggregate(self, rnd, global_state, results, round_lr):
        weights = [r.num_samples if self.p.weighted else 1.0 for r in results]
        w_glob = weighted_mean([r.state for r in results], weights)
        m = len(results)
        ctrl_rate = self.p.mut_acc_rate * (1.0 - min(rnd / self.p.mut_bound, 1.0))
        float_keys = [k for k, v in global_state.items() if v.is_floating_point()]
        signs = _mutation_signs(float_keys, m, ctrl_rate, self.pyrng)
        states = []
        for j in range(m):
            w = {k: v.clone() for k, v in w_glob.items()}
            if not (j == m - 1 and m % 2 == 1):
                for key in float_keys:
                    delta = w_glob[key] - global_state[key]
                    w[key] = w_glob[key] + delta * signs[key][j] * self.p.radius
            states.append(w)
        self.pending = states
        return w_glob
