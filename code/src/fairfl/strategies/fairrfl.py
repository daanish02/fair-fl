from __future__ import annotations

import math

import numpy as np
import torch
from pydantic import BaseModel, Field

from fairfl.registry import register_strategy
from fairfl.strategies._qffl_step import qffl_step
from fairfl.strategies.base import Strategy


class FairRFLParams(BaseModel):
    tau: float = Field(2.5, ge=0.0, description="MAD threshold for flagging selfish updates (paper: 2.5).")
    q: float = Field(1.0, ge=0.0, description="Base fairness exponent for Dq-FFL; 0 disables Dq-FFL weighting.")
    use_dqffl: bool = Field(True, description="FairRFL = RFL-Self recovery + Dq-FFL. False gives RFL-Self only.")


def _detect_selfish(deltas: torch.Tensor, tau: float) -> tuple[list[int], float]:
    """Algorithm 1: flag updates whose norm exceeds the median norm by more than tau MADs."""
    norms = deltas.flatten(1).norm(dim=1).cpu().numpy()
    n_med = float(np.median(norms))
    mad = 1.4826 * float(np.median(np.abs(n_med - norms)))
    return [i for i, n in enumerate(norms) if mad > 0 and (n - n_med) / mad > tau], n_med


def _recover(selfish: torch.Tensor, med: torch.Tensor, n_med: float) -> torch.Tensor:
    """Eq 8-9: beta * s_hat + (1 - beta) * med with norm N_med; largest root in [0, 1]."""
    d = selfish - med
    a = float((d * d).sum())
    b = 2.0 * float((med * d).sum())
    c = float((med * med).sum()) - n_med**2
    disc = b * b - 4 * a * c
    if a == 0 or disc < 0:
        beta = 0.0
    else:
        roots = [(-b + math.sqrt(disc)) / (2 * a), (-b - math.sqrt(disc)) / (2 * a)]
        valid = [r for r in roots if 0.0 <= r <= 1.0]
        beta = max(valid) if valid else 0.0
    return beta * selfish + (1 - beta) * med


def _rfl_self(deltas: torch.Tensor, tau: float) -> tuple[torch.Tensor, list[int]]:
    """Algorithm 2 lines 5-10: returns the (partly recovered) update stack and the flagged rows."""
    flagged, n_med = _detect_selfish(deltas, tau)
    med = deltas.median(dim=0).values
    out = deltas.clone()
    for i in flagged:
        out[i] = _recover(deltas[i], med, n_med)
    return out, flagged


@register_strategy("fairrfl")
class FairRFL(Strategy):
    """Augello, Gupta, Lo Re, Das, IEEE TETC 2026: RFL-Self (Alg 1-2) + Dq-FFL (Alg 3).

    Dq-FFL scales each client's q by l_med / l_i, using the previous round's reported losses.
    """

    Params = FairRFLParams
    needs_loss_before = True

    def initialize(self, global_state, num_clients, client_sizes):
        super().initialize(global_state, num_clients, client_sizes)
        self.last_loss: dict[int, float] = {}
        self.l_med: float | None = None

    def aggregate(self, rnd, global_state, results):
        recovered_per_key = {}
        flagged_ids = set()
        for key, g in global_state.items():
            if not g.is_floating_point():
                continue
            deltas = torch.stack([r.state[key] - g for r in results])
            recovered, flagged = _rfl_self(deltas, self.p.tau)
            recovered_per_key[key] = recovered
            flagged_ids.update(results[i].client_id for i in flagged)
        self.info = {"flagged": sorted(flagged_ids)}

        if not self.p.use_dqffl:
            out = {}
            for key, g in global_state.items():
                out[key] = g + recovered_per_key[key].mean(dim=0) if key in recovered_per_key else results[0].state[key]
            return out

        recovered_results = []
        for i, r in enumerate(results):
            state = {key: (global_state[key] + recovered_per_key[key][i]) if key in recovered_per_key
                     else r.state[key] for key in global_state}
            recovered_results.append(type(r)(client_id=r.client_id, state=state, num_samples=r.num_samples,
                                              metrics=r.metrics))

        qs = []
        for r in results:
            li = self.last_loss.get(r.client_id)
            qs.append(self.p.q if self.l_med is None or not li else self.p.q * self.l_med / li)
        new = qffl_step(global_state, recovered_results, qs, self.train_cfg.lr)
        for r in results:
            self.last_loss[r.client_id] = float(r.metrics["loss_before"])
        self.l_med = float(np.median([r.metrics["loss_before"] for r in results]))
        return new
