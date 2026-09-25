import math

import numpy as np
import torch
from pydantic import BaseModel, Field

from fairfl.core.registry import register_strategy
from fairfl.strategies.base import Strategy
from fairfl.strategies.qffl import qffl_step


class FairRFLParams(BaseModel):
    tau: float = Field(2.5, ge=0.0, description="MAD threshold for flagging selfish updates (paper: 2.5).")
    q: float = Field(1.0, ge=0.0, description="Base fairness exponent for Dq-FFL; 0 disables Dq-FFL weighting.")
    use_dqffl: bool = Field(True, description="FairRFL = RFL-Self recovery + Dq-FFL. False gives RFL-Self only.")


def detect_selfish(deltas: torch.Tensor, tau: float) -> tuple[list[int], float]:
    """Algorithm 1: flag updates whose norm exceeds the median norm by more than tau MADs."""
    norms = deltas.norm(dim=1).cpu().numpy()
    n_med = float(np.median(norms))
    mad = 1.4826 * float(np.median(np.abs(n_med - norms)))
    flagged = [i for i, n in enumerate(norms) if mad > 0 and (n - n_med) / mad > tau]
    return flagged, n_med


def recover(selfish: torch.Tensor, med: torch.Tensor, n_med: float) -> tuple[torch.Tensor, float]:
    """Eq 8-9: beta * s_hat + (1 - beta) * med with norm N_med; largest root in [0, 1]."""
    d = selfish - med
    a = float(d @ d)
    b = 2.0 * float(med @ d)
    c = float(med @ med) - n_med**2
    disc = b * b - 4 * a * c
    if a == 0 or disc < 0:
        beta = 0.0
    else:
        roots = [(-b + math.sqrt(disc)) / (2 * a), (-b - math.sqrt(disc)) / (2 * a)]
        valid = [r for r in roots if 0.0 <= r <= 1.0]
        beta = max(valid) if valid else 0.0
    return beta * selfish + (1 - beta) * med, beta


def rfl_self(deltas: torch.Tensor, tau: float) -> tuple[torch.Tensor, list[int]]:
    """Algorithm 2 lines 5-10: returns the (partly recovered) update matrix and the flagged rows."""
    flagged, n_med = detect_selfish(deltas, tau)
    med = torch.from_numpy(np.median(deltas.cpu().numpy(), axis=0).astype(np.float32)).to(deltas.device)
    out = deltas.clone()
    for i in flagged:
        out[i], _ = recover(deltas[i], med, n_med)
    return out, flagged


@register_strategy("fairrfl")
class FairRFL(Strategy):
    """Augello, Gupta, Lo Re, Das, IEEE TETC 2026: RFL-Self (Alg 1-2) + Dq-FFL (Alg 3).

    Dq-FFL scales each client's q by l_med / l_i, using the previous round's reported losses.
    """

    Params = FairRFLParams

    def initialize(self, gparams, num_clients, client_sizes, layer_sizes):
        super().initialize(gparams, num_clients, client_sizes, layer_sizes)
        self.last_loss: dict[int, float] = {}
        self.l_med: float | None = None

    def aggregate(self, rnd, gparams, results):
        deltas = torch.stack([r.params - gparams for r in results])
        recovered, flagged = rfl_self(deltas, self.p.tau)
        for r, d in zip(results, recovered):
            r.params = gparams + d
        self.info = {"flagged": [results[i].client_id for i in flagged]}
        if not self.p.use_dqffl:
            return gparams + recovered.mean(dim=0)
        qs = []
        for r in results:
            li = self.last_loss.get(r.client_id)
            qs.append(self.p.q if self.l_med is None or not li else self.p.q * self.l_med / li)
        new = qffl_step(gparams, results, qs, self.train_cfg.lr)
        for r in results:
            self.last_loss[r.client_id] = float(r.metrics["loss_before"])
        self.l_med = float(np.median([r.metrics["loss_before"] for r in results]))
        return new
