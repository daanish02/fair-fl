import torch
from pydantic import BaseModel, Field

from fairfl.core.registry import register_strategy
from fairfl.strategies.base import Strategy


class QFFLParams(BaseModel):
    q: float = Field(1.0, ge=0.0, description="Fairness exponent (0 recovers FedAvg-like averaging).")


def qffl_step(gparams: torch.Tensor, results, qs: list[float], lr: float) -> torch.Tensor:
    """q-FedAvg server update (Li et al., ICLR 2020), also used by FairRFL's Dq-FFL with per-client q."""
    L = 1.0 / lr
    num = torch.zeros_like(gparams)
    den = 0.0
    for r, q in zip(results, qs):
        F = max(float(r.metrics["loss_before"]), 1e-10)
        dw = L * (gparams - r.params)
        num += (F ** q) * dw
        den += q * F ** (q - 1) * float(dw.pow(2).sum()) + L * F ** q
    return gparams - num / den


@register_strategy("qffl")
class QFFL(Strategy):
    """Li, Sanjabi, Beirami, Smith, ICLR 2020 (q-FedAvg)."""

    Params = QFFLParams

    def aggregate(self, rnd, gparams, results):
        return qffl_step(gparams, results, [self.p.q] * len(results), self.train_cfg.lr)
