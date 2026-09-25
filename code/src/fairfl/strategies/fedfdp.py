from typing import Literal

from pydantic import BaseModel, Field

from fairfl.clients.fedfdp import FedFairClient, FedFDPClient
from fairfl.core.params import weighted_mean
from fairfl.core.registry import register_strategy
from fairfl.core.types import FitIns
from fairfl.strategies.base import Strategy


class FedFairParams(BaseModel):
    lam: float = Field(0.1, ge=0.0, description="Fairness strength lambda (0 = FedAvg).")


class FedFDPParams(FedFairParams):
    lam: float = Field(0.01, ge=0.0, description="Fairness strength; paper Fig. 2 best on FMNIST (MNIST unstated).")
    clip: float = Field(0.1, gt=0.0, description="Gradient clipping bound C (paper default 0.1).")
    sigma: float = Field(2.0, ge=0.0, description="Gradient noise multiplier (paper default 2).")
    sigma_loss: float = Field(5.0, ge=0.0, description="Loss noise multiplier sigma_l (paper default 5).")
    loss_clip_init: float = Field(2.5, gt=0.0, description="Initial loss clipping bound C_l^0 (paper default 2.5).")
    sampling: Literal["poisson", "batches"] = Field("poisson", description="poisson = paper (one batch per round).")
    q: float = Field(0.05, gt=0.0, le=1.0, description="Poisson sampling rate (paper default 0.05).")
    dp_steps: int = Field(1, ge=1, description="DP-SGD steps per round on the sampled batch (paper: 1).")


@register_strategy("fedfair")
class FedFair(Strategy):
    """Ling et al., ACNS 2026 -- FedFair (Eq 5-7): local objective F_i + lambda/2 (F_i - F)^2.
    Server keeps F(w) = sum_i p_i F_i(w_{t+1}^i) and broadcasts it (Algorithm 1 lines 6-7)."""

    Params = FedFairParams
    client_cls = FedFairClient

    def initialize(self, gparams, num_clients, client_sizes, layer_sizes):
        super().initialize(gparams, num_clients, client_sizes, layer_sizes)
        self.global_loss: float | None = None

    def configure_round(self, rnd, gparams, ids, pre_eval):
        cfg = {**self.p.model_dump(), "global_loss": self.global_loss}
        return {cid: FitIns(params=gparams, config=cfg) for cid in self.sample_clients(rnd, ids)}

    def aggregate(self, rnd, gparams, results):
        w = [r.num_samples for r in results]
        self.global_loss = sum(wi * r.metrics["loss_after"] for wi, r in zip(w, results)) / sum(w)
        self.info = {"global_loss": self.global_loss}
        return weighted_mean([r.params for r in results], w)


@register_strategy("fedfdp")
class FedFDP(FedFair):
    """FedFair + fairness-aware DP-SGD clipping (Eq 10-11) + adaptive clipping of the uploaded loss (Eq 12-13)."""

    Params = FedFDPParams
    client_cls = FedFDPClient
