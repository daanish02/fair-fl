"""Server-side strategy interface. Adapted from Flower's Strategy split (configure/aggregate) without depending on it."""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar

import numpy as np
import torch
from pydantic import BaseModel

from fairfl.clients.base import ClientAlgorithm, DecisionRule, SGDClient, argmax_rule
from fairfl.core.config import TrainConfig
from fairfl.core.params import Params, weighted_mean
from fairfl.core.types import EvalRes, FitIns, FitRes, RoundLog


class NoParams(BaseModel):
    pass


class Strategy(ABC):
    name: ClassVar[str] = "base"
    Params: ClassVar[type[BaseModel]] = NoParams
    client_cls: ClassVar[type[ClientAlgorithm]] = SGDClient
    pre_eval_split: ClassVar[str] = "train"
    needs_loss_before: ClassVar[bool] = False  # clients report the pre-training loss on their whole train split

    def __init__(self, params: BaseModel, train_cfg: TrainConfig, rng: np.random.Generator):
        self.p = params
        self.train_cfg = train_cfg
        self.rng = rng
        self.info: dict[str, Any] = {}

    def initialize(self, gparams: Params, num_clients: int, client_sizes: list[int], layer_sizes: list[int]) -> None:
        self.num_clients = num_clients
        self.client_sizes = client_sizes
        self.layer_sizes = layer_sizes

    def sample_clients(self, rnd: int, ids: list[int]) -> list[int]:
        m = max(1, int(round(self.train_cfg.clients_per_round * len(ids))))
        return sorted(self.rng.choice(ids, size=m, replace=False).tolist())

    def configure_eval(self, rnd: int, ids: list[int]) -> list[int]:
        """Clients that evaluate the current global model (on `pre_eval_split`) before selection (e.g. FCFL)."""
        return []

    def configure_round(self, rnd: int, gparams: Params, ids: list[int], pre_eval: dict[int, EvalRes]) -> dict[int, FitIns]:
        return {cid: FitIns(params=gparams) for cid in self.sample_clients(rnd, ids)}

    def aggregate(self, rnd: int, gparams: Params, results: list[FitRes]) -> Params:
        return weighted_mean([r.params for r in results], [r.num_samples for r in results])

    def make_client(self) -> ClientAlgorithm:
        return self.client_cls(self.train_cfg)

    def decision_rule(self) -> DecisionRule:
        return argmax_rule

    def on_round_end(self, log: RoundLog) -> None:
        """Called with the evaluation of the deployed model; lets stateful strategies see outcomes."""

    def postprocess(self, gparams: Params, model: torch.nn.Module, clients: list) -> bool:
        """Optional post-training phase (LoGoFair). Return True if the deployed decision rule changed."""
        return False

    @classmethod
    def build(cls, raw: dict[str, Any], train_cfg: TrainConfig, rng: np.random.Generator) -> Strategy:
        return cls(cls.Params.model_validate(raw), train_cfg, rng)
