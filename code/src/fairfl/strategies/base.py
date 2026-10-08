"""Server-side strategy interface. Every hook defaults to today's FedAvg behavior."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from pydantic import BaseModel

from fairfl.config import TrainConfig

if TYPE_CHECKING:
    from fairfl.clients.base import ClientAlgorithm


class NoParams(BaseModel):
    pass


@dataclass
class FitIns:
    state: dict
    config: dict = field(default_factory=dict)


@dataclass
class FitRes:
    client_id: int
    state: dict
    num_samples: int
    metrics: dict = field(default_factory=dict)


@dataclass
class EvalRes:
    client_id: int
    num_samples: int
    loss: float
    accuracy: float


def flatten_state(state: dict) -> tuple[torch.Tensor, list[tuple[str, torch.Size]]]:
    """Flatten a state_dict's floating-point tensors into one vector, plus (key, shape) to unflatten."""
    key_shapes = [(k, v.shape) for k, v in state.items() if v.is_floating_point()]
    vector = torch.cat([state[k].flatten() for k, _ in key_shapes])
    return vector, key_shapes


def unflatten(vector: torch.Tensor, key_shapes: list[tuple[str, torch.Size]], base_state: dict) -> dict:
    """Inverse of flatten_state: non-floating-point keys come from `base_state` unchanged."""
    out = dict(base_state)
    i = 0
    for k, shape in key_shapes:
        n = shape.numel()
        out[k] = vector[i : i + n].view(shape)
        i += n
    return out


def weighted_mean(states: list[dict], weights: list[float]) -> dict:
    """Weighted mean of client state_dicts, weighted by `weights` (e.g. client sample counts)."""
    total_w = sum(weights)
    avg_state = {}
    for key in states[0]:
        if states[0][key].is_floating_point():
            avg_state[key] = sum(sd[key] * w for sd, w in zip(states, weights)) / total_w
        else:
            avg_state[key] = states[0][key]
    return avg_state


class Strategy(ABC):
    name: ClassVar[str] = "base"
    Params: ClassVar[type[BaseModel]] = NoParams
    client_cls: ClassVar[type["ClientAlgorithm"]] = None  # set to SGDClient in clients/base.py
    needs_loss_before: ClassVar[bool] = False  # clients report full-train-split loss before local training

    def __init__(self, params: BaseModel, train_cfg: TrainConfig, rng: np.random.Generator):
        self.p = params
        self.train_cfg = train_cfg
        self.rng = rng
        self.info: dict[str, Any] = {}

    def initialize(self, global_state: dict, num_clients: int, client_sizes: list[int]) -> None:
        self.num_clients = num_clients
        self.client_sizes = client_sizes

    def sample_clients(self, ids: list[int]) -> list[int]:
        m = max(1, int(round(self.train_cfg.clients_per_round * len(ids))))
        return sorted(self.rng.choice(ids, size=m, replace=False).tolist())

    def configure_eval(self, rnd: int, ids: list[int]) -> list[int]:
        """Clients to evaluate the current global model on before configure_round (e.g. FairFed, FCFL)."""
        return []

    def configure_round(self, rnd: int, global_state: dict, ids: list[int],
                        pre_eval: dict[int, EvalRes], round_lr: float) -> dict[int, FitIns]:
        return {cid: FitIns(state=global_state, config={"lr": round_lr}) for cid in self.sample_clients(ids)}

    def aggregate(self, rnd: int, global_state: dict, results: list[FitRes]) -> dict:
        return weighted_mean([r.state for r in results], [r.num_samples for r in results])

    def make_client(self) -> "ClientAlgorithm":
        return self.client_cls(self.train_cfg)

    def decision_rule(self):
        """(cid, logits, a) -> predicted labels; override for a custom eval-time rule (e.g. LoGoFair)."""
        return lambda cid, logits, a: logits.argmax(dim=1)

    def on_round_end(self, round_result) -> None:
        """Called with the finished round's result; lets stateful strategies see outcomes."""

    def postprocess(self, global_state: dict, model: torch.nn.Module, client_algo: "ClientAlgorithm",
                    client_data: list) -> bool:
        """Optional one-time phase after all rounds (e.g. LoGoFair). Return True if the decision rule changed."""
        return False

    @classmethod
    def build(cls, raw_params: dict[str, Any], train_cfg: TrainConfig, rng: np.random.Generator) -> "Strategy":
        return cls(cls.Params.model_validate(raw_params), train_cfg, rng)
