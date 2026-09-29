from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field


class GroupCounts(BaseModel):
    """Confusion counts indexed [sensitive group a][true label y].

    `pred_pos` is only meaningful for binary tasks (count of predictions == 1).
    Every fairness metric is computed from these counts, so any status function
    or scope can be evaluated after the fact from logs.
    """

    n: list[list[int]]
    correct: list[list[int]]
    pred_pos: list[list[int]]

    def __add__(self, other: GroupCounts) -> GroupCounts:
        def add(x: list[list[int]], y: list[list[int]]) -> list[list[int]]:
            return [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(x, y)]

        return GroupCounts(
            n=add(self.n, other.n),
            correct=add(self.correct, other.correct),
            pred_pos=add(self.pred_pos, other.pred_pos),
        )


class FitIns(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    params: torch.Tensor
    config: dict[str, Any] = Field(default_factory=dict)


class FitRes(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    client_id: int
    params: torch.Tensor
    num_samples: int
    metrics: dict[str, Any] = Field(default_factory=dict)


class EvalRes(BaseModel):
    client_id: int
    num_samples: int
    loss: float
    accuracy: float
    groups: GroupCounts


class RoundLog(BaseModel):
    round: int
    selected: list[int]
    train_loss: float | None = None
    global_accuracy: float
    global_loss: float
    clients: list[EvalRes]
    global_test: EvalRes | None = None
    train_evals: list[EvalRes] | None = None  # final model on each client's train split (last training log only)
    strategy_info: dict[str, Any] = Field(default_factory=dict)
    seconds: float = 0.0
