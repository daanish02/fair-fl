from __future__ import annotations

from abc import ABC
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from fairfl.core.config import TrainConfig
from fairfl.core.params import get_params, set_params
from fairfl.core.types import EvalRes, FitIns, FitRes, GroupCounts
from fairfl.data.datasets import TensorDataset3, random_crop_flip
from fairfl.data.federated import ClientData

DecisionRule = Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]
"""(client_id, probs[N, C], a[N]) -> predicted labels[N]."""


def argmax_rule(_cid: int, probs: torch.Tensor, _a: torch.Tensor) -> torch.Tensor:
    return probs.argmax(dim=1)


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def batches(n: int, batch_size: int, gen: torch.Generator):
    """Index batches from a CPU generator (indices stay on CPU; indexing a CUDA tensor with them is fine)."""
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, batch_size):
        yield perm[i : i + batch_size]


@torch.no_grad()
def mean_loss(model: nn.Module, data, batch_size: int = 2048) -> float:
    model.eval()
    dev = model_device(model)
    total, n = 0.0, len(data)
    for i in range(0, n, batch_size):
        X, y = data.X[i : i + batch_size].to(dev), data.y[i : i + batch_size].to(dev)
        total += F.cross_entropy(model(X), y, reduction="sum").item()
    return total / max(n, 1)


@torch.no_grad()
def predict_proba(model: nn.Module, X: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    """Class probabilities, returned on the CPU whatever the model's device."""
    model.eval()
    dev = model_device(model)
    return torch.cat([F.softmax(model(X[i : i + batch_size].to(dev)), dim=1).cpu()
                      for i in range(0, len(X), batch_size)])


@torch.no_grad()
def evaluate(model: nn.Module, test: TensorDataset3, client_id: int, rule: DecisionRule = argmax_rule) -> EvalRes:
    if len(test) == 0:
        z = [[0] * test.num_classes for _ in range(test.num_groups)]
        return EvalRes(client_id=client_id, num_samples=0, loss=float("nan"), accuracy=float("nan"),
                       groups=GroupCounts(n=z, correct=z, pred_pos=z))
    probs = predict_proba(model, test.X)
    y, a = test.y.cpu(), test.a.cpu()
    loss = F.nll_loss(torch.log(probs.clamp_min(1e-12)), y).item()
    pred = rule(client_id, probs, a)
    ga, gy = test.num_groups, test.num_classes
    n = [[0] * gy for _ in range(ga)]
    correct = [[0] * gy for _ in range(ga)]
    pos = [[0] * gy for _ in range(ga)]
    for g in range(ga):
        for c in range(gy):
            m = (a == g) & (y == c)
            n[g][c] = int(m.sum())
            correct[g][c] = int((pred[m] == c).sum())
            pos[g][c] = int((pred[m] == 1).sum()) if gy == 2 else 0
    acc = float((pred == y).float().mean()) if len(test) else 0.0
    return EvalRes(client_id=client_id, num_samples=len(test), loss=loss, accuracy=acc,
                   groups=GroupCounts(n=n, correct=correct, pred_pos=pos))


class ClientAlgorithm(ABC):
    """Local training. Subclasses override `batch_loss` or `fit` to change the local objective."""

    def __init__(self, train_cfg: TrainConfig):
        self.cfg = train_cfg

    def batch_loss(self, model: nn.Module, X, y, a, ins: FitIns, state: dict[str, Any]) -> torch.Tensor:
        return F.cross_entropy(model(X), y)

    def lr(self, ins: FitIns, state: dict[str, Any]) -> float:
        return float(ins.config.get("lr", self.cfg.lr))

    def fit(self, model: nn.Module, data: ClientData, ins: FitIns, state: dict[str, Any], gen: torch.Generator) -> FitRes:
        set_params(model, ins.params)
        train = data.train
        loss_before = mean_loss(model, train)
        opt = torch.optim.SGD(model.parameters(), lr=self.lr(ins, state), momentum=self.cfg.momentum,
                              weight_decay=self.cfg.weight_decay)
        dev = model_device(model)
        model.train()
        # Last-epoch loss is summed on the device and read once: a per-batch .item() forces a GPU sync every step.
        last_sum, last_n = torch.zeros((), dtype=torch.float64, device=dev), 0
        for epoch in range(int(ins.config.get("local_epochs", self.cfg.local_epochs))):
            last_sum.zero_()
            last_n = 0
            for idx in batches(len(train), self.cfg.batch_size, gen):
                X, y, a = train.X[idx].to(dev), train.y[idx].to(dev), train.a[idx].to(dev)
                if train.augment:
                    X = random_crop_flip(X, gen)
                opt.zero_grad()
                loss = self.batch_loss(model, X, y, a, ins, state)
                loss.backward()
                opt.step()
                last_sum += loss.detach().double()
                last_n += 1
        metrics = {"loss_before": loss_before, "train_loss": float(last_sum) / max(last_n, 1)}
        if ins.config.get("report_train_acc"):
            metrics["train_acc"] = evaluate(model, train, data.client_id).accuracy
        return FitRes(client_id=data.client_id, params=get_params(model), num_samples=len(train), metrics=metrics)


class SGDClient(ClientAlgorithm):
    pass
