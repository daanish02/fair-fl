"""Client-side local training. Subclasses override `batch_loss`/`lr`/`fit` to change the local objective."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairfl.config import TrainConfig
from fairfl.data.cifar import TensorDataset, random_crop_flip
from fairfl.strategies.base import FitIns, FitRes


class ClientAlgorithm:
    def __init__(self, train_cfg: TrainConfig):
        self.cfg = train_cfg

    def batch_loss(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor, a: torch.Tensor,
                   ins: FitIns) -> torch.Tensor:
        return F.cross_entropy(model(X), y)

    def lr(self, ins: FitIns) -> float:
        return float(ins.config.get("lr", self.cfg.lr))

    @torch.no_grad()
    def _mean_loss(self, model: nn.Module, ds: TensorDataset, device: torch.device, batch_size: int = 2048) -> float:
        model.eval()
        total, n = 0.0, len(ds)
        for i in range(0, n, batch_size):
            xb, yb = ds.X[i : i + batch_size].to(device), ds.y[i : i + batch_size].to(device)
            total += F.cross_entropy(model(xb), yb, reduction="sum").item()
        return total / max(n, 1)

    @torch.no_grad()
    def _mean_acc(self, model: nn.Module, ds: TensorDataset, device: torch.device, batch_size: int = 2048) -> float:
        model.eval()
        correct, n = 0, len(ds)
        for i in range(0, n, batch_size):
            xb, yb = ds.X[i : i + batch_size].to(device), ds.y[i : i + batch_size].to(device)
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
        return correct / max(n, 1)

    def fit(self, model: nn.Module, ds: TensorDataset, ins: FitIns, gen: torch.Generator, device: torch.device,
            client_id: int = 0, state: dict | None = None, augment: bool = True, log_every: int = 0,
            log_fn=None) -> FitRes:
        model.load_state_dict(ins.state)
        loss_before = self._mean_loss(model, ds, device) if ins.config.get("need_loss_before") else None
        opt = torch.optim.SGD(model.parameters(), lr=self.lr(ins), momentum=self.cfg.momentum,
                              weight_decay=self.cfg.weight_decay)
        model.train()
        n = len(ds)
        n_batches = (n + self.cfg.batch_size - 1) // self.cfg.batch_size
        epochs = int(ins.config.get("local_epochs", self.cfg.local_epochs))
        total_loss, steps = 0.0, 0
        for ep in range(epochs):
            perm = torch.randperm(n, generator=gen)
            for bi, i in enumerate(range(0, n, self.cfg.batch_size)):
                idx = perm[i : i + self.cfg.batch_size]
                xb = ds.X[idx].to(device)
                yb = ds.y[idx].to(device)
                ab = ds.a[idx].to(device)
                if augment:
                    xb = random_crop_flip(xb, gen)
                opt.zero_grad()
                loss = self.batch_loss(model, xb, yb, ab, ins)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                steps += 1
                if log_every and (bi + 1) % log_every == 0 and log_fn is not None:
                    log_fn(ep + 1, bi + 1, n_batches, loss.item())
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        metrics = {"train_loss": total_loss / max(steps, 1)}
        if loss_before is not None:
            metrics["loss_before"] = loss_before
        if ins.config.get("report_train_acc"):
            metrics["train_acc"] = self._mean_acc(model, ds, device)
        return FitRes(client_id=client_id, state=state, num_samples=n, metrics=metrics)


class SGDClient(ClientAlgorithm):
    pass


from fairfl.strategies.base import Strategy  # noqa: E402

Strategy.client_cls = SGDClient
