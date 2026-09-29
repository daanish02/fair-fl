from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from fairfl.config import RunConfig
from fairfl.data.cifar import TensorDataset, load_cifar10, random_crop_flip
from fairfl.data.partition import dirichlet_partition
from fairfl.models.resnet import build_resnet18

log = logging.getLogger("fairfl")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class RoundResult:
    round: int
    train_loss: float
    test_acc: float
    elapsed_s: float = 0.0
    eta_s: float = 0.0


@dataclass
class RunResult:
    history: list[RoundResult] = field(default_factory=list)
    client_acc: dict[int, float] = field(default_factory=dict)
    """Per-client test accuracy at the end of training. Key 0 alone means centralised."""


def _eval_acc(model: nn.Module, ds: TensorDataset, device: torch.device, batch_size: int = 512) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            xb = ds.X[i : i + batch_size].to(device)
            yb = ds.y[i : i + batch_size].to(device)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += len(yb)
    return correct / total


def _train_epochs(
    model: nn.Module, ds: TensorDataset, device: torch.device, epochs: int, opt: torch.optim.Optimizer,
    batch_size: int, gen: torch.Generator, augment: bool = True, log_every: int = 0,
) -> float:
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    n = len(ds)
    n_batches = (n + batch_size - 1) // batch_size
    total_loss, steps = 0.0, 0
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for bi, i in enumerate(range(0, n, batch_size)):
            idx = perm[i : i + batch_size]
            xb = ds.X[idx].to(device)
            yb = ds.y[idx].to(device)
            if augment:
                xb = random_crop_flip(xb, gen)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            steps += 1
            if log_every and (bi + 1) % log_every == 0:
                log.info("  epoch %d batch %d/%d loss=%.4f", ep + 1, bi + 1, n_batches, loss.item())
    return total_loss / max(steps, 1)


def run_centralised(cfg: RunConfig, stop_event: threading.Event | None = None) -> Iterator[RoundResult | RunResult]:
    """No clients: one model trained on the full CIFAR10 train set, `rounds` used as epoch count."""
    device = resolve_device(cfg.device)
    log.info("centralised: device=%s loading CIFAR10 from %s", device, cfg.data_root)
    t_data = time.perf_counter()
    gen = torch.Generator().manual_seed(cfg.seed)
    train_ds = load_cifar10(cfg.data_root, "train").to(device)
    test_ds = load_cifar10(cfg.data_root, "test").to(device)
    log.info("data loaded in %.1fs (train=%d test=%d)", time.perf_counter() - t_data, len(train_ds), len(test_ds))

    model = build_resnet18().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.rounds)
    history = []
    t_start = time.perf_counter()
    for r in range(1, cfg.rounds + 1):
        t0 = time.perf_counter()
        loss = _train_epochs(model, train_ds, device, 1, opt, cfg.batch_size, gen, log_every=cfg.log_every)
        scheduler.step()
        acc = _eval_acc(model, test_ds, device)
        elapsed = time.perf_counter() - t0
        avg = (time.perf_counter() - t_start) / r
        eta = avg * (cfg.rounds - r)
        rr = RoundResult(round=r, train_loss=loss, test_acc=acc, elapsed_s=elapsed, eta_s=eta)
        history.append(rr)
        log.info("epoch %d/%d loss=%.4f acc=%.4f lr=%.5f (%.1fs, ETA %.0fs)",
                  r, cfg.rounds, loss, acc, opt.param_groups[0]["lr"], elapsed, eta)
        yield rr
        if stop_event is not None and stop_event.is_set():
            log.info("stop requested, halting after epoch %d/%d", r, cfg.rounds)
            break

    yield RunResult(history=history, client_acc={0: history[-1].test_acc if history else _eval_acc(model, test_ds, device)})


def run_federated(cfg: RunConfig, stop_event: threading.Event | None = None) -> Iterator[RoundResult | RunResult]:
    """FedAvg: cfg.num_clients clients, Dirichlet(alpha) label-skew partition of the CIFAR10 train set."""
    device = resolve_device(cfg.device)
    log.info("federated: device=%s clients=%d alpha=%s loading CIFAR10 from %s",
              device, cfg.num_clients, cfg.alpha, cfg.data_root)
    t_data = time.perf_counter()
    gen = torch.Generator().manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    train_ds = load_cifar10(cfg.data_root, "train")
    test_ds = load_cifar10(cfg.data_root, "test").to(device)

    parts = dirichlet_partition(train_ds.y.numpy(), cfg.num_clients, cfg.alpha, rng)
    client_train = [train_ds.subset(idx).to(device) for idx in parts]

    client_test_idx = dirichlet_partition(test_ds.y.cpu().numpy(), cfg.num_clients, cfg.alpha, rng, min_samples=5)
    client_test = [test_ds.subset(idx) for idx in client_test_idx]
    log.info("data loaded + partitioned in %.1fs; client shard sizes=%s",
              time.perf_counter() - t_data, [len(c) for c in client_train])

    global_model = build_resnet18().to(device)
    local_model = build_resnet18().to(device)  # reused scratch model, avoids per-client deepcopy
    history = []
    t_start = time.perf_counter()

    # cosine-decay the lr by communication round (all clients in a round share the same lr)
    dummy_opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dummy_opt, T_max=cfg.rounds)

    for r in range(1, cfg.rounds + 1):
        t0 = time.perf_counter()
        round_lr = dummy_opt.param_groups[0]["lr"]
        global_state = global_model.state_dict()
        stopped_mid_round = False
        state_dicts, weights, round_loss = [], [], 0.0
        for c in range(cfg.num_clients):
            if stop_event is not None and stop_event.is_set():
                log.info("stop requested, halting mid-round %d (client %d/%d)", r, c, cfg.num_clients)
                stopped_mid_round = True
                break
            local_model.load_state_dict(global_state)
            local_opt = torch.optim.SGD(local_model.parameters(), lr=round_lr, momentum=0.9,
                                         weight_decay=cfg.weight_decay)
            log_every = cfg.log_every if c == 0 else 0  # only client 0 logs per-batch, avoids spam
            loss = _train_epochs(local_model, client_train[c], device, cfg.local_epochs, local_opt, cfg.batch_size,
                                  gen, log_every=log_every)
            state_dicts.append({k: v.detach().clone() for k, v in local_model.state_dict().items()})
            weights.append(len(client_train[c]))
            round_loss += loss * weights[-1]
            log.debug("  round %d client %d/%d loss=%.4f (%.1fs elapsed)",
                       r, c + 1, cfg.num_clients, loss, time.perf_counter() - t0)
        if stopped_mid_round:
            break
        round_loss /= sum(weights)

        avg_state = {}
        total_w = sum(weights)
        for key in state_dicts[0]:
            if state_dicts[0][key].is_floating_point():
                avg_state[key] = sum(sd[key] * w for sd, w in zip(state_dicts, weights)) / total_w
            else:
                avg_state[key] = state_dicts[0][key]
        global_model.load_state_dict(avg_state)
        scheduler.step()

        acc = _eval_acc(global_model, test_ds, device)
        elapsed = time.perf_counter() - t0
        avg = (time.perf_counter() - t_start) / r
        eta = avg * (cfg.rounds - r)
        rr = RoundResult(round=r, train_loss=round_loss, test_acc=acc, elapsed_s=elapsed, eta_s=eta)
        history.append(rr)
        log.info("round %d/%d loss=%.4f acc=%.4f lr=%.5f (%.1fs, ETA %.0fs)",
                  r, cfg.rounds, round_loss, acc, round_lr, elapsed, eta)
        yield rr
        if stop_event is not None and stop_event.is_set():
            log.info("stop requested, halting after round %d/%d", r, cfg.rounds)
            break

    log.info("evaluating per-client accuracy...")
    client_acc = {c: _eval_acc(global_model, client_test[c].to(device), device) for c in range(cfg.num_clients)}
    yield RunResult(history=history, client_acc=client_acc)


def run(cfg: RunConfig, stop_event: threading.Event | None = None) -> Iterator[RoundResult | RunResult]:
    fn = run_centralised if cfg.is_centralised else run_federated
    yield from fn(cfg, stop_event)
