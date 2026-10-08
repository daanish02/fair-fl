"""Client side of FairWeight (Kasyap et al., IEEE TSC 2026), following the authors' released code."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fairfl.clients.base import ClientAlgorithm
from fairfl.data.cifar import TensorDataset, random_crop_flip
from fairfl.strategies.base import FitIns, FitRes, flatten_state, unflatten


def dp_penalty(logits: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Squared gap of mean P(y=1) between sensitive groups (the paper's demographic-parity local loss)."""
    p = F.softmax(logits, dim=1)[:, 1] if logits.shape[1] == 2 else F.softmax(logits, dim=1).max(dim=1).values
    if (a == 0).sum() == 0 or (a == 1).sum() == 0:
        return logits.new_zeros(())
    return (p[a == 0].mean() - p[a == 1].mean()) ** 2


def weight_importance(model, theta: torch.Tensor, key_shapes, X, y, repeats: int,
                      gen: torch.Generator, step: int = 40) -> torch.Tensor:
    """Approximate Shapley value of every flattened parameter: |grad * theta| with 25% randomly zeroed."""
    wi = torch.zeros_like(theta)
    runs = max(1, len(y) // step)
    base_state = model.state_dict()
    for r in range(runs):
        xb, yb = X[r * step : (r + 1) * step], y[r * step : (r + 1) * step]
        for _ in range(repeats):
            keep = torch.ones_like(theta)
            keep[torch.randperm(len(theta), generator=gen)[: int(0.25 * len(theta))].to(theta.device)] = 0
            model.load_state_dict(unflatten(theta * keep, key_shapes, base_state))
            model.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            g = torch.cat([p.grad.flatten() for p in model.parameters()])
            wi += (g * theta).abs()
    model.load_state_dict(unflatten(theta, key_shapes, base_state))
    return wi / runs


class FairWeightClient(ClientAlgorithm):
    def batch_loss(self, model, X, y, a, ins: FitIns) -> torch.Tensor:
        logits = model(X)
        return F.cross_entropy(logits, y) + float(ins.config.get("fair_weight", 1.0)) * dp_penalty(logits, a)

    def fit(self, model, ds: TensorDataset, ins: FitIns, gen: torch.Generator, device: torch.device,
            client_id: int = 0, state: dict | None = None, augment: bool = True, log_every: int = 0,
            log_fn=None) -> FitRes:
        res = super().fit(model, ds, ins, gen, device, client_id, state, augment, log_every, log_fn)
        model.load_state_dict(res.state)
        # Gradients only exist for trainable parameters, never for buffers (e.g. BatchNorm running stats), so the
        # Shapley-style importance flatten must be restricted to parameter keys only, not all floating-point state.
        param_keys = {k for k, _ in model.named_parameters()}
        theta, key_shapes = flatten_state(res.state, keys=param_keys)
        X, y, a = ds.X.to(device), ds.y.to(device), ds.a.to(device)
        with torch.no_grad():
            pred = model(X).argmax(1)
        correct = pred == y
        # Official code: b = (true negatives, a=1) vs (false positives, a=0); c = (true negatives, a=0) vs (false negatives, a=0).
        sets = {
            "b1": (a == 1) & (y == 0) & correct, "b2": (a == 0) & (y == 0) & ~correct,
            "c1": (a == 0) & (y == 0) & correct, "c2": (a == 0) & (y == 1) & ~correct,
        }
        k = max(1, int(float(ins.config.get("top_fraction", 0.1)) * len(theta)))
        cap = int(ins.config.get("max_samples", 400))
        repeats = int(ins.config.get("repeats", 5))
        mask: set[int] = set()
        for x1, x2 in (("b1", "b2"), ("c1", "c2")):
            n = min(int(sets[x1].sum()), int(sets[x2].sum()), cap)
            if n < 1:
                continue
            i1, i2 = sets[x1].nonzero().flatten()[:n], sets[x2].nonzero().flatten()[:n]
            diff = (weight_importance(model, theta, key_shapes, X[i1], y[i1], repeats, gen)
                    - weight_importance(model, theta, key_shapes, X[i2], y[i2], repeats, gen))
            mask |= set(torch.topk(diff, k).indices.tolist())
        model.load_state_dict(res.state)
        res.metrics["mask"] = sorted(mask)
        return res
