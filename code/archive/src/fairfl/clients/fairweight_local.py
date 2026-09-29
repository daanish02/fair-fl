"""Client side of FairWeight (Kasyap et al., IEEE TSC 2026), following the authors' released code (fairweight.zip)."""

import torch
import torch.nn.functional as F

from fairfl.clients.base import ClientAlgorithm, model_device
from fairfl.core.params import get_params, set_params


def dp_penalty(logits: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Squared gap of mean P(y=1) between sensitive groups (the paper's demographic-parity local loss)."""
    p = F.softmax(logits, dim=1)[:, 1]
    if (a == 0).sum() == 0 or (a == 1).sum() == 0:
        return logits.new_zeros(())
    return (p[a == 0].mean() - p[a == 1].mean()) ** 2


def weight_importance(model, theta: torch.Tensor, X, y, repeats: int, gen: torch.Generator, step: int = 40) -> torch.Tensor:
    """Approximate Shapley value of every parameter: |grad * theta| with 25% of parameters randomly zeroed
    (calculate_shapley_values_fa in the official code; first-order Taylor, paper Eq 12)."""
    wi = torch.zeros_like(theta)
    runs = max(1, len(y) // step)
    for r in range(runs):
        xb, yb = X[r * step : (r + 1) * step], y[r * step : (r + 1) * step]
        for _ in range(repeats):
            keep = torch.ones_like(theta)
            keep[torch.randperm(len(theta), generator=gen)[: int(0.25 * len(theta))].to(theta.device)] = 0
            set_params(model, theta * keep)
            model.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            g = torch.cat([p.grad.flatten() for p in model.parameters()])
            wi += (g * theta).abs()
    set_params(model, theta)
    return wi / runs


class FairWeightClient(ClientAlgorithm):
    def batch_loss(self, model, X, y, a, ins, state):
        logits = model(X)
        return F.cross_entropy(logits, y) + float(ins.config.get("fair_weight", 1.0)) * dp_penalty(logits, a)

    def fit(self, model, data, ins, state, gen):
        res = super().fit(model, data, ins, state, gen)
        theta = res.params
        dev = model_device(model)
        X, y, a = data.train.X.to(dev), data.train.y.to(dev), data.train.a.to(dev)
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
            diff = (weight_importance(model, theta, X[i1], y[i1], repeats, gen)
                    - weight_importance(model, theta, X[i2], y[i2], repeats, gen))
            mask |= set(torch.topk(diff, k).indices.tolist())
        set_params(model, theta)
        res.metrics["mask"] = sorted(mask)
        res.params = get_params(model)
        return res
