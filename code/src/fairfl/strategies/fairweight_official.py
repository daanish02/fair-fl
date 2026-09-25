"""FairWeight as released by the authors (fairweight.zip, fairweight.py), ported line by line so the published
numbers can be reproduced. Kept on purpose (see docs/repro/fairweight.md section 9):
- the model outputs a probability p = sigmoid(z) and the loss is BCEWithLogits(p, y) (a second sigmoid) with
  pos_weight 10 for positives; the fairness constraint applies yet another sigmoid;
- Adam (lr 0.001) re-created every round; `local_steps` full-batch steps;
- DP loss: 100 * 2 * (E[s(p) | a=0] - E[s(p)])^2 (the constraint-matrix bug constrains group 0 only);
- weight importance: 25% of parameters zeroed per repeat, BCELoss(p, y), |grad * theta| summed over repeats and
  averaged over batches of 40; top-750 of the signed WI difference per group pair; union of the b and c sets;
- aggregation scores from the code's hard-coded table (0.66 / 0.33 for 3 clients), (beta1 s)^beta2, normalised per
  coordinate, unweighted mean for unflagged coordinates.
The ATE/FACE loss (Table III) needs the authors' propensity-matched potential outcomes and is not ported yet.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from fairfl.clients.base import ClientAlgorithm, model_device
from fairfl.core.params import get_params, set_params
from fairfl.core.registry import register_strategy
from fairfl.core.types import FitIns, FitRes
from fairfl.strategies.base import Strategy

# fairweight.py: normalised_scores per client count; values for 5/10/15 equal (n - f) / n, 3 is rounded.
SCORE_TABLES = {3: {0: 1, 1: 0.66, 2: 0.33, 3: 0}}


def official_score(n: int, freq: int) -> float:
    return SCORE_TABLES[n][freq] if n in SCORE_TABLES else (n - freq) / n


class FairWeightOfficialParams(BaseModel):
    fair_loss: Literal["dp", "none"] = Field("dp", description="dp = Table II; none = the FedAvg baseline row.")
    aggregation: Literal["fairweight", "mean"] = Field("fairweight", description="mean = plain unweighted FedAvg.")
    alpha: float = Field(100.0, description="Fairness constraint strength.")
    pos_weight: float = Field(10.0, description="cost_false_negatives.")
    local_steps: int = Field(15, ge=1, description="--epochs: full-batch Adam steps per round.")
    lr: float = 0.001
    beta1: float = 1.5
    beta2: float = 100.0
    top_k: int = Field(750, description="imp_weights_for_bias.")
    repeats: int = 100
    step: int = 40
    cap: int = 1000
    use_b_set: bool = Field(True, description="The code skips the b (bias) set for KDD.")


def prob(model, X) -> torch.Tensor:
    return F.softmax(model(X), dim=1)[:, 1]  # models expose [0, z]: softmax(.)[1] = sigmoid(z) = p


def weight_importance(model, theta, X, y, repeats, gen, step):
    """calculate_shapley_values_fa summed over repeats, accumulated over batches of `step`, divided by the batch count."""
    n_params = len(theta)
    total = torch.zeros_like(theta)
    runs = len(y) // step
    for r in range(runs):
        xb, yb = X[r * step : (r + 1) * step], y[r * step : (r + 1) * step].float()
        for _ in range(repeats):
            keep = torch.ones_like(theta)
            keep[torch.randperm(n_params, generator=gen)[: int(n_params * 0.25)].to(theta.device)] = 0
            set_params(model, theta * keep)
            model.zero_grad()
            F.binary_cross_entropy(prob(model, xb), yb).backward()
            g = torch.cat([p.grad.flatten() for p in model.parameters()])
            total += (g * theta).abs()
    set_params(model, theta)
    return total / runs if runs else None


class FairWeightOfficialClient(ClientAlgorithm):
    def fit(self, model, data, ins, state, gen):
        c = ins.config
        set_params(model, ins.params)
        dev = model_device(model)
        X, y, a = data.train.X.to(dev), data.train.y.to(dev), data.train.a.to(dev)
        yf = y.float()
        opt = torch.optim.Adam(model.parameters(), lr=float(c["lr"]))
        pw = torch.where(y == 1, float(c["pos_weight"]), 1.0)
        model.train()
        for _ in range(int(c["local_steps"])):
            opt.zero_grad()
            p = prob(model, X)
            loss = F.binary_cross_entropy_with_logits(p, yf, pos_weight=pw)
            if c["fair_loss"] == "dp":
                sp = torch.sigmoid(p)
                loss = loss + float(c["alpha"]) * 2 * (sp[a == 0].mean() - sp.mean()) ** 2
            loss.backward()
            opt.step()
        theta = get_params(model)
        metrics = {"train_loss": float(loss.detach())}
        if c["aggregation"] == "fairweight":
            with torch.no_grad():
                pred = prob(model, X).round().long()
            correct = pred == y
            t_map = torch.nonzero((a == 1) & (y == 0) & correct).flatten()
            f_mip = torch.nonzero((a == 0) & (y == 0) & ~correct).flatten()
            t_mip = torch.nonzero((a == 0) & (y == 0) & correct).flatten()
            f_min = torch.nonzero((a == 0) & (y == 1) & ~correct).flatten()
            pairs = ([(t_map, f_mip)] if c["use_b_set"] else []) + [(t_mip, f_min)]
            mask: set[int] = set()
            for s1, s2 in pairs:
                n = min(len(s1), len(s2), int(c["cap"]))
                i1, i2 = s1[:n], s2[:n]
                w1 = weight_importance(model, theta, X[i1], y[i1], int(c["repeats"]), gen, int(c["step"]))
                w2 = weight_importance(model, theta, X[i2], y[i2], int(c["repeats"]), gen, int(c["step"]))
                if w1 is None or w2 is None:
                    continue
                k = min(int(c["top_k"]), len(theta))
                mask |= set(torch.topk(w1 - w2, k).indices.tolist())
            metrics["mask"] = sorted(mask)
        set_params(model, theta)
        return FitRes(client_id=data.client_id, params=theta, num_samples=len(y), metrics=metrics)


@register_strategy("fairweight_official")
class FairWeightOfficial(Strategy):
    """Kasyap et al., IEEE TSC 2026, as released. Full participation."""

    Params = FairWeightOfficialParams
    client_cls = FairWeightOfficialClient

    def configure_round(self, rnd, gparams, ids, pre_eval):
        return {cid: FitIns(params=gparams, config=self.p.model_dump()) for cid in ids}

    def aggregate(self, rnd, gparams, results):
        stack = torch.stack([r.params for r in results])
        out = stack.mean(dim=0)
        if self.p.aggregation == "mean":
            return out
        masks = [set(r.metrics.get("mask", [])) for r in results]
        coords = sorted(set().union(*masks))
        if not coords:
            return out
        n = len(results)
        freq = {j: sum(j in m for m in masks) for j in coords}
        s = np.array([[1.0 if j not in m else official_score(n, freq[j]) for j in coords] for m in masks])
        w = np.power(self.p.beta1 * s, self.p.beta2)
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.nan_to_num(w / w.sum(axis=0, keepdims=True))
        idx = torch.tensor(coords, device=stack.device)
        out[idx] = (stack[:, idx] * torch.from_numpy(w).float().to(stack.device)).sum(dim=0)
        self.info = {"flagged_coords": len(coords)}
        return out
