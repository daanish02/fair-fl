"""LoGoFair DP post-processing, ported line by line from the official code (github.com/liizhang/LoGofair,
fedlearn/models/FedFairPostClient.py: calibration, obj_H, true_H, local_fair_post, local_post_eval; and
fedlearn/algorithm/FedFairPost.py: the post-processing rounds). Code quirks are kept on purpose because the goal is
to reproduce the published numbers: gradients accumulate across the mu and lambda steps of local_fair_post (no
zero_grad), lambda is an unclamped scalar penalised by |lambda|, p_A uses all-split group counts over the validation
size, and the server averages the clients' lambdas. See docs/repro/logofair.md for the paper-vs-code list.
Sensitive groups follow the official split files: A = 0 male, A = 1 female.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, Field

from fairfl.clients.base import predict_proba
from fairfl.core.params import set_params
from fairfl.core.registry import register_strategy
from fairfl.strategies.base import Strategy


class LoGoFairOfficialParams(BaseModel):
    mode: Literal["lg", "l", "g"] = Field("lg", description="lg = both constraints; l fixes lambda = 0; g fixes mu = 0.")
    delta_g: float = Field(0.01, ge=0.0, description="Paper Table 1: 0.01 (code default 0.02).")
    delta_l: float = Field(0.01, ge=0.0, description="Paper Table 1: 0.01 (code default 0.02).")
    beta: float = Field(1000.0, gt=0.0, description="FFP_beta.")
    post_lr: float = 0.005
    post_rounds: int = 30
    mu_steps: int = 20
    lamb_steps: int = 20
    lamb_init: float = 0.01
    mu_init: float = 0.5
    calibrate: bool = True


class BetaCalibration:
    """Beta calibration (Kull et al. 2017; netcal default): logistic regression on [ln s, -ln(1 - s)]."""

    def fit(self, s: np.ndarray, y: np.ndarray) -> BetaCalibration:
        from sklearn.linear_model import LogisticRegression

        self.lr = LogisticRegression(C=1e6, max_iter=2000).fit(self._feats(s), y)
        return self

    @staticmethod
    def _feats(s: np.ndarray) -> np.ndarray:
        s = np.clip(s, 1e-7, 1 - 1e-7)
        return np.stack([np.log(s), -np.log(1 - s)], axis=1)

    def transform(self, s: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(self._feats(s))[:, 1]


def r_beta(x: torch.Tensor, beta: float) -> torch.Tensor:
    # Official code: (1/beta) * log(1 + exp(beta * x)). In float32 exp(beta * x) overflows once beta * x > 88
    # (every score near 1 with beta = 1000), which turns the gradients into NaN. softplus is the same function,
    # computed stably.
    return torch.nn.functional.softplus(x, beta=beta, threshold=20.0)


class _ClientPP:
    def __init__(self, score, A, pi, N, p_A, p: LoGoFairOfficialParams, num_clients: int):
        self.score, self.A, self.pi, self.N, self.p_A, self.p, self.C = score, A, pi, N, p_A, p, num_clients
        self.mu = torch.tensor([p.mu_init, p.mu_init])
        self.dg, self.dl = torch.tensor(p.delta_g), torch.tensor(p.delta_l)

    def _terms(self, mu, lamb, fn):
        s0, s1 = self.score[self.A == 0], self.score[self.A == 1]
        f0 = 1 / self.N[0] * fn(self.pi[0] * (2 * s0 - 1) - lamb * self.pi[0] / self.p_A[0] - (mu[0] - mu[1]))
        f1 = 1 / self.N[1] * fn(self.pi[1] * (2 * s1 - 1) + lamb * self.pi[1] / self.p_A[1] + (mu[0] - mu[1]))
        return f0.sum() + f1.sum() + self.dg * torch.abs(lamb) / self.C + self.dl * (mu[0] + mu[1])

    def obj_H(self, mu, lamb):
        return self._terms(mu, lamb, lambda x: r_beta(x, self.p.beta))

    def true_H(self, mu, lamb):
        return self._terms(mu, lamb, torch.relu)

    def local_fair_post(self, lamb: torch.Tensor) -> torch.Tensor:
        if self.p.mode != "g":
            mu = self.mu.clone().detach().requires_grad_(True)
            for _ in range(self.p.mu_steps):
                self.obj_H(mu, lamb).backward()  # grad accumulates, as in the official code
                with torch.no_grad():
                    mu *= torch.exp(-self.p.post_lr * mu.grad)
                    mu.clamp_(min=0)
            self.mu = mu.detach().clone()
        if self.p.mode == "l":
            return torch.zeros(1)
        lam = lamb.clone().detach().requires_grad_(True)
        for _ in range(self.p.lamb_steps):
            self.obj_H(self.mu, lam).backward()
            with torch.no_grad():
                lam -= self.p.post_lr * 5 * lam.grad
        return lam.detach().clone()

    def thresholds(self, lamb: torch.Tensor) -> tuple[float, float]:
        if self.p.mode != "g":
            mu = self.mu.clone().detach().requires_grad_(True)
            for _ in range(self.p.mu_steps * 5):
                self.true_H(mu, lamb).backward()
                with torch.no_grad():
                    mu *= torch.exp(-self.p.post_lr * mu.grad)
                    mu.clamp_(min=0)
                mu.grad.zero_()
            self.mu = mu.detach().clone()
        m = self.mu[0] - self.mu[1] if self.p.mode != "g" else torch.tensor(0.0)
        t0 = 0.5 + 0.5 / self.p_A[0] * lamb + 0.5 / self.pi[0] * m
        t1 = 0.5 - 0.5 / self.p_A[1] * lamb - 0.5 / self.pi[1] * m
        return float(t0), float(t1)


@register_strategy("logofair_official")
class LoGoFairOfficial(Strategy):
    """LoGoFair (Zhang et al., AAAI 2025), DP variant, as implemented in the authors' code."""

    Params = LoGoFairOfficialParams

    def initialize(self, gparams, num_clients, client_sizes, layer_sizes):
        super().initialize(gparams, num_clients, client_sizes, layer_sizes)
        self.rules = None

    def postprocess(self, gparams, model, clients) -> bool:
        set_params(model, gparams)
        vals = [c.val if c.val is not None else c.train for c in clients]
        val_num = sum(len(v) for v in vals)
        a_info = [sum(int((d.a == g).sum()) for c in clients for d in (c.train, c.val or c.train, c.test))
                  for g in (0, 1)]
        p_A = [torch.tensor(a_info[0] / val_num), torch.tensor(a_info[1] / val_num)]
        self.cal: dict[int, list] = {}
        pps = []
        for c, v in zip(clients, vals):
            score = predict_proba(model, v.X)[:, 1].float().cpu()
            A = v.a.cpu()
            cals = [None, None]
            if self.p.calibrate:
                for g in (0, 1):
                    m = A == g
                    cals[g] = BetaCalibration().fit(score[m].numpy(), v.y.cpu()[m].numpy())
                    score[m] = torch.tensor(cals[g].transform(score[m].numpy()), dtype=torch.float32)
            self.cal[c.client_id] = cals
            N = [torch.tensor(float((A == g).sum())) for g in (0, 1)]
            pi = [n / val_num for n in N]
            pps.append(_ClientPP(score, A, pi, N, p_A, self.p, len(clients)))
        lamb = torch.tensor([self.p.lamb_init if self.p.mode != "l" else 0.0])
        for _ in range(self.p.post_rounds):
            lamb = sum(pp.local_fair_post(lamb) for pp in pps) / len(pps)
        self.lamb = lamb
        self.thres = {c.client_id: pp.thresholds(lamb) for c, pp in zip(clients, pps)}
        self.info = {"lambda": float(lamb), "thresholds": self.thres,
                     "mu": {c.client_id: pp.mu.tolist() for c, pp in zip(clients, pps)}}
        self.rules = True
        return True

    def decision_rule(self):
        if self.rules is None:
            return super().decision_rule()

        def rule(cid, probs, a):
            s = probs[:, 1].float().cpu().clone()
            a = a.cpu()
            cals = self.cal[cid]
            out = torch.zeros_like(s, dtype=torch.long)
            for g in (0, 1):
                m = a == g
                if not m.any():
                    continue
                sg = s[m]
                if cals[g] is not None:
                    sg = torch.tensor(cals[g].transform(sg.numpy()), dtype=torch.float32)
                out[m] = (sg >= self.thres[cid][g]).long()
            return out.to(probs.device)

        return rule
