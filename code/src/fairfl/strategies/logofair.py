"""LoGoFair (Zhang et al., AAAI 2025): federated post-processing towards the Bayes-optimal classifier under local
and global fairness constraints. lambda (global) is learned by federated projected gradient descent; each client's
mu_c is solved locally and never leaves the client. The hinge (.)_+ is smoothed with r_beta.

NOTE: ported onto CIFAR10 with a synthetic proxy group attribute (vehicle vs animal classes, see
fairfl.data.cifar.group_attr) since this codebase keeps one dataset across strategies; the fairness numbers this
produces describe that proxy split, not a real protected attribute. Also: archive's version is binary-label (y in
{0,1}); ported here with y treated as "predicted class == 0" vs "not", a further simplification for 10-class CIFAR10.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from fairfl.registry import register_strategy
from fairfl.strategies.base import Strategy


class LoGoFairParams(BaseModel):
    mode: Literal["lg", "l", "g"] = Field("lg", description="Constraints: local+global, local only, global only.")
    metric: Literal["dp", "eo"] = "dp"
    delta_l: float = Field(0.01, ge=0.0, description="Local fairness slack.")
    delta_g: float = Field(0.01, ge=0.0, description="Global fairness slack.")
    beta: float = Field(2000.0, gt=0.0, description="Smoothing sharpness of r_beta.")
    post_rounds: int = Field(40, ge=1, description="Federated post-processing rounds (paper: converges in ~10).")
    lr: float = Field(8.0, gt=0.0, description="Server step size for lambda, relative to mean P(A=a, C=c).")
    inner_steps: int = Field(100, ge=0)
    inner_lr: float = Field(0.5, gt=0.0, description="Local step for mu_c, relative to the client's smallest P(A=a, C=c).")
    calibrate: bool = Field(True, description="Per-client temperature scaling of eta.")
    min_group_count: int = Field(20, ge=0, description="Clients whose split has fewer samples of a group get no "
                                                        "local constraint (rate cannot be estimated).")


def temperature_scale(eta: torch.Tensor, T: float) -> torch.Tensor:
    logit = torch.logit(eta.clamp(1e-6, 1 - 1e-6))
    return torch.sigmoid(logit / T)


def fit_temperature(eta: torch.Tensor, y: torch.Tensor) -> float:
    grid = torch.linspace(0.25, 4.0, 61)
    nll = [F.binary_cross_entropy(temperature_scale(eta, float(T)), y.double()).item() for T in grid]
    return float(grid[int(torch.tensor(nll).argmin())])


@dataclass
class ClientPP:
    """Everything one client needs for its local piece of the post-processing problem."""

    eta: torch.Tensor
    s: torch.Tensor          # sign of the sensitive attribute, +1 / -1
    p_ac: dict[int, float]   # joint P(A=a, C=c)
    phi: dict[int, list]     # global disparity coefficients per group: list of (constant, slope)
    psi: dict[int, list]     # local disparity coefficients per group
    T: float = 1.0


def linear_terms(coefs: list, eta: torch.Tensor) -> torch.Tensor:
    return torch.stack([c0 + c1 * eta for c0, c1 in coefs], dim=1)


def calib_F(pp: ClientPP, eta: torch.Tensor, s: torch.Tensor, lam: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    K = lam.numel() // 2
    out = torch.empty_like(eta)
    for sign in (-1, 1):
        m = s == sign
        if not m.any():
            continue
        e = eta[m]
        g = linear_terms(pp.phi[sign], e) @ (lam[:K] - lam[K:])
        loc = linear_terms(pp.psi[sign], e) @ (mu[:K] - mu[K:])
        out[m] = pp.p_ac[sign] * (2 * e - 1) - g - loc
    return out


def local_objective(pp: ClientPP, lam, mu, beta, delta_g, delta_l, n_clients) -> torch.Tensor:
    Fv = calib_F(pp, pp.eta, pp.s, lam, mu)
    h = sum(F.softplus(Fv[pp.s == sign], beta=beta).mean() for sign in (-1, 1) if (pp.s == sign).any())
    return h + delta_g / n_clients * lam.sum() + delta_l * mu.sum()


@register_strategy("logofair")
class LoGoFair(Strategy):
    Params = LoGoFairParams

    def initialize(self, global_state, num_clients, client_sizes):
        super().initialize(global_state, num_clients, client_sizes)
        self.pp: dict[int, ClientPP] | None = None

    def _coefs(self, sign: int, weight: float, py1: float, py0: float) -> list:
        if self.p.metric == "dp":
            return [(sign * weight, 0.0)]
        return [(0.0, sign * weight / max(py1, 1e-6)), (sign * weight / max(py0, 1e-6), -sign * weight / max(py0, 1e-6))]

    def postprocess(self, global_state, model, client_algo, client_data) -> bool:
        model.load_state_dict(global_state)
        model.eval()
        dev = next(model.parameters()).device
        stats = {}
        with torch.no_grad():
            for cid, ds in enumerate(client_data):
                probs = F.softmax(model(ds.X.to(dev)), dim=1)
                eta = probs[:, 0].double().cpu()  # P(predicted class == 0), the binarisation of a 10-class model
                y = (ds.y == 0).double()
                a = ds.a.double()
                T = fit_temperature(eta, y) if self.p.calibrate else 1.0
                stats[cid] = (temperature_scale(eta, T), (2 * a - 1).double(), y, T)

        N = sum(len(s[1]) for s in stats.values())
        N_a = {sign: sum(int((s[1] == sign).sum()) for s in stats.values()) for sign in (-1, 1)}
        Py1_a = {sign: sum(float(s[2][s[1] == sign].sum()) for s in stats.values()) / max(N_a[sign], 1)
                for sign in (-1, 1)}
        use_g, use_l = self.p.mode in ("lg", "g"), self.p.mode in ("lg", "l")
        self.pp = {}
        for cid, (eta, s, y, T) in stats.items():
            n_ac = {sign: int((s == sign).sum()) for sign in (-1, 1)}
            py1_ac = {sign: float(y[s == sign].mean()) if n_ac[sign] else 0.5 for sign in (-1, 1)}
            self.pp[cid] = ClientPP(
                eta=eta, s=s, T=T,
                p_ac={sign: n_ac[sign] / N for sign in (-1, 1)},
                phi={sign: self._coefs(sign, n_ac[sign] / max(N_a[sign], 1), Py1_a[sign], 1 - Py1_a[sign])
                    for sign in (-1, 1)},
                psi={sign: self._coefs(sign, 1.0, py1_ac[sign], 1 - py1_ac[sign]) for sign in (-1, 1)},
            )
        self.local_ok = {cid for cid, pp in self.pp.items()
                         if min(int((pp.s == sign).sum()) for sign in (-1, 1)) >= self.p.min_group_count}
        K = 1 if self.p.metric == "dp" else 2
        lam = torch.zeros(2 * K, dtype=torch.float64)
        self.mu = {cid: torch.zeros(2 * K, dtype=torch.float64) for cid in self.pp}
        n = len(self.pp)
        args = (self.p.beta, self.p.delta_g, self.p.delta_l, n)
        mean_p = sum(v for pp in self.pp.values() for v in pp.p_ac.values()) / (2 * n)
        for t in range(self.p.post_rounds):
            g_lam = torch.zeros_like(lam)
            for cid, pp in self.pp.items():
                mu = self.mu[cid]
                if use_l and cid in self.local_ok:
                    scale = self.p.inner_lr * max(min(pp.p_ac.values()), 1e-6)
                    for k in range(self.p.inner_steps):
                        mu = mu.detach().requires_grad_(True)
                        (gm,) = torch.autograd.grad(local_objective(pp, lam, mu, *args), mu)
                        mu = (mu - scale / (k + 1) ** 0.5 * gm).clamp_min(0.0)
                    self.mu[cid] = mu.detach()
                if use_g:
                    lv = lam.detach().requires_grad_(True)
                    (gl,) = torch.autograd.grad(local_objective(pp, lv, self.mu[cid], *args), lv)
                    g_lam += gl
            if use_g:
                lam = (lam - self.p.lr * mean_p / (t + 1) ** 0.5 * g_lam).clamp_min(0.0)
        self.lam = lam
        self.info = {"lambda": lam.tolist(), "local_constraint_clients": sorted(self.local_ok)}
        return True

    def decision_rule(self):
        if self.pp is None:
            return super().decision_rule()

        def rule(cid, logits, a):
            if cid not in self.pp:
                return logits.argmax(dim=1)
            pp = self.pp[cid]
            probs = F.softmax(logits, dim=1)
            eta = temperature_scale(probs[:, 0].double(), pp.T)
            is_zero = calib_F(pp, eta, (2 * a.double() - 1), self.lam, self.mu[cid]) >= 0
            return torch.where(is_zero, torch.zeros_like(a), logits[:, 1:].argmax(dim=1) + 1)

        return rule
