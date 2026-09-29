"""RDP accountant for the Poisson-subsampled Gaussian mechanism (Mironov et al. 2019), with the (eps, delta)
conversion of Balle et al. 2020 (Opacus get_privacy_spent). Used to set FedFDP's number of rounds for a target eps."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln, logsumexp

ORDERS = list(range(2, 64))


def log_a(q: float, sigma: float, alpha: int) -> float:
    """log A(q, sigma, alpha) for integer alpha: one subsampled Gaussian step."""
    k = np.arange(alpha + 1)
    log_binom = gammaln(alpha + 1) - gammaln(k + 1) - gammaln(alpha - k + 1)
    with np.errstate(divide="ignore"):
        terms = log_binom + (alpha - k) * math.log1p(-q) + k * math.log(q) + (k * k - k) / (2 * sigma**2)
    return float(logsumexp(terms))


def rdp(q: float, sigma: float, steps: int, orders=ORDERS) -> np.ndarray:
    return np.array([steps * log_a(q, sigma, a) / (a - 1) for a in orders])


def eps_from_rdp(r: np.ndarray, delta: float, orders=ORDERS) -> float:
    a = np.array(orders, dtype=float)
    eps = r + np.log((a - 1) / a) - (math.log(delta) + np.log(a)) / (a - 1)
    return float(np.nanmin(eps))


def epsilon(q: float, sigma: float, steps: int, delta: float, extra: list[tuple[float, float]] = ()) -> float:
    """eps after `steps` rounds; `extra` = further (q, sigma) mechanisms composed each round (e.g. FedFDP's loss)."""
    r = rdp(q, sigma, steps)
    for q2, s2 in extra:
        r = r + rdp(q2, s2, steps)
    return eps_from_rdp(r, delta)


def max_steps(q: float, sigma: float, target_eps: float, delta: float, extra: list[tuple[float, float]] = ()) -> int:
    lo, hi = 0, 1
    while epsilon(q, sigma, hi, delta, extra) <= target_eps:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if epsilon(q, sigma, mid, delta, extra) <= target_eps else (lo, mid)
    return lo
