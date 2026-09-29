"""Status function U(tau_t): per-stakeholder benefit accumulated over the deployed history.

Benefits (not gaps) are accumulated, so an early advantage for one group is only cancelled by a later
advantage for the other group -- exactly the vaccine example of Alamdari et al. Unfairness at time t is
the spread of the accumulated statuses.
"""

from __future__ import annotations

import warnings

import numpy as np

from fairfl.core.types import RoundLog
from fairfl.fairness.scheme import FairnessScheme, StatusSpec
from fairfl.metrics.group import gap, group_benefit, total


def deployed_rounds(logs: list[RoundLog], deploy_every: int) -> list[int]:
    """Indices into `logs` of deployed models: every k-th round, plus the final (possibly post-processed) model."""
    if not logs:
        return []
    idx = [i for i, log in enumerate(logs) if (log.round + 1) % deploy_every == 0]
    if not idx or idx[-1] != len(logs) - 1:
        idx.append(len(logs) - 1)
    return idx


def benefit_matrix(logs: list[RoundLog], spec: StatusSpec, benefit: str) -> np.ndarray:
    """Array [T, K, G]: T deployed steps, K stakeholder sets (1 for global/clients), G stakeholders per set."""
    rows = []
    for log in logs:
        if spec.level == "global":
            rows.append(group_benefit(total([c.groups for c in log.clients]), benefit)[None, :])
        elif spec.level in ("local_max", "local_mean"):
            rows.append(np.stack([group_benefit(c.groups, benefit) for c in log.clients]))
        elif spec.level == "clients":
            rows.append(np.array([[c.accuracy for c in log.clients]]))
        else:
            raise ValueError(spec.level)
    return np.stack(rows)


def accumulate(b: np.ndarray, spec: StatusSpec) -> np.ndarray:
    """Running status U_t from per-step benefits b_t (NaN-aware normalised sums)."""
    T = b.shape[0]
    out = np.full_like(b, np.nan)
    for t in range(T):
        if spec.accumulation == "instant":
            w = np.zeros(t + 1)
            w[t] = 1.0
        elif spec.accumulation == "cumulative":
            w = np.ones(t + 1)
        elif spec.accumulation == "window":
            w = np.zeros(t + 1)
            w[max(0, t + 1 - spec.window):] = 1.0
        elif spec.accumulation == "discounted":
            w = spec.discount ** np.arange(t, -1, -1, dtype=float)
        else:
            raise ValueError(spec.accumulation)
        x = b[: t + 1]
        mask = ~np.isnan(x)
        ww = w.reshape(-1, *([1] * (x.ndim - 1))) * mask
        with np.errstate(invalid="ignore", divide="ignore"):
            out[t] = np.where(ww.sum(0) > 0, np.nansum(np.nan_to_num(x) * ww, 0) / ww.sum(0), np.nan)
    return out


def _spread(U_t: np.ndarray, level: str) -> float:
    if level == "clients":
        v = U_t[0][~np.isnan(U_t[0])]
        return float(v.mean() - v.min()) if len(v) else np.nan
    per_set = np.array([gap(row) for row in U_t])
    if np.all(np.isnan(per_set)):
        return np.nan
    return float(np.nanmax(per_set) if level in ("global", "local_max") else np.nanmean(per_set))


def unfairness_trajectory(logs: list[RoundLog], scheme: FairnessScheme) -> tuple[list[int], np.ndarray]:
    """Unfairness u_t of the accumulated status at each deployed round. Returns (log indices, u)."""
    dep = deployed_rounds(logs, scheme.deploy_every)
    dlogs = [logs[i] for i in dep]
    spec = scheme.status
    benefits = ["tpr", "fpr"] if spec.benefit == "eo" else [spec.benefit]
    us = []
    for b in benefits:
        U = accumulate(benefit_matrix(dlogs, spec, b), spec)
        us.append([_spread(U[t], spec.level) for t in range(len(dlogs))])
    arr = np.array(us, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN when a dataset has no sensitive attribute
        return dep, np.nanmax(arr, axis=0) if len(us) > 1 else arr[0]
