"""Temporal scopes: at which deployed rounds unfairness is judged, and how the judgements are aggregated (W_ex)."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from fairfl.core.types import RoundLog
from fairfl.fairness.scheme import FairnessScheme, ScopeSpec
from fairfl.fairness.status import unfairness_trajectory


class ScopeSummary(BaseModel):
    scope: str
    points: int
    final: float
    max: float
    mean: float
    violation_rate: float


def evaluation_points(dep_rounds: list[int], spec: ScopeSpec) -> list[int]:
    """Positions (into the deployed-rounds list) where fairness is judged."""
    n = len(dep_rounds)
    if n == 0:
        return []
    if spec.kind == "long_term":
        return [n - 1]
    if spec.kind == "anytime":
        return list(range(n))
    if spec.kind == "periodic":
        return list(range(spec.period - 1, n, spec.period)) or [n - 1]
    if spec.kind == "bounded":
        wanted = set(spec.rounds)
        return [i for i, r in enumerate(dep_rounds) if r in wanted] or [n - 1]
    raise ValueError(spec.kind)


def summarise(u: np.ndarray, points: list[int], epsilon: float, name: str) -> ScopeSummary:
    v = u[points]
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return ScopeSummary(scope=name, points=0, final=np.nan, max=np.nan, mean=np.nan, violation_rate=np.nan)
    return ScopeSummary(scope=name, points=len(v), final=float(v[-1]), max=float(v.max()), mean=float(v.mean()),
                        violation_rate=float((v > epsilon).mean()))


def evaluate_scheme(logs: list[RoundLog], scheme: FairnessScheme) -> ScopeSummary:
    dep, u = unfairness_trajectory(logs, scheme)
    pts = evaluation_points([logs[i].round for i in dep], scheme.scope)
    return summarise(u, pts, scheme.epsilon, scheme.scope.kind)


def evaluate_all_scopes(logs: list[RoundLog], scheme: FairnessScheme, period: int = 5) -> dict[str, ScopeSummary]:
    """Same status function judged under all four scopes (bounded = the scheme's own round list)."""
    out = {}
    for kind in ("long_term", "periodic", "anytime", "bounded"):
        s = scheme.model_copy(update={"scope": ScopeSpec(kind=kind, period=period, rounds=scheme.scope.rounds)})
        out[kind] = evaluate_scheme(logs, s)
    return out
