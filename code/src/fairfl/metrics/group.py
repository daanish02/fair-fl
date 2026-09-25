"""Per-group benefits and classic (single-model) group-fairness gaps from logged confusion counts."""

from __future__ import annotations

import numpy as np

from fairfl.core.types import GroupCounts


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else np.nan


def group_benefit(c: GroupCounts, benefit: str) -> np.ndarray:
    """Benefit per sensitive group; NaN where a group has no relevant samples."""
    n, pos, cor = np.array(c.n, float), np.array(c.pred_pos, float), np.array(c.correct, float)
    if benefit == "dp":
        return np.array([_ratio(pos[g].sum(), n[g].sum()) for g in range(len(n))])
    if benefit == "tpr":
        return np.array([_ratio(pos[g][1], n[g][1]) for g in range(len(n))])
    if benefit == "fpr":
        return np.array([_ratio(pos[g][0], n[g][0]) for g in range(len(n))])
    if benefit == "accuracy":
        return np.array([_ratio(cor[g].sum(), n[g].sum()) for g in range(len(n))])
    raise ValueError(f"unknown benefit {benefit!r}")


def gap(values: np.ndarray) -> float:
    v = values[~np.isnan(values)]
    return float(v.max() - v.min()) if len(v) >= 2 else np.nan


def dp_gap(c: GroupCounts) -> float:
    return gap(group_benefit(c, "dp"))


def eo_gap(c: GroupCounts) -> float:
    g = [gap(group_benefit(c, "tpr")), gap(group_benefit(c, "fpr"))]
    g = [x for x in g if not np.isnan(x)]
    return max(g) if g else np.nan


def total(counts: list[GroupCounts]) -> GroupCounts:
    out = counts[0]
    for c in counts[1:]:
        out = out + c
    return out
