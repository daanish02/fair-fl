"""Non-Markovian fairness scheme <U, W_ex, B> (Alamdari et al., ICML 2024) instantiated for FL.

Stakeholders are sensitive groups (global or per client) or clients. Each deployed
round contributes a per-stakeholder *benefit* (positive rate, TPR, FPR or accuracy).
The status U accumulates benefits over the deployed history; unfairness is the gap
between stakeholders' statuses; a scope decides at which deployed rounds it is judged.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Benefit = Literal["dp", "tpr", "fpr", "eo", "accuracy"]
Level = Literal["global", "local_max", "local_mean", "clients"]
Accumulation = Literal["instant", "cumulative", "window", "discounted"]
ScopeKind = Literal["long_term", "periodic", "anytime", "bounded"]


class StatusSpec(BaseModel):
    benefit: Benefit = "dp"
    level: Level = "global"
    accumulation: Accumulation = "cumulative"
    window: int = Field(10, ge=1)
    discount: float = Field(0.9, gt=0.0, le=1.0)


class ScopeSpec(BaseModel):
    kind: ScopeKind = "anytime"
    period: int = Field(5, ge=1)
    rounds: list[int] = Field(default_factory=list)


class FairnessScheme(BaseModel):
    status: StatusSpec = Field(default_factory=StatusSpec)
    scope: ScopeSpec = Field(default_factory=ScopeSpec)
    deploy_every: int = Field(1, ge=1, description="Filter B: a model is deployed every k rounds (and at the end).")
    epsilon: float = Field(0.05, ge=0.0, description="Unfairness threshold used for violation counts.")
