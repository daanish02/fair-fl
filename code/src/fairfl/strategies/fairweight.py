import numpy as np
import torch
from pydantic import BaseModel, Field

from fairfl.clients.fairweight_local import FairWeightClient
from fairfl.core.registry import register_strategy
from fairfl.core.types import FitIns
from fairfl.strategies.base import Strategy


class FairWeightParams(BaseModel):
    gamma1: float = Field(1.5, ge=1.0, description="beta1 in the official code; incentivises unbiased coordinates.")
    gamma2: float = Field(100.0, ge=1.0, description="beta2 in the official code; sharpness of the penalty.")
    top_fraction: float = Field(0.1, gt=0.0, le=1.0, description="Share of parameters marked responsible per group pair.")
    fair_weight: float = Field(1.0, ge=0.0, description="Weight of the local demographic-parity penalty.")
    repeats: int = Field(5, ge=1, description="Random-zeroing repeats for the Shapley approximation (official: 100).")
    max_samples: int = Field(400, ge=1)


def coordinate_scores(masks: list[set[int]], coords: list[int], gamma1: float, gamma2: float) -> np.ndarray:
    """Per-client score for each coordinate in `coords`: 1 if not flagged by the client, else (n - freq)/n;
    then (gamma1 * s) ** gamma2, normalised across clients (Section IV-3; 0/0 -> 0 as in the official code)."""
    n = len(masks)
    freq = {c: sum(c in m for m in masks) for c in coords}
    s = np.array([[1.0 if c not in m else (n - freq[c]) / n for c in coords] for m in masks])
    s = np.power(gamma1 * s, gamma2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(s / s.sum(axis=0, keepdims=True))


@register_strategy("fairweight")
class FairWeight(Strategy):
    """Kasyap, Atmaca, Maple, Lane, IEEE TSC 2026. Coordinates flagged as bias-responsible are aggregated with
    fairness scores; all other coordinates are averaged uniformly. Secure aggregation is simulated as a plain sum."""

    Params = FairWeightParams
    client_cls = FairWeightClient

    def configure_round(self, rnd, gparams, ids, pre_eval):
        cfg = self.p.model_dump(include={"top_fraction", "fair_weight", "repeats", "max_samples"})
        return {cid: FitIns(params=gparams, config=cfg) for cid in self.sample_clients(rnd, ids)}

    def aggregate(self, rnd, gparams, results):
        stack = torch.stack([r.params for r in results])
        out = stack.mean(dim=0)
        masks = [set(r.metrics.get("mask", [])) for r in results]
        coords = sorted(set().union(*masks))
        if coords:
            w = coordinate_scores(masks, coords, self.p.gamma1, self.p.gamma2)
            w = torch.from_numpy(w).float().to(stack.device)
            idx = torch.tensor(coords, device=stack.device)
            out[idx] = (stack[:, idx] * w).sum(dim=0)
        self.info = {"flagged_coords": len(coords)}
        return out
