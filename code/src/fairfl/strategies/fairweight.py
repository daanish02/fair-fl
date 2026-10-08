from __future__ import annotations

import numpy as np
import torch
from pydantic import BaseModel, Field

from fairfl.clients.fairweight import FairWeightClient
from fairfl.registry import register_strategy
from fairfl.strategies.base import FitIns, Strategy, flatten_state, unflatten


class FairWeightParams(BaseModel):
    gamma1: float = Field(1.5, ge=1.0, description="beta1 in the official code; incentivises unbiased coordinates.")
    gamma2: float = Field(100.0, ge=1.0, description="beta2 in the official code; sharpness of the penalty.")
    top_fraction: float = Field(0.1, gt=0.0, le=1.0, description="Share of parameters marked responsible per group pair.")
    fair_weight: float = Field(1.0, ge=0.0, description="Weight of the local demographic-parity penalty.")
    repeats: int = Field(5, ge=1, description="Random-zeroing repeats for the Shapley approximation (official: 100).")
    max_samples: int = Field(400, ge=1)


def coordinate_scores(masks: list[set[int]], coords: list[int], gamma1: float, gamma2: float) -> np.ndarray:
    """Per-client score for each coordinate in `coords`: 1 if not flagged by the client, else (n - freq)/n;
    then (gamma1 * s) ** gamma2, normalised across clients (0/0 -> 0 as in the official code)."""
    n = len(masks)
    freq = {c: sum(c in m for m in masks) for c in coords}
    s = np.array([[1.0 if c not in m else (n - freq[c]) / n for c in coords] for m in masks])
    s = np.power(gamma1 * s, gamma2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(s / s.sum(axis=0, keepdims=True))


@register_strategy("fairweight")
class FairWeight(Strategy):
    """Kasyap, Atmaca, Maple, Lane, IEEE TSC 2026. Coordinates flagged as bias-responsible are aggregated with
    fairness scores; all other coordinates are averaged uniformly. Secure aggregation is simulated as a plain sum.

    NOTE: ported onto CIFAR10 with a synthetic proxy group attribute (vehicle vs animal classes, see
    fairfl.data.cifar.group_attr) since this codebase keeps one dataset across strategies; the demographic-parity
    numbers this produces describe that proxy split, not a real protected attribute.
    """

    Params = FairWeightParams
    client_cls = FairWeightClient

    def configure_round(self, rnd, global_state, ids, pre_eval, round_lr):
        cfg = {"lr": round_lr, **self.p.model_dump(include={"top_fraction", "fair_weight", "repeats", "max_samples"})}
        return {cid: FitIns(state=global_state, config=cfg) for cid in ids}

    def aggregate(self, rnd, global_state, results, round_lr):
        # Client-side masks index into parameter-only space (clients/fairweight.py restricts the Shapley flatten
        # to trainable parameters, since buffers like BatchNorm running stats have no gradient); mirror that here
        # so a mask index means the same coordinate on both sides. Buffers aren't in state_dict's own metadata,
        # so they're excluded by the standard PyTorch naming convention instead.
        param_keys = {k for k in global_state if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))}
        _, key_shapes = flatten_state(global_state, keys=param_keys)
        stack = torch.stack([flatten_state(r.state, keys=param_keys)[0] for r in results])
        out = stack.mean(dim=0)
        masks = [set(r.metrics.get("mask", [])) for r in results]
        coords = sorted(set().union(*masks))
        if coords:
            w = coordinate_scores(masks, coords, self.p.gamma1, self.p.gamma2)
            w = torch.from_numpy(w).float().to(stack.device)
            idx = torch.tensor(coords, device=stack.device)
            out[idx] = (stack[:, idx] * w).sum(dim=0)
        self.info = {"flagged_coords": len(coords)}
        return unflatten(out, key_shapes, global_state)
