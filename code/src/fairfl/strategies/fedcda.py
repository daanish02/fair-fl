from __future__ import annotations

import itertools
from collections import deque

import numpy as np
import torch
from pydantic import BaseModel, Field

from fairfl.registry import register_strategy
from fairfl.strategies.base import Strategy, weighted_mean


class FedCDAParams(BaseModel):
    K: int = Field(3, ge=1, description="Cached local models per client.")
    num_batches: int = Field(3, ge=1, description="B: participants are split into B near-equal batches (paper: 3).")
    smoothness: float = Field(1.0, ge=0.0, description="L in L_n(w) = F_n(w) + L/2 ||w||^2.")
    warmup: int = Field(50, ge=0, description="FedAvg rounds before cross-round selection starts (paper: 50).")


def _dot(a: dict, b: dict) -> float:
    return sum(float((a[k] * b[k]).sum()) for k in a if a[k].is_floating_point())


@register_strategy("fedcda")
class FedCDA(Strategy):
    """Wang et al., ICLR 2024 (Algorithm 1, batch-greedy objective Eq 8).

    For each participating client pick, from its K most recent local models, the combination minimising
    mean_n F_n(w_n) + L/2 * Var(w) jointly with the fixed models of the other clients; the global model is
    the plain average over every client's selected model.
    """

    Params = FedCDAParams

    def initialize(self, global_state, num_clients, client_sizes):
        super().initialize(global_state, num_clients, client_sizes)
        self.cache: dict[int, deque] = {}
        self.selected: dict[int, tuple[dict, float]] = {}

    def _objective(self, models: list[tuple[dict, float]]) -> float:
        L = self.p.smoothness
        keys = [k for k in models[0][0] if models[0][0][k].is_floating_point()]
        mean = {k: torch.stack([w[k] for w, _ in models]).mean(0) for k in keys}
        return sum(f + L / 2 * _dot(w, w) for w, f in models) / len(models) - L / 2 * _dot(mean, mean)

    def aggregate(self, rnd, global_state, results, round_lr):
        for r in results:
            self.cache.setdefault(r.client_id, deque(maxlen=self.p.K)).append(
                (r.state, float(r.metrics["train_loss"])))
        if rnd < self.p.warmup:
            for r in results:
                self.selected[r.client_id] = self.cache[r.client_id][-1]
            return weighted_mean([r.state for r in results], [r.num_samples for r in results])

        part = [r.client_id for r in results]
        order = self.rng.permutation(part).tolist()
        fixed = {c: m for c, m in self.selected.items() if c not in part}
        best_obj = float("inf")
        for batch in ([int(c) for c in x] for x in np.array_split(order, min(self.p.num_batches, len(order)))):
            options = [list(self.cache[c]) for c in batch]
            best, best_obj = None, float("inf")
            for combo in itertools.product(*options):
                obj = self._objective(list(fixed.values()) + list(combo))
                if obj < best_obj:
                    best, best_obj = combo, obj
            for c, m in zip(batch, best):
                fixed[c] = m
        self.selected = fixed
        self.info = {"objective": best_obj}
        keys = [k for k in global_state if global_state[k].is_floating_point()]
        out = {k: torch.stack([w[k] for w, _ in self.selected.values()]).mean(0) for k in keys}
        for k in global_state:
            if k not in out:
                out[k] = results[0].state[k]
        return out
