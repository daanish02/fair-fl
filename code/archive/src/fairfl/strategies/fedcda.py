import itertools
from collections import deque

import numpy as np
import torch
from pydantic import BaseModel, Field

from fairfl.core.params import weighted_mean
from fairfl.core.registry import register_strategy
from fairfl.strategies.base import Strategy


class FedCDAParams(BaseModel):
    K: int = Field(3, ge=1, description="Cached local models per client.")
    num_batches: int = Field(3, ge=1, description="B: participants are split into B near-equal batches (paper: 3).")
    smoothness: float = Field(1.0, ge=0.0, description="L in L_n(w) = F_n(w) + L/2 ||w||^2.")
    warmup: int = Field(50, ge=0, description="FedAvg rounds before cross-round selection starts (paper: 50).")


@register_strategy("fedcda")
class FedCDA(Strategy):
    """Wang et al., ICLR 2024 (Algorithm 1, batch-greedy objective Eq 8).

    For each participating client pick, from its K most recent local models, the combination minimising
    mean_n F_n(w_n) + L/2 * Var(w) jointly with the fixed models of the other clients; the global model is
    the plain average over every client's selected model.
    """

    Params = FedCDAParams

    def initialize(self, gparams, num_clients, client_sizes, layer_sizes):
        super().initialize(gparams, num_clients, client_sizes, layer_sizes)
        self.cache: dict[int, deque] = {}
        self.selected: dict[int, tuple[torch.Tensor, float]] = {}

    def _objective(self, models: list[tuple[torch.Tensor, float]]) -> float:
        L = self.p.smoothness
        mean = torch.stack([w for w, _ in models]).mean(0)
        return sum(f + L / 2 * float(w @ w) for w, f in models) / len(models) - L / 2 * float(mean @ mean)

    def aggregate(self, rnd, gparams, results):
        for r in results:
            self.cache.setdefault(r.client_id, deque(maxlen=self.p.K)).append((r.params, float(r.metrics["train_loss"])))
        if rnd < self.p.warmup:
            for r in results:
                self.selected[r.client_id] = self.cache[r.client_id][-1]
            return weighted_mean([r.params for r in results], [r.num_samples for r in results])
        part = [r.client_id for r in results]
        order = self.rng.permutation(part).tolist()
        fixed = {c: m for c, m in self.selected.items() if c not in part}
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
        return torch.stack([w for w, _ in self.selected.values()]).mean(0)
