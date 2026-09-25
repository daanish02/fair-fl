import numpy as np
from pydantic import BaseModel, Field

from fairfl.core.params import weighted_mean
from fairfl.core.registry import register_strategy
from fairfl.core.types import FitIns
from fairfl.strategies.base import Strategy


class FCFLParams(BaseModel):
    eta: float = Field(0.5, ge=0.0, description="Weight of unfairness in the queue update (official default "
                                                 "alpha = 0.5; 0 = FedAvg).")
    random_fraction: float = Field(0.0, ge=0.0, le=1.0,
                                   description="Share r of clients picked uniformly; the rest are top-Q. "
                                               "Official code uses pure top-Q (r = 0).")


@register_strategy("fcfl")
class FCFL(Strategy):
    """Wang et al., ECML-PKDD 2024. Accumulated unfairness queues drive client selection and aggregation weights.

    uf_i = max(Acc_global_est - Acc_i, 0)                          (Eq 1)
    Q_i  = max(Q_i + eta * uf_i - p_i * 1[selected last round], 0)  (Eq 2; p_i = last aggregation weight)
    p_i  = Q_i / sum_selected Q   (or n_i / sum n if all selected Q are 0)   (Eq 3)
    """

    Params = FCFLParams
    pre_eval_split = "test"  # official code: accuracy of the global model on each client's local test loader

    def initialize(self, gparams, num_clients, client_sizes, layer_sizes):
        super().initialize(gparams, num_clients, client_sizes, layer_sizes)
        self.Q = np.zeros(num_clients)
        self.last_p = np.zeros(num_clients)
        self.acc_est = 0.0
        self.weights: dict[int, float] = {}

    def configure_eval(self, rnd, ids):
        return ids if rnd > 0 else []

    def _select(self, ids: list[int], m: int) -> list[int]:
        k_rand = int(round(self.p.random_fraction * m))
        rand = self.rng.choice(ids, size=k_rand, replace=False).tolist() if k_rand else []
        rest = [i for i in ids if i not in rand]
        tie = self.rng.random(len(rest))
        order = sorted(range(len(rest)), key=lambda j: (-self.Q[rest[j]], tie[j]))
        return sorted(rand + [rest[j] for j in order[: m - k_rand]])

    def configure_round(self, rnd, gparams, ids, pre_eval):
        m = max(1, int(round(self.train_cfg.clients_per_round * len(ids))))
        if rnd == 0:
            chosen = self.sample_clients(rnd, ids)
        else:
            for i in ids:
                uf = max(self.acc_est - pre_eval[i].accuracy, 0.0)
                self.Q[i] = max(self.Q[i] + self.p.eta * uf - self.last_p[i], 0.0)
            chosen = self._select(ids, m)
        q = self.Q[chosen]
        if q.sum() == 0:
            n = np.array([self.client_sizes[i] for i in chosen], float)
            p = n / n.sum()
        else:
            p = q / q.sum()
        self.weights = dict(zip(chosen, p.tolist()))
        self.info = {"Q": self.Q.round(4).tolist()}
        return {cid: FitIns(params=gparams, config={"report_train_acc": True}) for cid in chosen}

    def aggregate(self, rnd, gparams, results):
        p = [self.weights[r.client_id] for r in results]
        self.acc_est = float(sum(pi * r.metrics["train_acc"] for pi, r in zip(p, results)) / sum(p))
        self.last_p = np.zeros(self.num_clients)
        for pi, r in zip(p, results):
            self.last_p[r.client_id] = pi
        return weighted_mean([r.params for r in results], p)
