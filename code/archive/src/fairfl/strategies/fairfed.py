from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from fairfl.clients.base import SGDClient
from fairfl.clients.reweigh import ReweighClient
from fairfl.core.params import weighted_mean
from fairfl.core.registry import register_strategy
from fairfl.core.types import FitIns, GroupCounts
from fairfl.metrics.group import group_benefit, total
from fairfl.strategies.base import Strategy


class FairFedParams(BaseModel):
    beta: float = Field(1.0, ge=0.0, description="Fairness budget (0 = FedAvg).")
    metric: Literal["eod", "spd"] = "eod"
    local_reweight: bool = Field(True, description="Clients debias locally with reweighing, as in the paper.")


def signed_metric(c: GroupCounts, metric: str) -> float:
    b = group_benefit(c, "tpr" if metric == "eod" else "dp")
    return float(b[0] - b[1]) if len(b) == 2 else float("nan")


@register_strategy("fairfed")
class FairFed(Strategy):
    """Ezzeldin et al., AAAI 2023 (Eq 6).

    Delta_k = |F_global - F_k| (or |Acc_global - Acc_k| if F_k is undefined locally)
    w_k^t   = w_k^{t-1} - beta * (Delta_k - mean(Delta)),  normalised
    Metrics are computed on each client's local data with the current global model.
    """

    Params = FairFedParams

    def initialize(self, gparams, num_clients, client_sizes, layer_sizes):
        super().initialize(gparams, num_clients, client_sizes, layer_sizes)
        n = np.array(client_sizes, float)
        self.wbar = n / n.sum()

    def make_client(self):
        return (ReweighClient if self.p.local_reweight else SGDClient)(self.train_cfg)

    def configure_eval(self, rnd, ids):
        return ids if rnd > 0 else []

    def configure_round(self, rnd, gparams, ids, pre_eval):
        if rnd > 0:
            f_glob = signed_metric(total([e.groups for e in pre_eval.values()]), self.p.metric)
            n_all = sum(e.num_samples for e in pre_eval.values())
            acc_glob = sum(e.accuracy * e.num_samples for e in pre_eval.values()) / n_all
            delta = np.zeros(self.num_clients)
            for i in ids:
                f_k = signed_metric(pre_eval[i].groups, self.p.metric)
                delta[i] = abs(f_glob - f_k) if not (np.isnan(f_k) or np.isnan(f_glob)) else abs(acc_glob - pre_eval[i].accuracy)
            self.wbar = np.maximum(self.wbar - self.p.beta * (delta - delta[ids].mean()), 0.0)
            if self.wbar.sum() == 0:
                self.wbar = np.ones(self.num_clients)
        self.info = {"weights": (self.wbar / self.wbar.sum()).round(4).tolist()}
        return {cid: FitIns(params=gparams) for cid in self.sample_clients(rnd, ids)}

    def aggregate(self, rnd, gparams, results):
        return weighted_mean([r.params for r in results], [max(self.wbar[r.client_id], 1e-12) for r in results])
