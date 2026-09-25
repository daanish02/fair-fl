import torch
import torch.nn.functional as F

from fairfl.clients.base import ClientAlgorithm, model_device


class ReweighClient(ClientAlgorithm):
    """Local pre-processing debiasing (Kamiran & Calders 2012): sample weight P(a)P(y)/P(a,y) on local data.

    FairFed (AAAI 2023) is evaluated with this as its local debiasing method.
    """

    def batch_loss(self, model, X, y, a, ins, state):
        w = state.get("reweigh")
        if w is None:
            return F.cross_entropy(model(X), y)
        per = F.cross_entropy(model(X), y, reduction="none")
        sw = w[a, y]
        return (per * sw).sum() / sw.sum()

    def fit(self, model, data, ins, state, gen):
        if "reweigh" not in state:
            tr = data.train
            A, Y = tr.num_groups, tr.num_classes
            n = len(tr)
            w = torch.ones(A, Y, device=model_device(model))
            for g in range(A):
                for c in range(Y):
                    nag = int(((tr.a == g) & (tr.y == c)).sum())
                    if nag:
                        w[g, c] = (int((tr.a == g).sum()) / n) * (int((tr.y == c).sum()) / n) / (nag / n)
            state["reweigh"] = w
        return super().fit(model, data, ins, state, gen)
