"""Client side of FedFair and FedFDP (Ling et al., ACNS 2026; arXiv 2402.16028 Algorithm 1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from fairfl.clients.base import ClientAlgorithm, batches, mean_loss, model_device
from fairfl.core.params import get_params, set_params
from fairfl.core.types import FitRes


class FedFairClient(ClientAlgorithm):
    """Eq 6-7: the local step is scaled by (1 + lambda * Delta), Delta = F_i(w) - F(w_t).

    F_i is the batch loss at the current iterate; F(w_t) is the broadcast global loss. The scale is floored
    at 0 so a batch far below the global loss cannot reverse the gradient (the paper does not discuss this case).
    """

    def batch_loss(self, model, X, y, a, ins, state):
        loss = F.cross_entropy(model(X), y)
        f_glob = ins.config.get("global_loss")
        if f_glob is None:
            return loss
        scale = max(0.0, 1.0 + float(ins.config["lam"]) * (loss.item() - float(f_glob)))
        return scale * loss

    def fit(self, model, data, ins, state, gen):
        res = super().fit(model, data, ins, state, gen)
        res.metrics["loss_after"] = mean_loss(model, data.train)
        return res


class FedFDPClient(ClientAlgorithm):
    """FedFDP local update (Ling et al., ACNS 2026, Algorithm 2 and Eqs. 12-15).

    sampling = "poisson" (paper): each round, draw one batch B with inclusion probability q, take `dp_steps`
    DP-SGD steps w <- w - eta/|B| (sum_j C_ij g_j + sigma C N(0, I)), with fair clip
    C_ij = min(1 + lambda (F(w; j) - F~(w_t)), C / ||g_j||). Then upload
    F~_i = (sum_j clip(F(w_new; j), 0, C_l) + sigma_l C_l N(0, 1)) / |B|, with the adaptive bound C_l = previous
    round's upload (C_l^0 given). sampling = "batches": epochs over fixed-size shuffled batches (for tests).
    """

    def fit(self, model, data, ins, state, gen):
        c = ins.config
        lam, C, sigma, sigma_l = float(c["lam"]), float(c["clip"]), float(c["sigma"]), float(c["sigma_loss"])
        f_glob = c.get("global_loss")
        set_params(model, ins.params)
        tr = data.train
        loss_before = mean_loss(model, tr)
        names = [n for n, _ in model.named_parameters()]
        lr = self.lr(ins, state)
        dev = model_device(model)

        def sample_loss(params, x, y):
            out = functional_call(model, dict(zip(names, params)), (x.unsqueeze(0),))
            return F.cross_entropy(out, y.unsqueeze(0))

        per_grad = vmap(grad(sample_loss), in_dims=(None, 0, 0))
        per_loss = vmap(sample_loss, in_dims=(None, 0, 0))
        model.train()
        params = [p.detach().clone() for p in model.parameters()]

        def dp_step(idx):
            X, y = tr.X[idx].to(dev), tr.y[idx].to(dev)
            grads = per_grad(tuple(params), X, y)
            flat = torch.cat([g.reshape(len(idx), -1) for g in grads], dim=1)
            norms = flat.norm(dim=1).clamp_min(1e-12)
            if f_glob is None:
                scale = torch.minimum(torch.ones_like(norms), C / norms)
            else:
                delta = per_loss(tuple(params), X, y).detach() - float(f_glob)
                scale = torch.minimum(1.0 + lam * delta, C / norms)
            noisy = (flat * scale[:, None]).sum(0) + sigma * C * torch.randn(flat.shape[1], generator=gen).to(dev)
            step = noisy / len(idx)
            off = 0
            for p in params:
                n = p.numel()
                p -= lr * step[off : off + n].view_as(p)
                off += n

        if c.get("sampling", "poisson") == "poisson":
            q = float(c["q"])
            batch = torch.nonzero(torch.rand(len(tr), generator=gen) < q).flatten()
            if len(batch) == 0:
                batch = torch.randint(len(tr), (1,), generator=gen)
            for _ in range(int(c.get("dp_steps", 1))):
                dp_step(batch)
        else:
            batch = torch.arange(len(tr))
            for _ in range(int(c.get("local_epochs", self.cfg.local_epochs))):
                for idx in batches(len(tr), self.cfg.batch_size, gen):
                    dp_step(idx)
        for p, new in zip(model.parameters(), params):
            p.data.copy_(new)
        with torch.no_grad():
            losses = per_loss(tuple(params), tr.X[batch].to(dev), tr.y[batch].to(dev))
        C_l = state.get("loss_clip", float(c.get("loss_clip_init", 2.5)))
        upload = float((losses.clamp(0, C_l).sum() + sigma_l * C_l * torch.randn((), generator=gen)) / len(losses))
        state["loss_clip"] = max(upload, 1e-3)  # Eq. 14 bound for the next round; the paper leaves <= 0 unhandled
        return FitRes(client_id=data.client_id, params=get_params(model), num_samples=len(tr),
                      metrics={"loss_before": loss_before, "loss_after": upload, "train_loss": float(losses.mean())})
