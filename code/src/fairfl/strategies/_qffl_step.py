"""q-FedAvg server update (Li et al., ICLR 2020), shared by fairrfl's Dq-FFL weighting. Not a selectable strategy."""

from __future__ import annotations

from fairfl.strategies.base import FitRes


def qffl_step(global_state: dict, results: list[FitRes], qs: list[float], lr: float) -> dict:
    L = 1.0 / lr
    out = {}
    for key, g in global_state.items():
        if not g.is_floating_point():
            out[key] = results[0].state[key]
            continue
        num = None
        den = 0.0
        for r, q in zip(results, qs):
            F_i = max(float(r.metrics["loss_before"]), 1e-10)
            dw = L * (g - r.state[key])
            term = (F_i ** q) * dw
            num = term if num is None else num + term
            den += q * F_i ** (q - 1) * float(dw.pow(2).sum()) + L * F_i ** q
        out[key] = g - num / den
    return out
