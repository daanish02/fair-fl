import random

import numpy as np
import pytest
import torch

from fairfl.core.config import DataConfig, ExperimentConfig, ModelConfig, StrategySpec, TrainConfig
from fairfl.core.engine import Simulator
from fairfl.core.types import FitRes
from fairfl.strategies.fedmut import mutation_signs
from fairfl.strategies.median import Median, MedianParams
from fairfl.strategies.qffl import qffl_step


def final_params(strategy: str, params: dict | None = None, rounds: int = 3) -> torch.Tensor:
    cfg = ExperimentConfig(
        seed=1,
        data=DataConfig(name="synthetic", num_clients=4, alpha=1.0, partition_on="sensitive"),
        model=ModelConfig(kind="mlp", hidden=[8]),
        train=TrainConfig(rounds=rounds, clients_per_round=1.0, lr=0.1),
        strategy=StrategySpec(name=strategy, params=params or {}),
    )
    sim = Simulator(cfg)
    sim.run()
    return sim.final_params


@pytest.fixture(scope="module")
def fedavg_params():
    return final_params("fedavg")


def test_fcfl_eta_zero_is_fedavg(fedavg_params):
    torch.testing.assert_close(final_params("fcfl", {"eta": 0.0}), fedavg_params)


def test_fairfed_beta_zero_without_reweight_is_fedavg(fedavg_params):
    torch.testing.assert_close(final_params("fairfed", {"beta": 0.0, "local_reweight": False}), fedavg_params)


def test_fcfl_eta_positive_differs(fedavg_params):
    assert not torch.allclose(final_params("fcfl", {"eta": 5.0}, rounds=4), final_params("fedavg", rounds=4))


def test_fedmut_signs_pair_up():
    signs = mutation_signs(num_layers=3, m=6, ctrl_rate=0.0, rnd=random.Random(0))
    for layer in signs:
        assert sorted(layer) == [-1.0] * 3 + [1.0] * 3
        assert sum(layer) == pytest.approx(0.0)


def test_fedmut_acceleration_skews_signs():
    layer = mutation_signs(1, 4, ctrl_rate=0.3, rnd=random.Random(0))[0]
    assert sorted(layer) == pytest.approx([-0.7, -0.7, 1.0, 1.0])


def res(cid, params, loss=1.0):
    return FitRes(client_id=cid, params=torch.tensor(params, dtype=torch.float32), num_samples=10,
                  metrics={"loss_before": loss})


def test_median_is_coordinate_wise():
    s = Median(MedianParams(), TrainConfig(), np.random.default_rng(0))
    g = torch.zeros(2)
    out = s.aggregate(0, g, [res(0, [1, 100]), res(1, [2, -5]), res(2, [3, 0])])
    torch.testing.assert_close(out, torch.tensor([2.0, 0.0]))


def test_qffl_q_zero_equals_mean_of_updates():
    g = torch.zeros(2)
    rs = [res(0, [1, 0], loss=0.5), res(1, [3, 2], loss=2.0)]
    torch.testing.assert_close(qffl_step(g, rs, [0.0, 0.0], lr=0.1), torch.tensor([2.0, 1.0]))


def test_qffl_upweights_high_loss_clients():
    g = torch.zeros(1)
    rs = [res(0, [1.0], loss=0.1), res(1, [-1.0], loss=2.0)]
    assert qffl_step(g, rs, [2.0, 2.0], lr=0.1).item() < 0


# FairRFL (TETC 2026) Examples 4-5. The selfish update s_hat is recovered from the paper's reported numbers:
# s_hat = (Delta_s - (1 - beta) * Delta_med) / beta = ([0.52, 0.96] - 0.55 * [-0.20, 0.55]) / 0.45.
FAIRRFL_UPDATES = torch.tensor([[0.95, 0.55], [-0.20, 0.90], [-0.60, 0.55], [-1.20, 0.10], [1.40, 1.46]])


def test_fairrfl_detection_matches_example_4():
    from fairfl.strategies.fairrfl import detect_selfish

    flagged, n_med = detect_selfish(FAIRRFL_UPDATES, tau=2.5)
    assert flagged == [4]
    assert n_med == pytest.approx(1.1, abs=0.01)


def test_fairrfl_recovery_matches_example_5():
    from fairfl.strategies.fairrfl import recover

    med = torch.from_numpy(np.median(FAIRRFL_UPDATES.numpy(), axis=0))
    torch.testing.assert_close(med, torch.tensor([-0.20, 0.55]))
    rec, beta = recover(FAIRRFL_UPDATES[4], med, n_med=1.1)
    assert beta == pytest.approx(0.45, abs=0.01)
    torch.testing.assert_close(rec, torch.tensor([0.52, 0.96]), atol=0.02, rtol=0)


def test_fairrfl_and_fedcda_run():
    final_params("fairrfl", rounds=2)
    final_params("fedcda", {"warmup": 1}, rounds=3)


def test_fairweight_paper_toy_scores():
    """Section IV-3 example: 3 clients, 4 weights; masks w1=[0,0,0,1], w2=[0,1,0,1], w3=[0,1,1,1]."""
    from fairfl.strategies.fairweight import coordinate_scores

    masks = [{3}, {1, 3}, {1, 2, 3}]
    raw = coordinate_scores(masks, [0, 1, 2, 3], gamma1=1.0, gamma2=1.0)
    # Individual scores [1,1,1,0], [1,.33,1,0], [1,.33,.66,0] normalised per coordinate
    # -> paper's averaged weightage w1=[.33,.6,.37,0], w2=[.33,.19,.37,0], w3=[.33,.19,.24,0].
    np.testing.assert_allclose(raw, [[1 / 3, 0.6, 0.375, 0], [1 / 3, 0.2, 0.375, 0], [1 / 3, 0.2, 0.25, 0]], atol=0.01)


def test_fairweight_runs():
    final_params("fairweight", {"repeats": 1, "max_samples": 40}, rounds=2)


def test_fedfair_lambda_zero_is_fedavg(fedavg_params):
    torch.testing.assert_close(final_params("fedfair", {"lam": 0.0}), fedavg_params)


def test_fedfdp_runs_and_learns():
    p = final_params("fedfdp", {"sigma": 0.5, "clip": 1.0, "q": 0.2}, rounds=3)
    assert torch.isfinite(p).all()


def test_fedfdp_without_noise_or_clipping_is_fedavg(fedavg_params):
    p = final_params("fedfdp", {"lam": 0.0, "sigma": 0.0, "sigma_loss": 0.0, "clip": 1e9, "sampling": "batches"})
    torch.testing.assert_close(p, fedavg_params, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("mode", ["g", "lg"])
def test_logofair_reduces_global_dp_gap(mode):
    from fairfl.metrics.group import dp_gap, total

    cfg = ExperimentConfig(
        seed=0,
        data=DataConfig(name="synthetic", num_clients=5, alpha=0.5, partition_on="sensitive", val_fraction=0.3),
        model=ModelConfig(kind="mlp", hidden=[16]),
        train=TrainConfig(rounds=10, lr=0.1),
        strategy=StrategySpec(name="logofair", params={"mode": mode}),
    )
    logs = Simulator(cfg).run()
    gap = lambda log: dp_gap(total([c.groups for c in log.clients]))
    assert gap(logs[-1]) < 0.5 * gap(logs[-2])
