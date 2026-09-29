import torch

from fairfl.core.config import DataConfig, ExperimentConfig, ModelConfig, StrategySpec, TrainConfig
from fairfl.core.engine import Simulator


def cfg(rounds: int) -> ExperimentConfig:
    return ExperimentConfig(seed=3, data=DataConfig(name="synthetic", num_clients=6),
                            model=ModelConfig(kind="mlp", hidden=[8]),
                            train=TrainConfig(rounds=rounds, clients_per_round=0.5, checkpoint_every=3),
                            strategy=StrategySpec(name="fcfl", params={"eta": 0.3, "random_fraction": 0.4}),
                            device="cpu")


def test_resume_matches_uninterrupted_run(tmp_path):
    straight = Simulator(cfg(6))
    straight.run()
    ckpt = tmp_path / "ckpt.pt"
    Simulator(cfg(3)).run(checkpoint=ckpt)       # "crash" after round 3; checkpoint saved at round 3
    assert ckpt.exists()
    resumed = Simulator(cfg(6))
    logs = resumed.run(checkpoint=ckpt)
    torch.testing.assert_close(resumed.final_params, straight.final_params)
    assert [log.round for log in logs] == [log.round for log in straight.logs]


def test_run_stops_when_loss_diverges():
    sim = Simulator(cfg(6))
    orig = sim.strategy.aggregate

    def blow_up(rnd, gparams, results):  # NaN weights from round 2 on
        out = orig(rnd, gparams, results)
        return out * float("nan") if rnd >= 1 else out

    sim.strategy.aggregate = blow_up
    logs = sim.run()
    assert sim.diverged_at == 2
    assert len(logs) == 2
