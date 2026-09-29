import json

import numpy as np

from fairfl.core.config import DataConfig, ExperimentConfig, ModelConfig, StrategySpec, TrainConfig
from fairfl.core.engine import Simulator
from fairfl.experiments.runner import summarise_run


def test_partition_file_and_no_local_test(tmp_path):
    idx = np.random.default_rng(0).permutation(4000)
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"train_data": {str(i): idx[i * 1000:(i + 1) * 1000].tolist() for i in range(4)}}))
    cfg = ExperimentConfig(data=DataConfig(name="synthetic", num_clients=4, partition_file=str(f), test_fraction=0.0),
                           model=ModelConfig(kind="mlp", hidden=[8]), train=TrainConfig(rounds=2), device="cpu")
    sim = Simulator(cfg)
    assert [len(c.train) for c in sim.data.clients] == [1000] * 4
    assert all(len(c.test) == 0 for c in sim.data.clients)
    logs = sim.run()
    s = summarise_run(logs, cfg.fairness)
    assert s["final_worst_client_acc"] is None and np.isnan(logs[-1].global_accuracy)


def test_fedcda_uses_b_batches():
    cfg = ExperimentConfig(data=DataConfig(name="synthetic", num_clients=6), model=ModelConfig(kind="mlp", hidden=[8]),
                           train=TrainConfig(rounds=3, clients_per_round=0.7),
                           strategy=StrategySpec(name="fedcda", params={"warmup": 1, "num_batches": 3}), device="cpu")
    Simulator(cfg).run()
