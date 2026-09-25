from pathlib import Path

import numpy as np
import pytest
import torch

from fairfl.core.config import DataConfig, ExperimentConfig, ModelConfig, StrategySpec, TrainConfig
from fairfl.core.engine import Simulator
from fairfl.data.datasets import load_celeba, load_dataset, random_crop_flip
from fairfl.data.partition import shard_partition
from fairfl.models import build_model


def make_cfg(strategy: str = "fedavg", params: dict | None = None, rounds: int = 3, **train) -> ExperimentConfig:
    return ExperimentConfig(
        seed=1,
        data=DataConfig(name="synthetic", num_clients=4, alpha=1.0, partition_on="sensitive"),
        model=ModelConfig(kind="mlp", hidden=[8]),
        train=TrainConfig(rounds=rounds, clients_per_round=1.0, lr=0.1, **train),
        strategy=StrategySpec(name=strategy, params=params or {}),
    )


def run(cfg: ExperimentConfig) -> Simulator:
    sim = Simulator(cfg)
    sim.run()
    return sim


def test_shards_partition_gives_each_client_two_classes():
    labels = np.repeat(np.arange(10), 100)
    parts = shard_partition(labels, 5, 2, np.random.default_rng(0))
    allidx = np.concatenate(parts)
    assert len(allidx) == len(labels) and len(np.unique(allidx)) == len(labels)
    assert all(len(np.unique(labels[p])) <= 2 for p in parts)
    assert all(len(p) == 200 for p in parts)


def test_shards_partition_via_config():
    cfg = DataConfig(name="synthetic", num_clients=4, partition_on="shards", shards_per_client=1)
    from fairfl.data.federated import build_federated

    fed = build_federated(cfg, 0)
    assert sum(len(c.train) + len(c.test) for c in fed.clients) == 4000
    assert sum(len(torch.cat([c.train.y, c.test.y]).unique()) == 1 for c in fed.clients) >= 2


@pytest.mark.parametrize("kind,shape,classes", [
    ("cnn_fmnist", (1, 28, 28), 10), ("cnn_cifar", (3, 32, 32), 10), ("resnet18", (3, 32, 32), 100),
    ("vgg16", (3, 32, 32), 10), ("cnn_celeba", (3, 64, 64), 2),
])
def test_new_models_forward_and_have_no_buffers(kind, shape, classes):
    m = build_model(ModelConfig(kind=kind), shape, classes)
    m.eval()
    assert m(torch.zeros(2, *shape)).shape == (2, classes)
    assert not list(m.buffers()), "buffers would not travel in the flat params vector"


def test_random_crop_flip_keeps_shape_and_content():
    X = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).view(2, 3, 4, 4)
    gen = torch.Generator().manual_seed(0)
    assert random_crop_flip(X, gen).shape == X.shape
    out = random_crop_flip(X, gen, pad=0)
    for i in range(2):
        assert torch.equal(out[i], X[i]) or torch.equal(out[i], X[i].flip(2))


def test_eval_every_logs_only_evaluated_and_final_rounds():
    sim = run(make_cfg(rounds=5, eval_every=2))
    assert [log.round for log in sim.logs] == [1, 3, 4]
    torch.testing.assert_close(sim.final_params, run(make_cfg(rounds=5)).final_params, rtol=0, atol=0)


def test_zero_momentum_options_are_bit_identical():
    base = run(make_cfg()).final_params
    explicit = run(make_cfg(momentum=0.0, weight_decay=0.0, server_momentum=0.0, eval_every=1)).final_params
    assert torch.equal(base, explicit)
    assert not torch.equal(base, run(make_cfg(momentum=0.9)).final_params)
    assert not torch.equal(base, run(make_cfg(server_momentum=0.5)).final_params)


@pytest.mark.parametrize("strategy,params", [
    ("fedavg", {}), ("median", {"sigma": 0.01}), ("fairrfl", {}), ("fairfed", {}), ("fcfl", {}),
    ("fedmut", {}), ("fedfdp", {"sigma": 0.1}), ("fairweight", {"repeats": 1, "max_samples": 20}),
])
def test_explicit_cpu_device_keeps_every_tensor_on_model_device(strategy, params):
    cfg = make_cfg(strategy, params, rounds=2, server_momentum=0.5).model_copy(update={"device": "cpu"})
    sim = Simulator(cfg)
    dev = next(sim.model.parameters()).device
    assert dev == torch.device("cpu") and sim.data_on_device
    seen = []
    agg = sim.strategy.aggregate

    def checked(rnd, gparams, results):
        seen.extend([gparams.device] + [r.params.device for r in results])
        out = agg(rnd, gparams, results)
        seen.append(out.device)
        return out

    sim.strategy.aggregate = checked
    sim.run()
    assert seen and all(d == dev for d in seen)
    for c in sim.data.clients:
        assert all(t.device == dev for t in (c.train.X, c.train.y, c.train.a, c.test.X))


def test_tabular_loader_if_csv_present():
    if not Path("data/bank.csv").exists():
        pytest.skip("data/bank.csv not present")
    d = load_dataset("bank", "data")
    assert d.X.shape[1] == 16 and set(d.a.unique().tolist()) == {0, 1} and set(d.y.unique().tolist()) == {0, 1}


def test_tabular_loader_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="fairweight.zip"):
        load_dataset("law", tmp_path)


def test_celeba_loader_from_local_folder(tmp_path):
    from PIL import Image

    base = tmp_path / "celeba"
    (base / "img_align_celeba").mkdir(parents=True)
    rows = []
    for i in range(3):
        name = f"{i:06d}.jpg"
        Image.new("RGB", (178, 218), (40 * i, 0, 0)).save(base / "img_align_celeba" / name)
        rows.append(f"{name} {1 if i else -1} {-1 if i == 2 else 1} -1")
    (base / "list_attr_celeba.txt").write_text("3\nMale Smiling Young\n" + "\n".join(rows) + "\n")
    d = load_celeba(tmp_path)
    assert d.X.shape == (3, 3, 64, 64)
    assert d.y.tolist() == [1, 1, 0] and d.a.tolist() == [0, 1, 1]
    assert load_celeba(tmp_path, target_attr="Young").y.tolist() == [0, 0, 0]
