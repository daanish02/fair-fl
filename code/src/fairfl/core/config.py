from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from fairfl.fairness.scheme import FairnessScheme


class DataConfig(BaseModel):
    name: Literal["adult", "mnist", "synthetic", "split_file", "fmnist", "cifar10", "cifar100", "celeba",
                  "bank", "default", "law", "kdd"] = "adult"
    num_clients: int = Field(10, ge=2)
    alpha: float = Field(0.5, gt=0.0, description="Dirichlet concentration; smaller = more heterogeneous.")
    partition_on: Literal["label", "sensitive", "joint", "iid", "shards"] = "sensitive"
    shards_per_client: int = Field(2, ge=1, description="partition_on=shards: label-sorted shards per client.")
    source: Literal["train+test", "train"] = Field("train+test", description="Image datasets: pool the official "
                                                   "train and test sets, or use the train set only (as FCFL does).")
    test_fraction: float = Field(0.2, ge=0.0, lt=1.0, description="0 = no local test split (all client data "
                                 "trains; use data.global_test for evaluation).")
    partition_file: str | None = Field(None, description="JSON {train_data: {client: [indices]}} fixing the "
                                       "client partition (e.g. FedMut's shipped Dirichlet splits).")
    normalize: Literal["dataset", "half"] = Field("dataset", description="Image normalisation: per-dataset "
                                                  "mean/std, or 0.5/0.5 (FedMut CIFAR-10 code).")
    split_file: str | None = Field(None, description="name=split_file: JSON {users, user_data: {id: {x, y, A}}} "
                                   "with a fixed client partition (e.g. LoGoFair's shipped Adult split).")
    split_seed: int | None = Field(None, description="Legacy np.random seed for per-client permutations "
                                   "(LoGoFair code: np.random.seed(1 + 111)); None = experiment rng.")
    val_from_train: bool = Field(False, description="LoGoFair code: validation set = the whole local train split.")
    client_split: Literal["shuffle", "ordered"] = Field("shuffle", description="ordered = FCFL code: no shuffling; "
                                                        "local train / val / test are consecutive slices.")
    global_test: Literal["none", "official"] = Field("none", description="official = also evaluate the global "
                                                     "model on the dataset's official test set (needs source=train).")
    val_fraction: float = Field(0.0, ge=0.0, lt=1.0, description="Share of each client's training data held out "
                                                                 "for post-processing/calibration (LoGoFair).")
    min_samples: int = Field(20, ge=1)
    root: str = "data"
    augment: bool = Field(False, description="Random crop (pad 4) + horizontal flip per training batch; images only.")
    target_attr: str = Field("Smiling", description="CelebA label attribute.")
    sensitive_attr: str = Field("Male", description="CelebA sensitive attribute.")
    subsample: int = Field(0, ge=0, description="CelebA: keep this many random images (0 = all ~202k).")


class ModelConfig(BaseModel):
    kind: Literal["logistic", "mlp", "mlp_dropout", "cnn", "cnn_mnist", "logreg", "cnn_fmnist", "cnn_cifar", "resnet18", "vgg16", "cnn_celeba"] = "mlp"
    hidden: list[int] = Field(default_factory=lambda: [32])
    init_from: str | None = Field(None, description="Load initial parameters (flat tensor or state_dict) from file.")


class TrainConfig(BaseModel):
    rounds: int = Field(50, ge=0, description="0 = no training (e.g. post-process a loaded model).")
    clients_per_round: float = Field(1.0, gt=0.0, le=1.0, description="Fraction of clients sampled per round.")
    local_epochs: int = Field(1, ge=1)
    batch_size: int = Field(64, ge=1)
    lr: float = Field(0.05, gt=0.0)
    momentum: float = Field(0.0, ge=0.0, lt=1.0, description="Client SGD momentum (reset every round).")
    weight_decay: float = Field(0.0, ge=0.0, description="Client SGD weight decay.")
    server_momentum: float = Field(0.0, ge=0.0, lt=1.0, description="FedAvgM momentum on the global update, "
                                                                     "applied after strategy.aggregate.")
    checkpoint_every: int = Field(25, ge=0, description="Save a resumable checkpoint every k rounds "
                                                         "(0 = off). A rerun resumes from it.")
    eval_every: int = Field(1, ge=1, description="Evaluate and log every k-th round (plus the final and "
                                                 "post-processed models). k > 1 coarsens the fairness scopes.")


class StrategySpec(BaseModel):
    name: str = "fedavg"
    params: dict[str, Any] = Field(default_factory=dict)


class ScenarioSpec(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    name: str = "experiment"
    seed: int = 0
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    strategy: StrategySpec = Field(default_factory=StrategySpec)
    fairness: FairnessScheme = Field(default_factory=FairnessScheme)
    scenarios: list[ScenarioSpec] = Field(default_factory=list)
    output_dir: str = "runs"
    device: Literal["auto", "cpu", "cuda"] = Field("auto", description="auto = cuda if available, else cpu.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f) or {})
