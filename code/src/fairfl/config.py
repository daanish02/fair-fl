from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def _set_dotted(d: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


class DatasetConfig(BaseModel):
    """Meant to stay mostly fixed across runs (CIFAR10, Dirichlet-partitioned)."""

    data_root: str = "data"
    num_clients: int = Field(default=10, ge=0, description="0 = centralised/server-only")
    alpha: float = Field(default=0.5, gt=0, description="Dirichlet concentration; lower = more non-IID")
    seed: int = Field(default=0, description="partition seed")


class ModelConfig(BaseModel):
    name: str = Field(default="resnet18", description="see fairfl.models for registered names")


class TrainConfig(BaseModel):
    rounds: int = Field(default=30, ge=1, description="communication rounds (FL) or epochs (centralised)")
    clients_per_round: float = Field(default=1.0, gt=0, le=1.0, description="fraction of clients sampled each round")
    local_epochs: int = Field(default=5, ge=1, description="local epochs per client per round (FL only)")
    lr: float = Field(default=0.1, gt=0)
    momentum: float = Field(default=0.9, ge=0)
    weight_decay: float = Field(default=5e-4, ge=0)
    batch_size: int = Field(default=64, ge=1)
    device: str = Field(default="auto", description="'auto', 'cpu', or 'cuda'")
    log_every: int = Field(default=50, ge=0, description="log every N batches within an epoch; 0 = off")


class StrategyConfig(BaseModel):
    name: str = Field(default="fedavg", description="see fairfl.registry for registered names")
    params: dict[str, Any] = Field(default_factory=dict, description="strategy-specific hyperparameters")


class RunConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)

    @property
    def is_centralised(self) -> bool:
        return self.dataset.num_clients == 0

    @classmethod
    def from_yaml_and_overrides(cls, path: str | Path | None, overrides: list[str]) -> "RunConfig":
        import yaml

        raw: dict[str, Any] = (yaml.safe_load(Path(path).read_text()) if path else None) or {}
        for kv in overrides:
            key, _, val = kv.partition("=")
            _set_dotted(raw, key.strip(), yaml.safe_load(val.strip()))
        return cls.model_validate(raw)
