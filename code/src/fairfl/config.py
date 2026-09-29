from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    """Hyperparameters for a training run. num_clients=0 means centralised (no FL, single model)."""

    num_clients: int = Field(default=10, ge=0, description="0 = centralised/server-only")
    rounds: int = Field(default=30, ge=1, description="communication rounds (FL) or epochs (centralised)")
    local_epochs: int = Field(default=5, ge=1, description="local epochs per client per round (FL only)")
    alpha: float = Field(default=0.5, gt=0, description="Dirichlet concentration; lower = more non-IID")
    lr: float = Field(default=0.1, gt=0)
    weight_decay: float = Field(default=5e-4, ge=0)
    batch_size: int = Field(default=64, ge=1)
    seed: int = Field(default=0)
    data_root: str = Field(default="data")
    device: str = Field(default="auto", description="'auto', 'cpu', or 'cuda'")
    log_every: int = Field(default=50, ge=0, description="log every N batches within an epoch; 0 = off")

    @property
    def is_centralised(self) -> bool:
        return self.num_clients == 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
