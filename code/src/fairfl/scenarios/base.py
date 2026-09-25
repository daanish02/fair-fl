from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from fairfl.core.types import FitRes
from fairfl.data.federated import FederatedData

_SCENARIOS: dict[str, type[Scenario]] = {}


def register_scenario(name: str):
    def deco(cls):
        cls.name = name
        _SCENARIOS[name] = cls
        return cls

    return deco


class Scenario:
    """Perturbs the simulation: data changes before a round, or client messages after local training."""

    name: ClassVar[str] = "base"
    Params: ClassVar[type[BaseModel]] = BaseModel

    def __init__(self, params: BaseModel, seed: int):
        self.p = params
        self.seed = seed

    def before_round(self, rnd: int, data: FederatedData) -> None: ...

    def after_fit(self, rnd: int, res: FitRes, global_params) -> FitRes:
        return res


def build_scenario(name: str, raw: dict[str, Any], seed: int) -> Scenario:
    import fairfl.scenarios  # noqa: F401

    cls = _SCENARIOS[name]
    return cls(cls.Params.model_validate(raw), seed)
