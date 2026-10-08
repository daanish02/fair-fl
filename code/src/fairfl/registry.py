"""Strategy registry: maps strategy name -> Strategy class. Populated by importing strategies/."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fairfl.strategies.base import Strategy

STRATEGIES: dict[str, "type[Strategy]"] = {}


def register_strategy(name: str):
    def deco(cls: "type[Strategy]") -> "type[Strategy]":
        cls.name = name
        STRATEGIES[name] = cls
        return cls

    return deco


def get_strategy(name: str) -> "type[Strategy]":
    import fairfl.strategies  # noqa: F401  (populates the registry; also sets Strategy.client_cls default)

    try:
        return STRATEGIES[name]
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(STRATEGIES)}") from None


def list_strategies() -> list[str]:
    import fairfl.strategies  # noqa: F401

    return sorted(STRATEGIES)
