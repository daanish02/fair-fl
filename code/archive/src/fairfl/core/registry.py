from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from fairfl.strategies.base import Strategy

_STRATEGIES: dict[str, type[Strategy]] = {}

S = TypeVar("S", bound="type[Strategy]")


def register_strategy(name: str):
    def deco(cls: S) -> S:
        if name in _STRATEGIES:
            raise ValueError(f"strategy {name!r} registered twice")
        cls.name = name
        _STRATEGIES[name] = cls
        return cls

    return deco


def get_strategy(name: str) -> type[Strategy]:
    import fairfl.strategies  # noqa: F401  (populates the registry)

    try:
        return _STRATEGIES[name]
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(_STRATEGIES)}") from None


def list_strategies() -> list[str]:
    import fairfl.strategies  # noqa: F401

    return sorted(_STRATEGIES)
