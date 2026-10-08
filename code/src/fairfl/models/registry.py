"""Model registry: maps model name -> builder. Populated by importing models/."""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

BUILDERS: dict[str, Callable[[int], nn.Module]] = {}


def register_model(name: str):
    def deco(fn: Callable[[int], nn.Module]) -> Callable[[int], nn.Module]:
        BUILDERS[name] = fn
        return fn

    return deco


def get_model(name: str) -> Callable[[int], nn.Module]:
    import fairfl.models  # noqa: F401  (populates the registry)

    try:
        return BUILDERS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; available: {sorted(BUILDERS)}") from None


def list_models() -> list[str]:
    import fairfl.models  # noqa: F401

    return sorted(BUILDERS)
