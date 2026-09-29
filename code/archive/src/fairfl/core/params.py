"""Model parameters travel as one flat float tensor; these helpers move them in and out of a module."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

Params = torch.Tensor


def get_params(model: nn.Module) -> Params:
    return parameters_to_vector(model.parameters()).detach().clone()


def set_params(model: nn.Module, params: Params) -> None:
    vector_to_parameters(params.detach().clone(), model.parameters())


def weighted_mean(vectors: list[Params], weights: list[float]) -> Params:
    w = torch.tensor(weights, dtype=vectors[0].dtype, device=vectors[0].device)
    w = w / w.sum()
    return (torch.stack(vectors) * w[:, None]).sum(dim=0)
