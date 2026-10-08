from __future__ import annotations

import torch.nn as nn

from fairfl.models.registry import register_model


@register_model("cnn_cifar")
def build_cnn_cifar(num_classes: int = 10) -> nn.Module:
    """LeNet-5-style CNN for 32x32 CIFAR input (FedMut repo's CNNCifar): conv(6)-pool-conv(16)-pool-120-84-out."""
    return nn.Sequential(
        nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, num_classes),
    )
