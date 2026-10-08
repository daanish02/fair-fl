from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18

from fairfl.models.registry import register_model


@register_model("resnet18")
def build_resnet18(num_classes: int = 10) -> nn.Module:
    """ResNet18 adapted for 32x32 CIFAR input: 3x3 stem stride 1, no initial maxpool."""
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
