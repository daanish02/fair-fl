from __future__ import annotations

import math

import torch
from torch import nn

from fairfl.core.config import ModelConfig


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], num_classes: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """Two conv layers + linear head (the usual MNIST FL model)."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# Parameters travel as one flat vector (core/params.py), which carries parameters but not buffers.
# BatchNorm's running statistics are buffers, so the larger nets below use GroupNorm(2, C) instead,
# the usual substitute in FL (Hsieh et al., 2020).
def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(2, c)


def _flat_dim(features: nn.Module, input_shape: tuple[int, ...]) -> int:
    with torch.no_grad():
        return features(torch.zeros(1, *input_shape)).numel()


class ConvNet(nn.Module):
    def __init__(self, features: nn.Sequential, head: nn.Sequential):
        super().__init__()
        self.features, self.head = features, head

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def cnn_fmnist(input_shape: tuple[int, ...], num_classes: int) -> nn.Module:
    """FedCDA's CNN: conv5x5(32)-pool-conv5x5(64)-pool-fc512-out ('same' padding, as in FedAvg's MNIST CNN)."""
    f = nn.Sequential(
        nn.Conv2d(input_shape[0], 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
    )
    return ConvNet(f, nn.Sequential(nn.Linear(_flat_dim(f, input_shape), 512), nn.ReLU(), nn.Linear(512, num_classes)))


def cnn_cifar(input_shape: tuple[int, ...], num_classes: int) -> nn.Module:
    """CNNCifar from the FedMut repo (models/Nets.py): LeNet-5 style, conv(6)-pool-conv(16)-pool-120-84-out."""
    f = nn.Sequential(
        nn.Conv2d(input_shape[0], 6, 5), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
    )
    return ConvNet(f, nn.Sequential(nn.Linear(_flat_dim(f, input_shape), 120), nn.ReLU(),
                                    nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, num_classes)))


def cnn_celeba(input_shape: tuple[int, ...], num_classes: int) -> nn.Module:
    """LEAF-style CelebA CNN for 64x64 inputs: 4 x [conv3x3(32)-ReLU-pool], then a linear head (no BatchNorm)."""
    layers: list[nn.Module] = []
    c = input_shape[0]
    for _ in range(4):
        layers += [nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        c = 32
    f = nn.Sequential(*layers)
    return ConvNet(f, nn.Sequential(nn.Linear(_flat_dim(f, input_shape), num_classes)))


def vgg16(input_shape: tuple[int, ...], num_classes: int) -> nn.Module:
    """VGG16 for 32x32 inputs following the FedMut repo, with GroupNorm in place of BatchNorm."""
    layers: list[nn.Module] = []
    c = input_shape[0]
    for v in [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]:
        if v == "M":
            layers.append(nn.MaxPool2d(2))
        else:
            layers += [nn.Conv2d(c, v, 3, padding=1), _gn(v), nn.ReLU(inplace=True)]
            c = v
    f = nn.Sequential(*layers)
    return ConvNet(f, nn.Sequential(nn.Linear(_flat_dim(f, input_shape), 4096), nn.ReLU(True), nn.Dropout(),
                                    nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes)))


def resnet18(input_shape: tuple[int, ...], num_classes: int) -> nn.Module:
    """torchvision ResNet-18, CIFAR variant (3x3 stride-1 stem, no max-pool), GroupNorm in place of BatchNorm."""
    from torchvision.models import resnet18 as tv_resnet18

    m = tv_resnet18(num_classes=num_classes, norm_layer=_gn)
    m.conv1 = nn.Conv2d(input_shape[0], 64, 3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


class Logistic(nn.Module):
    """Sigmoid logistic regression (LoGoFair code) exposed as two logits [0, z]: softmax(.)[1] = sigmoid(z).
    Parameter order (weight, bias) matches the official code's flat vector, so its checkpoints load directly."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.layer = nn.Linear(in_dim, 1)

    def forward(self, x):
        z = self.layer(x.flatten(1))
        return torch.cat([torch.zeros_like(z), z], dim=1)


class CNNMnist(nn.Module):
    """CNNMnist from the FCFL repo (models/Nets.py): conv10-pool-conv20-dropout2d-pool-fc50-dropout-out."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.drop2d = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.drop2d(self.conv2(x)), 2))
        x = torch.relu(self.fc1(x.flatten(1)))
        return self.fc2(nn.functional.dropout(x, training=self.training))


class DropoutMLP(nn.Module):
    """MLP from the FCFL repo: one hidden layer with dropout before the ReLU."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, hidden), nn.Dropout(), nn.ReLU(),
                                 nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.net(x)


BUILDERS = {"cnn_fmnist": cnn_fmnist, "cnn_cifar": cnn_cifar, "cnn_celeba": cnn_celeba, "vgg16": vgg16,
            "resnet18": resnet18}


def build_model(cfg: ModelConfig, input_shape: tuple[int, ...], num_classes: int) -> nn.Module:
    if cfg.kind in BUILDERS:
        if len(input_shape) != 3:
            raise ValueError(f"model {cfg.kind!r} needs image data, got input shape {input_shape}")
        return BUILDERS[cfg.kind](input_shape, num_classes)
    if cfg.kind == "logistic":
        return Logistic(math.prod(input_shape))
    if cfg.kind == "cnn_mnist":
        return CNNMnist(input_shape[0], num_classes)
    if cfg.kind == "mlp_dropout":
        return DropoutMLP(math.prod(input_shape), cfg.hidden[0] if cfg.hidden else 64, num_classes)
    if cfg.kind == "cnn":
        return SmallCNN(input_shape[0], num_classes)
    hidden = [] if cfg.kind == "logreg" else cfg.hidden
    return MLP(math.prod(input_shape), hidden, num_classes)
