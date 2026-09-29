from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


@dataclass
class TensorDataset:
    X: torch.Tensor
    y: torch.Tensor

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> "TensorDataset":
        i = torch.as_tensor(idx, dtype=torch.long)
        return TensorDataset(self.X[i], self.y[i])

    def to(self, device: torch.device | str) -> "TensorDataset":
        return TensorDataset(self.X.to(device), self.y.to(device))


def load_cifar10(root: str | Path, split: str) -> TensorDataset:
    from torchvision import datasets

    ds = datasets.CIFAR10(str(root), train=(split == "train"), download=True)
    data = torch.as_tensor(ds.data).permute(0, 3, 1, 2).float().div(255.0)
    m = torch.tensor(CIFAR10_MEAN).view(1, -1, 1, 1)
    s = torch.tensor(CIFAR10_STD).view(1, -1, 1, 1)
    X = (data - m) / s
    y = torch.as_tensor(ds.targets).long()
    return TensorDataset(X.contiguous(), y)


def random_crop_flip(X: torch.Tensor, gen: torch.Generator, pad: int = 4) -> torch.Tensor:
    """Per-sample random crop (zero padding) and horizontal flip, on X's device."""
    n, _, h, w = X.shape
    off = torch.randint(0, 2 * pad + 1, (2, n), generator=gen).to(X.device)
    flip = (torch.rand(n, generator=gen) < 0.5).to(X.device)
    Xp = torch.nn.functional.pad(X, (pad, pad, pad, pad)).permute(0, 2, 3, 1)
    rows = (off[0][:, None] + torch.arange(h, device=X.device))[:, :, None]
    cols = (off[1][:, None] + torch.arange(w, device=X.device))[:, None, :]
    out = Xp[torch.arange(n, device=X.device)[:, None, None], rows, cols].permute(0, 3, 1, 2)
    return torch.where(flip[:, None, None, None], out.flip(3), out).contiguous()
