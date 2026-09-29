"""Centralised datasets as (X, y, a) tensors; `a` is the binary sensitive attribute (all zeros if none)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TensorDataset3:
    X: torch.Tensor
    y: torch.Tensor
    a: torch.Tensor
    num_classes: int
    num_groups: int

    def __len__(self) -> int:
        return len(self.y)

    augment: bool = False

    def subset(self, idx: np.ndarray) -> TensorDataset3:
        i = torch.as_tensor(idx, dtype=torch.long)
        return TensorDataset3(self.X[i], self.y[i], self.a[i], self.num_classes, self.num_groups, self.augment)

    def to(self, device: torch.device | str) -> TensorDataset3:
        return TensorDataset3(self.X.to(device), self.y.to(device), self.a.to(device), self.num_classes,
                              self.num_groups, self.augment)

    def nbytes(self) -> int:
        return sum(t.element_size() * t.numel() for t in (self.X, self.y, self.a))


def random_crop_flip(X: torch.Tensor, gen: torch.Generator, pad: int = 4) -> torch.Tensor:
    """Per-sample random crop (zero padding) and horizontal flip of a batch [N, C, H, W], on X's device.
    Offsets come from the CPU generator, so runs stay reproducible across devices."""
    n, _, h, w = X.shape
    off = torch.randint(0, 2 * pad + 1, (2, n), generator=gen).to(X.device)
    flip = (torch.rand(n, generator=gen) < 0.5).to(X.device)
    Xp = torch.nn.functional.pad(X, (pad, pad, pad, pad)).permute(0, 2, 3, 1)
    rows = (off[0][:, None] + torch.arange(h, device=X.device))[:, :, None]
    cols = (off[1][:, None] + torch.arange(w, device=X.device))[:, None, :]
    out = Xp[torch.arange(n, device=X.device)[:, None, None], rows, cols].permute(0, 3, 1, 2)
    return torch.where(flip[:, None, None, None], out.flip(3), out).contiguous()


def load_adult(root: str | Path) -> TensorDataset3:
    """UCI Adult; label income >50K, sensitive attribute sex (1 = male)."""
    import pandas as pd
    from sklearn.datasets import fetch_openml

    df = fetch_openml("adult", version=2, as_frame=True, data_home=str(root)).frame
    df = df.replace("?", np.nan).dropna().reset_index(drop=True)
    y = (df.pop("class").astype(str).str.strip().str.startswith(">50K")).astype(np.int64).to_numpy().copy()
    a = (df["sex"].astype(str).str.strip() == "Male").astype(np.int64).to_numpy().copy()
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    num = df[num_cols].astype(np.float32)
    num = (num - num.mean()) / (num.std() + 1e-8)
    cat = pd.get_dummies(df[cat_cols].astype(str), dtype=np.float32)
    X = np.concatenate([num.to_numpy(), cat.to_numpy()], axis=1).astype(np.float32)
    return TensorDataset3(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(a), 2, 2)


_SPLITS = {"all": (True, False), "train": (True,), "test": (False,)}


def load_mnist(root: str | Path, split: str = "all") -> TensorDataset3:
    from torchvision import datasets

    xs, ys = [], []
    for train in _SPLITS[split]:
        ds = datasets.MNIST(str(root), train=train, download=True)
        xs.append(ds.data)
        ys.append(ds.targets)
    X = (torch.cat(xs).float().div(255.0).sub(0.1307).div(0.3081)).unsqueeze(1)
    y = torch.cat(ys).long()
    return TensorDataset3(X, y, torch.zeros_like(y), 10, 1)


def _load_torchvision(cls_name: str, root: str | Path, mean: list[float], std: list[float], split: str = "all"):
    """Train and test merged (or the official train set only), scaled to [0, 1] and normalised per channel.
    Clients then split their shard into local train/test."""
    from torchvision import datasets

    xs, ys = [], []
    for train in _SPLITS[split]:
        ds = getattr(datasets, cls_name)(str(root), train=train, download=True)
        data = torch.as_tensor(ds.data)
        xs.append(data.unsqueeze(1) if data.dim() == 3 else data.permute(0, 3, 1, 2))
        ys.append(torch.as_tensor(ds.targets))
    X = torch.cat(xs).float().div(255.0)
    m, s = torch.tensor(mean).view(1, -1, 1, 1), torch.tensor(std).view(1, -1, 1, 1)
    return ((X - m) / s).contiguous(), torch.cat(ys).long()


def load_fmnist(root: str | Path, split: str = "all") -> TensorDataset3:
    X, y = _load_torchvision("FashionMNIST", root, [0.2860], [0.3530], split)
    return TensorDataset3(X, y, torch.zeros_like(y), 10, 1)


def load_cifar(root: str | Path, classes: int, split: str = "all", half: bool = False) -> TensorDataset3:
    if classes == 10 and half:
        X, y = _load_torchvision("CIFAR10", root, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], split)
    elif classes == 10:
        X, y = _load_torchvision("CIFAR10", root, [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616], split)
    else:
        X, y = _load_torchvision("CIFAR100", root, [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762], split)
    return TensorDataset3(X, y, torch.zeros_like(y), classes, 1)


CELEBA_HELP = (
    "CelebA not found and the torchvision download failed (its Google Drive links are often rate-limited). "
    "Download it manually and lay it out as torchvision expects: {root}/celeba/list_attr_celeba.txt and "
    "{root}/celeba/img_align_celeba/*.jpg (unzip img_align_celeba.zip there). "
    "Source: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"
)


def _celeba_cache(root: Path, size: int) -> dict:
    """Decode every aligned image once (centre crop 178x178, resize to size x size) and cache it as uint8."""
    import pandas as pd
    from PIL import Image

    base = root / "celeba"
    cache = base / f"celeba{size}_uint8.pt"
    if cache.exists():
        return torch.load(cache)
    attr_file, img_dir = base / "list_attr_celeba.txt", base / "img_align_celeba"
    if not (attr_file.exists() and img_dir.is_dir()):
        try:
            from torchvision import datasets

            datasets.CelebA(str(root), split="all", target_type="attr", download=True)
        except Exception as e:
            raise FileNotFoundError(CELEBA_HELP.format(root=root)) from e
    attrs = pd.read_csv(attr_file, sep=r"\s+", skiprows=1)
    X = torch.empty((len(attrs), 3, size, size), dtype=torch.uint8)
    for i, fname in enumerate(attrs.index):
        with Image.open(img_dir / fname) as im:
            w, h = im.size
            left, top = (w - 178) // 2, (h - 178) // 2
            im = im.convert("RGB").crop((left, top, left + 178, top + 178)).resize((size, size), Image.BILINEAR)
            X[i] = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
    out = {"X": X, "attr": torch.from_numpy((attrs.to_numpy() > 0).astype(np.int64)), "names": list(attrs.columns)}
    torch.save(out, cache)
    return out


def load_celeba(root: str | Path, target_attr: str = "Smiling", sensitive_attr: str = "Male",
                subsample: int = 0, seed: int = 0, size: int = 64) -> TensorDataset3:
    """CelebA aligned faces at 64x64; label = `target_attr`, sensitive attribute = `sensitive_attr` (1 = present)."""
    d = _celeba_cache(Path(root), size)
    names = d["names"]
    for att in (target_attr, sensitive_attr):
        if att not in names:
            raise ValueError(f"unknown CelebA attribute {att!r}; choose from {names}")
    idx = torch.arange(len(d["X"]))
    if subsample and subsample < len(idx):
        idx = torch.from_numpy(np.sort(np.random.default_rng(seed).choice(len(idx), subsample, replace=False)))
    X = d["X"][idx].float().div(255.0).sub(0.5).div(0.5)
    attr = d["attr"][idx]
    y, a = attr[:, names.index(target_attr)].clone(), attr[:, names.index(sensitive_attr)].clone()
    return TensorDataset3(X, y, a, 2, 2)


# FairWeight tabular datasets, preprocessed as in the authors' load_data_utilities.py: label-encoded
# categoricals (kept as integer codes), standardised numerics, the sensitive attribute kept as a feature.
TABULAR: dict[str, dict] = {
    "bank": dict(label="y", sensitive="marital",
                 cat=["job", "marital", "education", "default", "housing", "loan", "contact", "month", "previous",
                      "poutcome"],
                 num=["age", "balance", "day", "duration", "campaign", "pdays"]),
    "default": dict(label="y", sensitive="SEX",
                    cat=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5",
                         "PAY_6"],
                    num=["AGE"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]),
    "law": dict(label="y", sensitive="sex",
                cat=["decile1b", "decile3", "fulltime", "fam_inc", "sex", "race", "tier"],
                num=["lsat", "ugpa", "zfygpa", "zgpa"]),
    "kdd": dict(label="class", sensitive="sex",
                cat=["class-of-worker", "education", "enroll-in-edu-inst-last-wk", "marital-stat",
                     "major-industry-code", "major-occupation-code", "race", "hispanic-origin", "sex",
                     "member-of-a-labor-union", "reason-for-unemployment", "full-or-part-time-employment-stat",
                     "tax-filer-stat", "region-of-previous-residence", "state-of-previous-residence",
                     "detailed-household-and-family-stat", "detailed-household-summary-in-household",
                     "migration-code-change-in-msa", "migration-code-change-in-reg",
                     "migration-code-move-within-reg", "live-in-this-house-1-year-ago",
                     "migration-prev-res-in-sunbelt", "family-members-under-18", "country-of-birth-father",
                     "country-of-birth-mother", "country-of-birth-self", "citizenship",
                     "own-business-or-self-employed", "fill-inc-questionnaire-for-veterans-admin"],
                num=["age", "detailed-industry-recode", "detailed-occupation-recode", "wage-per-hour",
                     "capital-gains", "capital-losses", "num-persons-worked-for-employer", "dividends-from-stocks",
                     "veterans-benefits", "weeks-worked-in-year", "year"]),
}


def load_tabular(name: str, root: str | Path) -> TensorDataset3:
    """Sensitive attribute (label-encoded as the authors do): bank marital (single = 1, married = 0),
    default SEX (male = 1), law sex (flipped as in the authors' code), kdd sex (male = 1).
    Label: bank y (yes = 1), default y, law y (flipped as in the authors' code), kdd class."""
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    path = Path(root) / f"{name}.csv"
    if not path.exists():
        src = "bank-full.csv" if name == "bank" else f"{name}.csv"
        raise FileNotFoundError(f"{path} not found. Copy datasets/{src} from the FairWeight authors' code "
                                f"archive (fairweight.zip) to {path}.")
    spec = TABULAR[name]
    df = pd.read_csv(path)
    if name == "law":
        df["y"] = 1 - df["y"]
        df["sex"] = 1 - df["sex"]
    for c in spec["cat"]:
        df[c] = LabelEncoder().fit_transform(df[c])
    num = df[spec["num"]].astype(np.float64)
    df[spec["num"]] = (num - num.mean()) / num.std(ddof=0)
    y = LabelEncoder().fit_transform(df.pop(spec["label"])).astype(np.int64)
    a = df[spec["sensitive"]].to_numpy().astype(np.int64)
    if set(np.unique(a).tolist()) - {0, 1}:
        raise ValueError(f"{name}: sensitive attribute {spec['sensitive']!r} is not binary")
    X = df.to_numpy().astype(np.float32)
    return TensorDataset3(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(a), 2, 2)


def load_synthetic(seed: int = 0, n: int = 4000, d: int = 10) -> TensorDataset3:
    """Small biased binary task for tests and quick runs: group 1 has a shifted positive rate."""
    g = np.random.default_rng(seed)
    a = g.integers(0, 2, n)
    X = g.normal(size=(n, d)).astype(np.float32)
    X[:, 0] += 0.8 * a
    logits = X[:, :3].sum(axis=1) + 0.7 * a - 0.5
    y = (logits + g.normal(scale=0.5, size=n) > 0).astype(np.int64)
    return TensorDataset3(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(a.astype(np.int64)), 2, 2)


def load_dataset(name: str, root: str | Path, seed: int = 0, split: str = "all", half: bool = False, **celeba_kw) -> TensorDataset3:
    Path(root).mkdir(parents=True, exist_ok=True)
    if split != "all" and name not in ("mnist", "fmnist", "cifar10", "cifar100"):
        raise ValueError(f"split={split!r} only applies to torchvision datasets, not {name!r}")
    if name == "adult":
        return load_adult(root)
    if name == "mnist":
        return load_mnist(root, split)
    if name == "synthetic":
        return load_synthetic(seed)
    if name == "fmnist":
        return load_fmnist(root, split)
    if name in ("cifar10", "cifar100"):
        return load_cifar(root, int(name[5:]), split, half)
    if name == "celeba":
        return load_celeba(root, seed=seed, **celeba_kw)
    if name in TABULAR:
        return load_tabular(name, root)
    raise ValueError(f"unknown dataset {name!r}")
