"""LoGoFair's Adult client splits and pre-trained FedAvg model, generated exactly as the official code
(github.com/liizhang/LoGofair) does when no shipped split exists. The repo ships only alpha = 0.5; this builds
alpha = 5 and 100 for Table 1.

Pipeline (fedlearn/utils/data_utils.py, sampling.py, algorithm/FedFairPost.py):
- adult.data and adult.test are processed separately (one-hot, per-file min-max scaling), aligned, concatenated
  (48,842 rows); `sex` becomes A (Female = 1) and is removed from X (106 features);
- np.random.seed(1 + 111), then Dirichlet(2 * alpha) over 5 clients on A, redrawn until every client holds at least
  20 samples of each group;
- the generating run then permutes each client 70/30 with the RNG state left by the partition and pre-trains FedAvg
  on that 70%: logistic model, BCELoss, Adam (lr 0.005, weight decay 1e-4, state kept per client), 40 batches of 512
  per round, 25 rounds, unweighted mean;
- later runs (the Table 1 methods) reload the split with a fresh np.random.seed(112) permutation, which is what
  data.split_seed: 112 does in our split_file loader.

Usage: uv run python -m fairfl.data.logofair_data --alpha 5 --alpha 100 [--check]
Raw files: <root>/logofair/raw/adult.data and adult.test (UCI).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

CATEGORICAL = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "native-country"]
CONTINUOUS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
           "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"]


def _process(path: Path, favorable: str):
    import pandas as pd

    df = pd.read_csv(path, delimiter=",", header=None, na_values=[], skiprows=1 if path.name.endswith("test") else 0)
    df.columns = COLUMNS
    df = pd.get_dummies(df[COLUMNS], columns=CATEGORICAL)
    df[CONTINUOUS] = df[CONTINUOUS].apply(lambda v: (v - v.min()) / (v.max() - v.min()), axis=0)
    df["salary"] = (df["salary"] == favorable).astype(int)
    df["sex"] = df["sex"].map({" Male": 0, " Female": 1})
    return df


def load_adult(raw: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd

    train = _process(raw / "adult.data", " >50K")
    test = _process(raw / "adult.test", " >50K.")
    test["native-country_ Holand-Netherlands"] = False
    test = test[train.columns]
    df = pd.concat([train, test], axis=0)
    cols = df.drop(columns="salary").columns.tolist()
    X = df.drop(columns="salary").to_numpy().astype(np.float32)
    y = df["salary"].to_numpy().astype(np.float32)
    i = cols.index("sex")
    return np.delete(X, i, axis=1), y, X[:, i].copy()


def dirichlet(targets: np.ndarray, num_users: int, alpha: float, least_samples: int = 20) -> list[np.ndarray]:
    """sampling.dirichlet, including its Dir(2 * alpha) concentration and the per-client cap of N / num_users."""
    t = np.array(targets, dtype=np.int32)
    by_label = [np.where(t == k)[0] for k in range(len(np.unique(t)))]
    min_size = 0
    while min_size < least_samples:
        idx = [np.array([], dtype=np.int64) for _ in range(num_users)]
        sizes = []
        for k in range(len(by_label)):
            np.random.shuffle(by_label[k])
            p = np.random.dirichlet(np.repeat(alpha * 2, num_users))
            p = np.array([pj * (len(ij) < len(t) / num_users) for pj, ij in zip(p, idx)])
            p = p / p.sum()
            cut = (np.cumsum(p) * len(by_label[k])).astype(int)[:-1]
            idx = [np.concatenate((ij, part.tolist())).astype(np.int64)
                   for ij, part in zip(idx, np.split(by_label[k], cut))]
            sizes.append(min(len(np.intersect1d(ij, by_label[k])) for ij in idx))
        min_size = min(sizes)
    return idx


def pretrain(clients: list[tuple[np.ndarray, np.ndarray]], rounds: int = 25, lr: float = 0.005, wd: float = 1e-4,
             batch: int = 512, steps: int = 40) -> torch.Tensor:
    torch.manual_seed(12 + 111)
    dim = clients[0][0].shape[1]
    glob = nn.Linear(dim, 1)
    flat = lambda m: torch.cat([p.data.flatten() for p in m.parameters()])  # noqa: E731
    models, opts, iters, loaders = [], [], [], []
    for X, y in clients:
        m = nn.Linear(dim, 1)
        models.append(m)
        opts.append(torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd))
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y).reshape(-1, 1))
        loaders.append(torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True))
        iters.append(iter(loaders[-1]))
    g = flat(glob)
    loss_fn = nn.BCELoss()
    for _ in range(rounds):
        solns = []
        for c, m in enumerate(models):
            torch.nn.utils.vector_to_parameters(g.clone(), m.parameters())
            for _ in range(steps):
                try:
                    x, t = next(iters[c])
                except StopIteration:
                    iters[c] = iter(loaders[c])
                    x, t = next(iters[c])
                opts[c].zero_grad()
                loss_fn(torch.sigmoid(m(x)), t).backward()
                opts[c].step()
            solns.append(flat(m))
        g = torch.stack(solns).mean(0)
    return g


def build(alpha: float, root: Path, num_users: int = 5, seed: int = 111) -> dict:
    X, y, A = load_adult(root / "logofair" / "raw")
    np.random.seed(1 + seed)
    idx = dirichlet(A, num_users, alpha)
    split = {"users": [str(i) for i in range(num_users)], "user_data": {}, "num_samples": {}}
    train = []
    for i, ii in enumerate(idx):
        split["user_data"][str(i)] = {"x": X[ii].tolist(), "y": y[ii].tolist(), "A": A[ii].tolist()}
        split["num_samples"][str(i)] = len(ii)
        perm = np.random.permutation(len(ii))[: int(len(ii) * 0.7)]  # read_data, RNG continued from the partition
        train.append((X[ii][perm], y[ii][perm]))
    out = root / "logofair"
    with open(out / f"adult_alpha{alpha:g}_split.json", "w") as f:
        json.dump(split, f)
    torch.save(pretrain(train), out / f"adult_alpha{alpha:g}_pre_model.pth")
    return {str(i): (len(ii), int((A[ii] == 0).sum()), int((A[ii] == 1).sum())) for i, ii in enumerate(idx)}


def check_against_shipped(root: Path) -> None:
    """Regenerates alpha = 0.5 in memory and compares client counts with the shipped split."""
    X, y, A = load_adult(root / "logofair" / "raw")
    np.random.seed(112)
    idx = dirichlet(A, 5, 0.5)
    shipped = json.load(open(root / "logofair" / "adult_alpha0.5_split.json"))
    for i, ii in enumerate(idx):
        s = shipped["user_data"][str(i)]
        same = len(ii) == len(s["y"]) and np.allclose(X[ii], np.array(s["x"], dtype=np.float32), atol=1e-6)
        print(f"client {i}: ours {len(ii)} shipped {len(s['y'])} rows identical: {same}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, action="append", default=[])
    ap.add_argument("--root", default="data")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    if a.check:
        check_against_shipped(root)
    for alpha in a.alpha:
        print(alpha, build(alpha, root))


if __name__ == "__main__":
    main()
