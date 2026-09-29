from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairfl.core.config import DataConfig
from fairfl.data.datasets import TensorDataset3, load_dataset
from fairfl.data.partition import dirichlet_partition, iid_partition, partition_keys, shard_partition


@dataclass
class ClientData:
    client_id: int
    train: TensorDataset3
    test: TensorDataset3
    val: TensorDataset3 | None = None

    def to(self, device) -> ClientData:
        return ClientData(self.client_id, self.train.to(device), self.test.to(device),
                          self.val.to(device) if self.val is not None else None)

    def nbytes(self) -> int:
        return sum(d.nbytes() for d in (self.train, self.test, self.val) if d is not None)


@dataclass
class FederatedData:
    clients: list[ClientData]
    num_features: tuple[int, ...]
    num_classes: int
    num_groups: int
    global_test: TensorDataset3 | None = None


def build_from_split_file(cfg: DataConfig, seed: int) -> FederatedData:
    """Fixed client partition from a JSON file; per-client random train/test split as in the LoGoFair code."""
    import json

    import torch

    with open(cfg.split_file, encoding="utf-8") as f:
        raw = json.load(f)
    legacy = np.random.RandomState(cfg.split_seed) if cfg.split_seed is not None else None
    rng = np.random.default_rng(seed)
    clients = []
    for cid, key in enumerate(raw["users"]):
        u = raw["user_data"][key]
        ds = TensorDataset3(torch.tensor(u["x"], dtype=torch.float32), torch.tensor(u["y"]).long(),
                            torch.tensor(u["A"]).long(), 2, 2)
        n = len(ds)
        perm = legacy.permutation(n) if legacy is not None else rng.permutation(n)
        n_tr = int(n * (1 - cfg.test_fraction))
        train, test = ds.subset(perm[:n_tr]), ds.subset(perm[n_tr:])
        if cfg.val_from_train:
            val = train
        elif cfg.val_fraction > 0:
            n_val = int(round(n_tr * cfg.val_fraction))
            val, train = ds.subset(perm[:n_val]), ds.subset(perm[n_val:n_tr])
        else:
            val = None
        clients.append(ClientData(cid, train, test, val))
    return FederatedData(clients, tuple(clients[0].train.X.shape[1:]), 2, 2)


def build_federated(cfg: DataConfig, seed: int) -> FederatedData:
    if cfg.name == "split_file":
        return build_from_split_file(cfg, seed)
    if cfg.name.startswith("fw_"):
        from fairfl.data.fairweight_data import build_fairweight

        return build_fairweight(cfg.name[3:], cfg.root, cfg.num_clients)
    rng = np.random.default_rng(seed)
    kw = ({"target_attr": cfg.target_attr, "sensitive_attr": cfg.sensitive_attr, "subsample": cfg.subsample}
          if cfg.name == "celeba" else {})
    full = load_dataset(cfg.name, cfg.root, seed, split="train" if cfg.source == "train" else "all",
                        half=cfg.normalize == "half", **kw)
    global_test = None
    if cfg.global_test == "official":
        if cfg.source != "train":
            raise ValueError("data.global_test=official needs data.source=train (else it overlaps client data)")
        global_test = load_dataset(cfg.name, cfg.root, seed, split="test", **kw)
    ordered = cfg.client_split == "ordered"
    if cfg.partition_file:
        import json

        with open(cfg.partition_file, encoding="utf-8") as f:
            raw = json.load(f)["train_data"]
        parts = [np.asarray(raw[k], dtype=np.int64) for k in sorted(raw, key=int)]
        if len(parts) != cfg.num_clients:
            raise ValueError(f"partition_file has {len(parts)} clients, config says {cfg.num_clients}")
    if cfg.augment and full.X.dim() != 4:
        raise ValueError("data.augment needs image data [N, C, H, W]")
    y, a = full.y.numpy(), full.a.numpy()
    if cfg.partition_file:
        pass
    elif cfg.partition_on == "iid":
        parts = iid_partition(len(full), cfg.num_clients, rng)
    elif cfg.partition_on == "shards":
        parts = shard_partition(y, cfg.num_clients, cfg.shards_per_client, rng, shuffle=not ordered)
    else:
        keys = partition_keys(y, a, cfg.partition_on, full.num_classes)
        parts = dirichlet_partition(keys, cfg.num_clients, cfg.alpha, rng, cfg.min_samples)
    clients = []
    for cid, idx in enumerate(parts):
        n_test = int(round(len(idx) * cfg.test_fraction))
        if cfg.test_fraction > 0:
            n_test = max(1, n_test)
        if ordered:
            # FCFL code (Update.train_val_test): consecutive slices train | val | test, no shuffling.
            n_val = int(round(len(idx) * cfg.val_fraction))
            n_tr = len(idx) - n_val - n_test
            train, test = full.subset(idx[:n_tr]), full.subset(idx[n_tr + n_val:])
            val = full.subset(idx[n_tr:n_tr + n_val]) if n_val else None
            train.augment = cfg.augment
            clients.append(ClientData(cid, train, test, val))
            continue
        idx = rng.permutation(idx)
        rest = idx[n_test:]
        n_val = int(round(len(rest) * cfg.val_fraction))
        val = full.subset(rest[:n_val]) if n_val else None
        train = full.subset(rest[n_val:])
        train.augment = cfg.augment
        clients.append(ClientData(cid, train, full.subset(idx[:n_test]), val))
    return FederatedData(clients, tuple(full.X.shape[1:]), full.num_classes, full.num_groups, global_test)
