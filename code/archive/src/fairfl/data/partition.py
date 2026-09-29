from __future__ import annotations

import numpy as np


def dirichlet_partition(
    keys: np.ndarray, num_clients: int, alpha: float, rng: np.random.Generator, min_samples: int = 20
) -> list[np.ndarray]:
    """Split sample indices so each key value (label, group, ...) is spread over clients by Dir(alpha).

    Redraws until every client holds at least `min_samples` samples.
    """
    keys = np.asarray(keys)
    classes = np.unique(keys)
    for _ in range(1000):
        buckets: list[list[int]] = [[] for _ in range(num_clients)]
        for k in classes:
            idx = rng.permutation(np.flatnonzero(keys == k))
            props = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for c, part in enumerate(np.split(idx, cuts)):
                buckets[c].extend(part.tolist())
        if min(len(b) for b in buckets) >= min_samples:
            return [rng.permutation(np.array(b, dtype=np.int64)) for b in buckets]
    raise RuntimeError("could not satisfy min_samples; raise alpha or lower min_samples")


def iid_partition(n: int, num_clients: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [np.sort(p) for p in np.array_split(rng.permutation(n), num_clients)]


def shard_partition(labels: np.ndarray, num_clients: int, shards_per_client: int,
                    rng: np.random.Generator, shuffle: bool = True) -> list[np.ndarray]:
    """McMahan et al. (2017): sort by label, cut into num_clients * shards_per_client contiguous shards
    (sizes differ by at most one), and give each client shards_per_client random shards."""
    labels = np.asarray(labels)
    order = rng.permutation(len(labels))
    order = order[np.argsort(labels[order], kind="stable")]
    shards = np.array_split(order, num_clients * shards_per_client)
    assign = rng.permutation(len(shards)).reshape(num_clients, shards_per_client)
    parts = [np.concatenate([shards[j] for j in row]) for row in assign]
    return [rng.permutation(p) for p in parts] if shuffle else parts


def partition_keys(y: np.ndarray, a: np.ndarray, on: str, num_classes: int) -> np.ndarray:
    if on == "label":
        return y
    if on == "sensitive":
        return a
    if on == "joint":
        return a * num_classes + y
    raise ValueError(f"unknown partition key {on!r}")
