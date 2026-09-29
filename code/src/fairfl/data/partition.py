from __future__ import annotations

import numpy as np


def dirichlet_partition(
    labels: np.ndarray, num_clients: int, alpha: float, rng: np.random.Generator, min_samples: int = 20
) -> list[np.ndarray]:
    """Split sample indices by label so each class is spread over clients by Dir(alpha).

    Redraws until every client holds at least `min_samples` samples.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    for _ in range(1000):
        buckets: list[list[int]] = [[] for _ in range(num_clients)]
        for k in classes:
            idx = rng.permutation(np.flatnonzero(labels == k))
            props = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for c, part in enumerate(np.split(idx, cuts)):
                buckets[c].extend(part.tolist())
        if min(len(b) for b in buckets) >= min_samples:
            return [rng.permutation(np.array(b, dtype=np.int64)) for b in buckets]
    raise RuntimeError("could not satisfy min_samples; raise alpha or lower min_samples")
