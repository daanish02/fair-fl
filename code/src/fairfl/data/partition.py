from __future__ import annotations

import numpy as np


def dirichlet_partition(
    labels: np.ndarray, num_clients: int, alpha: float, rng: np.random.Generator, min_samples: int = 20,
    class_props: dict[int, np.ndarray] | None = None,
) -> tuple[list[np.ndarray], dict[int, np.ndarray]]:
    """Split sample indices by label so each class is spread over clients by Dir(alpha).

    Redraws until every client holds at least `min_samples` samples. Returns (buckets, class_props):
    class_props maps each class to the per-client proportions drawn for it, so a second call (e.g. to
    partition a different split of the same classes) can pass it back in via `class_props` to reuse the
    exact same per-client label mix instead of drawing fresh, independent proportions.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    for _ in range(1000):
        buckets: list[list[int]] = [[] for _ in range(num_clients)]
        used_props: dict[int, np.ndarray] = {}
        for k in classes:
            idx = rng.permutation(np.flatnonzero(labels == k))
            props = class_props[int(k)] if class_props is not None else rng.dirichlet(np.full(num_clients, alpha))
            used_props[int(k)] = props
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for c, part in enumerate(np.split(idx, cuts)):
                buckets[c].extend(part.tolist())
        if min(len(b) for b in buckets) >= min_samples:
            return [rng.permutation(np.array(b, dtype=np.int64)) for b in buckets], used_props
    raise RuntimeError("could not satisfy min_samples; raise alpha or lower min_samples")
