import numpy as np

from fairfl.data.partition import dirichlet_partition, iid_partition


def test_dirichlet_covers_every_index_once():
    keys = np.repeat(np.arange(4), 250)
    parts = dirichlet_partition(keys, 8, 0.5, np.random.default_rng(0), min_samples=10)
    allidx = np.concatenate(parts)
    assert len(allidx) == len(keys) and len(np.unique(allidx)) == len(keys)
    assert min(len(p) for p in parts) >= 10


def test_small_alpha_is_more_skewed():
    keys = np.repeat(np.arange(2), 2000)

    def skew(alpha):
        parts = dirichlet_partition(keys, 10, alpha, np.random.default_rng(1), min_samples=5)
        return np.mean([abs(keys[p].mean() - 0.5) for p in parts])

    assert skew(0.1) > skew(100.0)


def test_iid_partition_sizes():
    parts = iid_partition(103, 5, np.random.default_rng(0))
    assert sorted(len(p) for p in parts) == [20, 20, 21, 21, 21]
