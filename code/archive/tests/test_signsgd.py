import numpy as np
import torch

from fairfl.core.config import TrainConfig
from fairfl.core.types import FitRes
from fairfl.strategies.signsgd import SignSGD, SignSGDParams


def test_majority_vote_is_sign_of_median_for_odd_clients():
    s = SignSGD(SignSGDParams(step=0.1), TrainConfig(lr=1.0), np.random.default_rng(0))
    g = torch.zeros(3)
    grads = torch.tensor([[1.0, -2.0, 0.5], [3.0, -1.0, -0.5], [-1.0, 4.0, -0.2]])
    res = [FitRes(client_id=i, params=g - gi, num_samples=1) for i, gi in enumerate(grads)]  # w_i = w - lr * g_i
    out = s.aggregate(0, g, res)
    torch.testing.assert_close(out, -0.1 * torch.sign(grads.median(dim=0).values))
