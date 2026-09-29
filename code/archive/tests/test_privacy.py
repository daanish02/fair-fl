import pytest

from fairfl.metrics.privacy import max_steps


@pytest.mark.parametrize("sigma, paper_T", [(1.5, 115), (2.0, 268), (2.5, 463), (3.0, 708)])
def test_fedfdp_fig4_round_counts(sigma, paper_T):
    """FedFDP (ACNS 2026) Fig. 4 text: max rounds at eps = 2, q = 0.05, delta = 1e-5 (gradient mechanism only)."""
    assert abs(max_steps(0.05, sigma, 2.0, 1e-5) - paper_T) <= 7


def test_fedfdp_default_budget():
    assert max_steps(0.05, 2.0, 3.52, 1e-5) == pytest.approx(776, abs=3)
