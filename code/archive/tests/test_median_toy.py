import pytest

from fairfl.experiments.median_toy import run


def test_median_and_signsgd_stall_at_a2_with_mean_gradient_7_over_3():
    for method in ("median", "signsgd"):
        r = run(method)
        assert r["x"] == pytest.approx(2.0, abs=0.01)
        assert r["mean_grad"] == pytest.approx(-7 / 3, abs=0.01)


def test_noise_before_median_moves_towards_mean():
    assert run("mean")["x"] == pytest.approx(13 / 3, abs=0.01)
    assert abs(run("median", b=10.0, steps=100_000)["mean_grad"]) < 0.2
