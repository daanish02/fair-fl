import numpy as np
import pytest

from fairfl.core.types import EvalRes, GroupCounts, RoundLog
from fairfl.fairness.scheme import FairnessScheme, ScopeSpec, StatusSpec
from fairfl.fairness.scopes import evaluate_scheme
from fairfl.fairness.status import unfairness_trajectory


def round_log(rnd: int, pos_rate: tuple[float, float], n: int = 100) -> RoundLog:
    """One client, two groups; group g gets `pos_rate[g]` positive predictions (the 'benefit')."""
    pos = [[int(pos_rate[g] * n), 0] for g in range(2)]
    counts = GroupCounts(n=[[n, 0], [n, 0]], correct=[[0, 0], [0, 0]], pred_pos=pos)
    ev = EvalRes(client_id=0, num_samples=2 * n, loss=0.0, accuracy=0.5, groups=counts)
    return RoundLog(round=rnd, selected=[0], train_loss=0.0, global_accuracy=0.5, global_loss=0.0, clients=[ev])


# Alamdari et al. vaccine example: A served in months 1-2, B in months 3-4.
VACCINE = [round_log(t, r) for t, r in enumerate([(1, 0), (1, 0), (0, 1), (0, 1)])]


def scheme(kind: str, accumulation: str = "cumulative") -> FairnessScheme:
    return FairnessScheme(status=StatusSpec(benefit="dp", level="global", accumulation=accumulation),
                          scope=ScopeSpec(kind=kind, period=2), epsilon=0.05)


def test_vaccine_trajectory():
    _, u = unfairness_trajectory(VACCINE, scheme("anytime"))
    np.testing.assert_allclose(u, [1.0, 1.0, 1 / 3, 0.0], atol=1e-9)


def test_vaccine_is_fair_long_term_but_not_anytime():
    assert evaluate_scheme(VACCINE, scheme("long_term")).max == pytest.approx(0.0)
    anytime = evaluate_scheme(VACCINE, scheme("anytime"))
    assert anytime.max == pytest.approx(1.0)
    assert anytime.violation_rate == pytest.approx(0.75)


def test_periodic_scope_uses_every_pth_deployment():
    s = evaluate_scheme(VACCINE, scheme("periodic"))
    assert s.points == 2 and s.max == pytest.approx(1.0) and s.final == pytest.approx(0.0)


def test_instant_status_is_markovian():
    _, u = unfairness_trajectory(VACCINE, scheme("anytime", "instant"))
    np.testing.assert_allclose(u, [1.0, 1.0, 1.0, 1.0])


def test_deploy_every_filters_history():
    s = FairnessScheme(status=StatusSpec(benefit="dp", level="global"), scope=ScopeSpec(kind="anytime"),
                       deploy_every=2)
    dep, u = unfairness_trajectory(VACCINE, s)
    assert dep == [1, 3]
    np.testing.assert_allclose(u, [1.0, 0.0])
