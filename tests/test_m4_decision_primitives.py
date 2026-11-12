"""Hand-checkable synthetic M4 examples; no datasets or model fitting."""

import numpy as np
import pytest

from src.decision.cost_simulation import cost_index, newsvendor_cost
from src.decision.quantile_models import (
    QUANTILE_LEVELS, coverage_gap, enforce_quantile_order,
    canonical_tau_key, normalize_tau, nonnegative_quantity, pinball_loss, quantile_coverage,
)


@pytest.mark.parametrize("tau, expected", [
    (0.5, [1, 1, 0]), (0.75, [1.5, 0.5, 0]),
    (5 / 6, [5 / 3, 1 / 3, 0]),
])
def test_pinball_hand_calculation(tau, expected):
    np.testing.assert_allclose(pinball_loss([4, 0, 2], [2, 2, 2], tau), expected)


@pytest.mark.parametrize("tau", QUANTILE_LEVELS)
def test_coverage_includes_ties_and_gap_has_sign(tau):
    assert quantile_coverage([1, 2, 4, 5], [1, 3, 3, 6]) == 0.75
    assert coverage_gap([1, 2, 4, 5], [1, 3, 3, 6], tau) == pytest.approx(0.75 - tau)


@pytest.mark.parametrize("under, expected", [(1, [2, 2, 0]), (3, [6, 2, 0]), (5, [10, 2, 0])])
def test_newsvendor_hand_calculation(under, expected):
    np.testing.assert_allclose(newsvendor_cost([4, 0, 2], [2, 2, 2], 1, under), expected)


@pytest.mark.parametrize("over, under, tau", [(1, 1, 0.5), (1, 3, 0.75), (1, 5, 5 / 6)])
def test_cost_equals_scaled_pinball_per_observation(over, under, tau):
    demand = np.array([0, 0, 1, 2.5, 10, 20.0])
    raw_quantity = np.array([-3, 2, 1, 3, 4, 25.0])
    quantity = nonnegative_quantity(raw_quantity)
    np.testing.assert_allclose(
        newsvendor_cost(demand, raw_quantity, over, under),
        (over + under) * pinball_loss(demand, quantity, tau),
    )


def test_nonnegative_quantity_preserves_fractional_values_and_input():
    raw = np.array([-3.0, 0, 2.5])
    np.testing.assert_array_equal(nonnegative_quantity(raw), [0, 0, 2.5])
    np.testing.assert_array_equal(raw, [-3, 0, 2.5])
    np.testing.assert_array_equal(newsvendor_cost([0, 2, 2.5], raw, 1, 3), [0, 6, 0])


def test_quantile_order_correction_clips_sorts_and_does_not_mutate():
    raw = np.array([[8, -1, 3], [1, 2, 2], [-3, -2, -1.0]])
    original = raw.copy()
    ordered = enforce_quantile_order(raw)
    np.testing.assert_array_equal(ordered, [[0, 3, 8], [1, 2, 2], [0, 0, 0]])
    assert np.all(np.diff(ordered, axis=1) >= 0)
    assert np.all(ordered >= 0)
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize("policy,baseline", [([0, 0], [0, 0]), ([1, 2], [0, 0]), ([], [])])
def test_cost_index_zero_denominator_is_nan(policy, baseline):
    assert np.isnan(cost_index(policy, baseline))


def test_cost_index_uses_ratio_of_totals_including_zero_cost_rows():
    assert cost_index([1, 3], [0, 8]) == 0.5
    assert cost_index([0, 8], [0, 8]) == 1.0


def test_empty_coverage_is_undefined_and_empty_losses_are_empty():
    assert np.isnan(quantile_coverage([], []))
    assert np.isnan(coverage_gap([], [], 0.75))
    assert pinball_loss([], [], 0.5).size == 0
    assert newsvendor_cost([], []).size == 0


@pytest.mark.parametrize("tau", [0, 1, -0.1, np.nan, np.inf])
def test_invalid_tau_rejected(tau):
    with pytest.raises(ValueError):
        pinball_loss([1], [1], tau)
    with pytest.raises(ValueError):
        coverage_gap([1], [1], tau)


@pytest.mark.parametrize("first, second", [([1, 2], [1]), ([np.nan], [1]), ([1], [np.inf]), ([[1]], [[1]])])
def test_invalid_observations_rejected_without_broadcasting(first, second):
    for function in (quantile_coverage, newsvendor_cost, cost_index):
        with pytest.raises(ValueError):
            function(first, second)
    with pytest.raises(ValueError):
        pinball_loss(first, second, 0.5)


@pytest.mark.parametrize("over,under", [(0, 1), (1, -1), (np.nan, 3), (1, np.inf)])
def test_invalid_cost_weights_rejected(over, under):
    with pytest.raises(ValueError):
        newsvendor_cost([1], [2], over, under)


def test_negative_demand_and_negative_costs_rejected():
    with pytest.raises(ValueError):
        newsvendor_cost([-1], [0])
    with pytest.raises(ValueError):
        cost_index([-1], [1])
    with pytest.raises(ValueError):
        cost_index([1], [-1])


@pytest.mark.parametrize("predictions,taus", [
    ([[1, 2, 3]], [0.75, 0.5, 5 / 6]),
    ([[1, 2, 3]], [0.5, 0.5, 5 / 6]),
    ([[1, 2, 3]], [0, 0.5, 1]),
    ([[1, 2]], QUANTILE_LEVELS),
    ([[1, np.nan, 3]], QUANTILE_LEVELS),
    ([1, 2, 3], QUANTILE_LEVELS),
    ([[]], []),
])
def test_invalid_quantile_matrix_or_levels_rejected(predictions, taus):
    with pytest.raises(ValueError):
        enforce_quantile_order(predictions, taus)


@pytest.mark.parametrize("tau,expected", [(0.5, 1e308), (0.75, 1.5e308), (5 / 6, (5 / 3) * 1e308)])
def test_pinball_overflowing_residual_with_representable_loss(tau, expected):
    with np.errstate(all="raise"):
        loss = pinball_loss([1e308], [-1e308], tau)
    assert loss[0] == pytest.approx(expected)


def test_pinball_overflowing_residual_for_overprediction():
    assert pinball_loss([-1e308], [1e308], 0.75)[0] == pytest.approx(5e307)


def test_pinball_unrepresentable_output_raises():
    maximum = np.finfo(float).max
    with pytest.raises(ValueError, match="float64 range"):
        pinball_loss([maximum], [-maximum], 0.75)


@pytest.mark.parametrize("policy,baseline,expected", [
    ([1e308, 1e308], [1e308, 1e308], 1.0),
    ([1e308, 0], [1e308, 1e308], 0.5),
    ([1e308, 1e308], [1e308, 0], 2.0),
    ([0, 0], [1e308, 1e308], 0.0),
    ([1, 0], [1e308, 1e308], 5e-309),
    ([1e308, 1e308], [1, 1], 1e308),
])
def test_cost_index_large_totals_stable(policy, baseline, expected):
    with np.errstate(all="raise"):
        actual = cost_index(policy, baseline)
    assert actual == pytest.approx(expected, rel=1e-14, abs=0)


@pytest.mark.parametrize("policy,baseline", [
    ([1e308], [1e-308]), ([1e-308], [1e308]),
])
def test_unrepresentable_cost_index_raises(policy, baseline):
    with pytest.raises(ValueError, match="float64 range"):
        cost_index(policy, baseline)


def test_newsvendor_large_representable_cost():
    np.testing.assert_allclose(newsvendor_cost([0, 1e308], [1e308, 0], 1, 1), [1e308, 1e308])


@pytest.mark.parametrize("demand,quantity,over,under", [
    ([1e308], [0], 1, 3), ([0], [1e308], 3, 1),
])
def test_newsvendor_unrepresentable_cost_raises(demand, quantity, over, under):
    with pytest.raises(ValueError, match="float64 range"):
        newsvendor_cost(demand, quantity, over, under)


def test_nonzero_underflow_is_not_silently_zero():
    tiny = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="float64 range"):
        pinball_loss([tiny], [0], 0.25)
    with pytest.raises(ValueError, match="float64 range"):
        newsvendor_cost([tiny], [0], 1, 0.25)


@pytest.mark.parametrize("bad", [
    np.array([1 + 2j]), np.array([1 + 0j]),
    np.array([1 + 2j], dtype=object),
    np.ma.array([1.0], mask=[True]), np.ma.array([1.0], mask=[False]),
])
def test_reject_complex_and_masked_arrays_everywhere(bad):
    for function in (quantile_coverage, newsvendor_cost, cost_index):
        with pytest.raises(ValueError, match="complex|masked"):
            function(bad, [1])
        with pytest.raises(ValueError, match="complex|masked"):
            function([1], bad)
    with pytest.raises(ValueError, match="complex|masked"):
        pinball_loss(bad, [1], 0.5)
    with pytest.raises(ValueError, match="complex|masked"):
        pinball_loss([1], bad, 0.5)
    with pytest.raises(ValueError, match="complex|masked"):
        nonnegative_quantity(bad)
    with pytest.raises(ValueError, match="complex|masked"):
        enforce_quantile_order(bad.reshape(1, 1), [0.5])
    with pytest.raises(ValueError, match="complex|masked"):
        enforce_quantile_order([[1]], bad)


@pytest.mark.parametrize("bad", ["0.75", 0.75 + 0j, True, None, [0.75], np.array(0.75), np.nan, np.inf])
def test_scalar_validation_has_clear_value_errors(bad):
    for function in (pinball_loss, coverage_gap):
        with pytest.raises(ValueError, match="tau"):
            function([1], [1], bad)
    for function in (normalize_tau, canonical_tau_key):
        with pytest.raises(ValueError, match="tau"):
            function(bad)
    with pytest.raises(ValueError, match="c_over"):
        newsvendor_cost([1], [1], bad, 1)
    with pytest.raises(ValueError, match="c_under"):
        newsvendor_cost([1], [1], 1, bad)


@pytest.mark.parametrize("tau,canonical,key", [
    (0.5, 0.5, "tau_0.50"), (np.float64(0.75), 0.75, "tau_0.75"),
    (0.8333333333, 5 / 6, "tau_5_6"), (5 / 6, 5 / 6, "tau_5_6"),
    (0.75 + 5e-10, 0.75, "tau_0.75"),
])
def test_canonical_tau_keys_accept_yaml_rounding(tau, canonical, key):
    assert normalize_tau(tau) == canonical
    assert canonical_tau_key(tau) == key
    assert canonical_tau_key(normalize_tau(tau)) == key


@pytest.mark.parametrize("tau", [0.8333, 0.51, 0.75 + 2e-9])
def test_canonical_tau_rejects_unsupported_or_coarse_values(tau):
    with pytest.raises(ValueError, match="project level"):
        canonical_tau_key(tau)


@pytest.mark.parametrize("over,under", [(2, 2), (2, 6), (2, 10), (0.25, 1.25)])
def test_nonunit_cost_scales_identity(over, under):
    demand, quantity = [0, 2.5, 10], [1, 2, 12]
    tau = under / (over + under)
    np.testing.assert_allclose(
        newsvendor_cost(demand, quantity, over, under),
        (over + under) * pinball_loss(demand, quantity, tau),
    )


def test_empty_quantile_matrix_and_single_level():
    assert enforce_quantile_order(np.empty((0, 3))).shape == (0, 3)
    np.testing.assert_array_equal(enforce_quantile_order([[-1], [2.5]], [0.5]), [[0], [2.5]])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_invalid_nonfinite_quantity_and_matrix(bad):
    with pytest.raises(ValueError, match="finite"):
        nonnegative_quantity([bad])
    with pytest.raises(ValueError, match="finite"):
        enforce_quantile_order([[0, bad, 1]])


@pytest.mark.parametrize("truth,prediction,expected", [
    ([0, 0, 0], [0, 0, 0], 1.0),
    ([0, 0, 0, 2], [0, 0, 0, 1], 0.75),
    ([1], [0], 0.0), ([1], [1], 1.0),
])
def test_ties_and_zero_heavy_coverage(truth, prediction, expected):
    assert quantile_coverage(truth, prediction) == expected
    assert coverage_gap(truth, prediction, 0.75) == expected - 0.75
