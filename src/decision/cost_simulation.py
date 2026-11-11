"""Pure one-period Newsvendor cost primitives; costs are relative, not money."""

from fractions import Fraction

import numpy as np

from .quantile_models import _paired_vectors, _real_scalar, nonnegative_quantity


def newsvendor_cost(demand, quantity, c_over=1.0, c_under=1.0):
    """Return c_over*(Q-D)+ + c_under*(D-Q)+ PER OBSERVATION.

    Demand is finite, nonnegative observed sales; finite Q is clipped at zero
    as required by cost_scenarios.yaml. Costs must be positive finite scalars.
    At tau=c_under/(c_over+c_under), this equals (c_over+c_under) times
    pinball_loss(demand, clipped_Q, tau). No training, averaging or I/O occurs.
    Raise ValueError on overflow or a nonzero cost rounded to zero (underflow).
    """
    c_over = _real_scalar(c_over, "c_over")
    c_under = _real_scalar(c_under, "c_under")
    if c_over <= 0 or c_under <= 0:
        raise ValueError("c_over and c_under must be positive finite real scalars")
    demand, quantity = _paired_vectors(demand, quantity)
    if np.any(demand < 0):
        raise ValueError("Demand must be nonnegative")
    quantity = nonnegative_quantity(quantity)
    difference = quantity - demand  # Nonnegative operands: subtraction cannot overflow.
    weight = np.where(difference >= 0, c_over, c_under)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        costs = weight * np.abs(difference)
    if not np.isfinite(costs).all() or np.any((difference != 0) & (costs == 0)):
        raise ValueError("Newsvendor cost is outside the nonzero finite float64 range")
    return costs


def cost_index(policy_costs, baseline_costs):
    """Ratio of SUMS on caller-aligned observations, with baseline index 1.

    Return NaN if summed baseline cost is zero (including empty inputs), even
    when policy cost is also zero. Never average observation-level ratios.
    Inputs must be finite nonnegative costs with identical shapes; the caller
    is responsible for matching date/store/family/fold keys before scoring.
    Overflowing totals use an exact rational fallback; unrepresentable final
    ratios raise ValueError. A genuinely zero policy cost still returns zero.
    """
    policy, baseline = _paired_vectors(policy_costs, baseline_costs)
    if np.any(policy < 0) or np.any(baseline < 0):
        raise ValueError("Costs must be nonnegative")
    if not np.any(baseline > 0):
        return float("nan")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        numerator, denominator = policy.sum(), baseline.sum()
        ratio = numerator / denominator
    if np.isfinite(numerator) and np.isfinite(denominator) and np.isfinite(ratio):
        if ratio > 0 or not np.any(policy > 0):
            return float(ratio)
    # Rare exceptional path: exact binary-float sums avoid overflowing totals
    # or underflowing scaled terms. Float conversion rounds only the final ratio.
    exact_policy = sum((Fraction.from_float(float(x)) for x in policy), Fraction(0))
    exact_baseline = sum((Fraction.from_float(float(x)) for x in baseline), Fraction(0))
    exact_ratio = exact_policy / exact_baseline
    try:
        ratio = float(exact_ratio)
    except OverflowError as exc:
        raise ValueError("Cost index is outside the nonzero finite float64 range") from exc
    if not np.isfinite(ratio) or (exact_ratio > 0 and ratio == 0):
        raise ValueError("Cost index is outside the nonzero finite float64 range")
    return ratio
