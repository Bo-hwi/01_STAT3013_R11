"""Pure quantile metrics and prediction constraints; no model fitting or I/O."""

from numbers import Real

import numpy as np


QUANTILE_LEVELS = (0.50, 0.75, 5 / 6)
_TAU_KEYS = ("tau_0.50", "tau_0.75", "tau_5_6")


def _finite_array(values, name):
    """Reject masks and complex data BEFORE float conversion can lose them."""
    if np.ma.isMaskedArray(values):
        raise ValueError(f"{name} must not be a masked array")
    try:
        raw = np.asarray(values)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must contain real values, not complex values")
        if raw.dtype.kind == "O":
            for value in raw.flat:
                if np.ma.isMaskedArray(value) or isinstance(value, (complex, np.complexfloating)):
                    raise ValueError(f"{name} must not contain masked or complex values")
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real values without masks or complex values") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real values")
    return array


def _finite_vector(values, name):
    array = _finite_array(values, name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return array


def _paired_vectors(first, second):
    first = _finite_vector(first, "first input")
    second = _finite_vector(second, "second input")
    if first.shape != second.shape:
        raise ValueError("Inputs must have identical shapes; broadcasting is forbidden")
    return first, second


def _real_scalar(value, name):
    """Accept Python/NumPy real numbers, excluding bool and array containers."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar (not bool, complex, string or array)")
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return value


def _validate_tau(tau):
    tau = _real_scalar(tau, "tau")
    if not 0 < tau < 1:
        raise ValueError("tau must be finite and strictly between 0 and 1")
    return tau


def normalize_tau(tau):
    """Map a supported tau to 0.5, 0.75 or exact float 5/6.

    Absolute tolerance 1e-9 accepts the existing YAML's 0.8333333333, but
    deliberately rejects coarse display rounding such as 0.8333. Generic
    pinball/coverage metrics still accept any tau strictly inside (0, 1).
    """
    tau = _validate_tau(tau)
    for level in QUANTILE_LEVELS:
        if np.isclose(tau, level, rtol=0, atol=1e-9):
            return level
    raise ValueError("tau must match a project level: 0.50, 0.75 or 5/6 (atol=1e-9)")


def canonical_tau_key(tau):
    """Return a stable string key for joins; normalize BOTH sides before joining."""
    normalized = normalize_tau(tau)
    return _TAU_KEYS[QUANTILE_LEVELS.index(normalized)]


def pinball_loss(y_true, quantile_prediction, tau):
    """Return loss PER OBSERVATION: tau*u for u>=0, (tau-1)*u otherwise.

    Here u = y_true - quantile_prediction. No clipping or averaging is implicit;
    use the final constrained Q when comparing this loss with Newsvendor cost.
    Empty paired inputs return an empty array. Inputs are never modified.
    Raise ValueError on overflow or a nonzero loss rounded to zero (underflow).
    """
    tau = _validate_tau(tau)
    truth, prediction = _paired_vectors(y_true, quantile_prediction)
    weight = np.where(truth >= prediction, tau, 1 - tau)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        residual = truth - prediction
        loss = weight * np.abs(residual)
        # Opposite-sign extremes can overflow subtraction even when the
        # weighted difference fits float64. Scale operands before subtracting.
        overflow = ~np.isfinite(residual)
        loss[overflow] = np.abs(
            weight[overflow] * truth[overflow] - weight[overflow] * prediction[overflow]
        )
    if not np.isfinite(loss).all() or np.any((truth != prediction) & (loss == 0)):
        raise ValueError("Pinball loss is outside the nonzero finite float64 range")
    return loss


def quantile_coverage(y_true, quantile_prediction):
    """Empirical P(y_true <= prediction), counting ties as covered.

    Return NaN for no observations (undefined denominator); do not drop NaNs
    silently. This is a diagnostic, not a calibration guarantee.
    """
    truth, prediction = _paired_vectors(y_true, quantile_prediction)
    return float(np.mean(truth <= prediction)) if truth.size else float("nan")


def coverage_gap(y_true, quantile_prediction, tau):
    """Signed coverage minus tau; positive means coverage exceeds the target."""
    tau = _validate_tau(tau)
    return quantile_coverage(y_true, quantile_prediction) - tau


def nonnegative_quantity(quantity):
    """Clip finite quantities at zero, preserving fractional sales units."""
    quantity = _finite_array(quantity, "Quantity")
    return np.maximum(quantity, 0.0)


def enforce_quantile_order(predictions, taus=QUANTILE_LEVELS):
    """Clip at zero and sort each row into ascending quantile order.

    Columns must correspond to strictly increasing taus. Return a new matrix;
    callers should measure crossing on raw predictions before this correction.
    No labels are used. Sorting does not guarantee improved calibration.
    """
    levels = _finite_vector(taus, "taus")
    if not levels.size or np.any((levels <= 0) | (levels >= 1)) or np.any(np.diff(levels) <= 0):
        raise ValueError("taus must be nonempty, strictly increasing and inside (0, 1)")
    values = _finite_array(predictions, "Predictions")
    if values.ndim != 2 or values.shape[1] != levels.size:
        raise ValueError("Predictions must have shape (n_observations, len(taus))")
    return np.sort(nonnegative_quantity(values), axis=1)
