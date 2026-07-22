"""Scientific metrics used by BayMorph validation experiments."""

from __future__ import annotations

import numpy as np


def parameter_bias(estimate: float, truth: float) -> float:
    """Return signed parameter bias, ``estimate - truth``."""
    return float(estimate - truth)


def fractional_error(estimate: float, truth: float) -> float:
    """Return signed fractional error relative to the true value.

    Raises
    ------
    ValueError
        If the true value is zero.
    """
    if truth == 0:
        raise ValueError("Fractional error is undefined for zero truth.")
    return float((estimate - truth) / truth)


def interval_contains_truth(
    lower: float,
    upper: float,
    truth: float,
) -> bool:
    """Return whether a closed posterior interval contains the truth.

    Raises
    ------
    ValueError
        If the interval bounds are reversed.
    """
    if lower > upper:
        raise ValueError("Posterior interval lower bound exceeds upper bound.")
    return bool(lower <= truth <= upper)


def bias(values: np.ndarray, truths: np.ndarray) -> float:
    """Return the mean signed bias across repeated experiments."""
    values, truths = _matching_finite_arrays(values, truths)
    return float(np.mean(values - truths))


def scatter(values: np.ndarray, truths: np.ndarray) -> float:
    """Return RMS scatter of recovery errors around their mean bias."""
    values, truths = _matching_finite_arrays(values, truths)
    errors = values - truths
    return float(np.sqrt(np.mean((errors - np.mean(errors)) ** 2)))


def coverage_fraction(covered: np.ndarray) -> float:
    """Return the fraction of experiments whose interval covers the truth."""
    covered = np.asarray(covered)
    if covered.ndim != 1 or covered.size == 0:
        raise ValueError("Coverage values must be a non-empty one-dimensional array.")
    return float(np.mean(covered.astype(bool)))


def _matching_finite_arrays(
    values: np.ndarray,
    truths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    truths = np.asarray(truths, dtype=float)

    if values.shape != truths.shape:
        raise ValueError("Recovered values and truths must have the same shape.")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Metric inputs must be non-empty one-dimensional arrays.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(truths)):
        raise ValueError("Metric inputs must contain only finite values.")

    return values, truths
