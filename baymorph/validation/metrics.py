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


def weighted_posterior_rank(
    samples: np.ndarray,
    weights: np.ndarray,
    truth: float,
) -> float:
    """Return the posterior probability mass below the injected truth.

    The returned value lies in [0, 1]. Under repeated simulation from the
    same generative model and prior, calibrated posterior ranks are expected
    to be uniformly distributed.

    Parameters
    ----------
    samples : numpy.ndarray
        One-dimensional posterior samples.
    weights : numpy.ndarray
        Non-negative posterior weights corresponding to ``samples``.
    truth : float
        Injected parameter value.
    """
    samples, weights = _normalized_weighted_samples(samples, weights)

    if not np.isfinite(truth):
        raise ValueError("Truth must be finite.")

    return float(np.sum(weights[samples < truth]))


def weighted_credible_interval(
    samples: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> tuple[float, float]:
    """Return an equal-tailed weighted posterior credible interval.

    Parameters
    ----------
    samples : numpy.ndarray
        One-dimensional posterior samples.
    weights : numpy.ndarray
        Non-negative posterior weights corresponding to ``samples``.
    probability : float
        Central posterior probability contained by the interval.
    """
    samples, weights = _normalized_weighted_samples(samples, weights)

    if not np.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("Credible probability must lie strictly between zero and one.")

    order = np.argsort(samples)
    ordered_samples = samples[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)

    tail = 0.5 * (1.0 - probability)

    lower = np.interp(
        tail,
        cumulative,
        ordered_samples,
    )
    upper = np.interp(
        1.0 - tail,
        cumulative,
        ordered_samples,
    )

    return float(lower), float(upper)


def _normalized_weighted_samples(
    samples: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Posterior samples must be a non-empty one-dimensional array.")

    if weights.shape != samples.shape:
        raise ValueError("Posterior samples and weights must have the same shape.")

    if not np.all(np.isfinite(samples)):
        raise ValueError("Posterior samples must contain only finite values.")

    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Posterior weights must be finite and non-negative.")

    total = float(np.sum(weights))

    if total <= 0.0:
        raise ValueError("Posterior weights must have positive total mass.")

    return samples, weights / total
