"""Weighted posterior statistics."""

import numpy as np


def weighted_quantile(
    values: np.ndarray,
    quantile: float,
    weights: np.ndarray,
) -> float:
    """Compute a weighted quantile.

    Parameters
    ----------
    values : ndarray
        Sample values.
    quantile : float
        Desired quantile in the interval ``[0, 1]``.
    weights : ndarray
        Non-negative sample weights.

    Returns
    -------
    float
        Weighted quantile.

    Raises
    ------
    ValueError
        If the value and weight arrays have different shapes, if the quantile
        lies outside ``[0, 1]``, if any weight is negative, or if the total
        weight is not positive.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if values.shape != weights.shape:
        raise ValueError("Values and weights must have the same shape.")

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("Quantile must be between 0 and 1.")

    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")

    if np.sum(weights) <= 0:
        raise ValueError("Weights must have positive total weight.")

    weights = weights / np.sum(weights)

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative = np.concatenate(([0.0], cumulative, [1.0]))
    values = np.concatenate(([values[0]], values, [values[-1]]))

    return float(
        np.interp(
            quantile,
            cumulative,
            values,
        )
    )


def posterior_summary(
    values: np.ndarray,
    weights: np.ndarray,
):
    """Compute standard posterior summary statistics.

    Parameters
    ----------
    values : ndarray
        Posterior samples.
    weights : ndarray
        Normalized or unnormalized posterior weights.

    Returns
    -------
    dict
        Dictionary containing the 16th percentile (``q16``), median
        (``median``), and 84th percentile (``q84``).
    """
    return {
        "q16": weighted_quantile(
            values,
            0.16,
            weights,
        ),
        "median": weighted_quantile(
            values,
            0.50,
            weights,
        ),
        "q84": weighted_quantile(
            values,
            0.84,
            weights,
        ),
    }