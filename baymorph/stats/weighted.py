"""Weighted posterior statistics."""

import numpy as np


def dynesty_weights(
    logwt: np.ndarray,
    logz: np.ndarray,
) -> np.ndarray:
    """Convert Dynesty weights to normalized posterior weights."""
    weights = np.exp(
        logwt - logz[-1]
    )

    weights /= np.sum(weights)

    return weights


def weighted_quantile(
    values: np.ndarray,
    quantile: float,
    weights: np.ndarray,
) -> float:
    """Compute weighted quantiles."""
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
    """Return median, q16 and q84."""
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