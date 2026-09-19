"""Weighted posterior statistics."""

import numpy as np


def _normalized_weights(
    weights: np.ndarray,
    expected_size: int,
) -> np.ndarray:
    """Return validated weights normalized to unit sum."""
    weights = np.asarray(
        weights,
        dtype=float,
    )

    if weights.shape != (expected_size,):
        raise ValueError(
            "weights must be one-dimensional and match the number of samples."
        )

    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "weights must be finite."
        )

    if np.any(weights < 0):
        raise ValueError(
            "weights must be non-negative."
        )

    total = float(
        weights.sum()
    )

    if total <= 0:
        raise ValueError(
            "weights must have a positive sum."
        )

    return weights / total


def weighted_covariance(
    samples: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute the weighted posterior covariance matrix.

    Parameters
    ----------
    samples : ndarray
        Posterior samples with shape ``(n_samples, n_parameters)``.
    weights : ndarray
        Non-negative sample weights. They are normalized internally.

    Returns
    -------
    ndarray
        Covariance matrix with shape ``(n_parameters, n_parameters)``.

    Raises
    ------
    ValueError
        If the shapes are inconsistent or the weights are invalid.

    Notes
    -----
    This is the weighted second central moment of the posterior. It describes
    the posterior exactly only when the posterior is Gaussian; for a skewed or
    multimodal posterior it remains a summary, not a complete description.
    """
    samples = np.asarray(
        samples,
        dtype=float,
    )

    if samples.ndim != 2:
        raise ValueError(
            "samples must have shape (n_samples, n_parameters)."
        )

    if not np.all(np.isfinite(samples)):
        raise ValueError(
            "samples must be finite."
        )

    normalized = _normalized_weights(
        weights,
        samples.shape[0],
    )

    mean = normalized @ samples
    deviation = samples - mean

    return np.einsum(
        "k,ki,kj->ij",
        normalized,
        deviation,
        deviation,
    )


def mahalanobis_distance_squared(
    samples: np.ndarray,
    weights: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Compute the squared Mahalanobis distance of a reference point.

    The distance measures how far a reference parameter vector lies from the
    posterior while accounting for parameter correlations. Correlated
    parameters must be assessed jointly: marginal per-parameter offsets of a
    correlated posterior double-count one shared displacement.

    Parameters
    ----------
    samples : ndarray
        Posterior samples with shape ``(n_samples, n_parameters)``.
    weights : ndarray
        Non-negative sample weights.
    reference : ndarray
        Reference parameter vector with shape ``(n_parameters,)``.

    Returns
    -------
    float
        Squared Mahalanobis distance.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, the inputs are not finite, or the
        posterior covariance is singular.

    Notes
    -----
    A singular covariance raises rather than falling back to a pseudo-inverse.
    A pseudo-inverted metric returns a plausible number that is not the
    declared quantity, which would hide a fixed or perfectly degenerate
    parameter instead of reporting it.
    """
    samples = np.asarray(
        samples,
        dtype=float,
    )
    reference = np.asarray(
        reference,
        dtype=float,
    )

    if samples.ndim != 2:
        raise ValueError(
            "samples must have shape (n_samples, n_parameters)."
        )

    if reference.shape != (samples.shape[1],):
        raise ValueError(
            "reference must match the number of parameters."
        )

    if not np.all(np.isfinite(reference)):
        raise ValueError(
            "reference must be finite."
        )

    covariance = weighted_covariance(
        samples,
        weights,
    )

    normalized = _normalized_weights(
        weights,
        samples.shape[0],
    )
    delta = (
        normalized @ samples
    ) - reference

    try:
        solved = np.linalg.solve(
            covariance,
            delta,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Posterior covariance is singular, so a Mahalanobis distance is "
            "undefined. This usually means a parameter is fixed or perfectly "
            "degenerate with another."
        ) from exc

    if not np.all(np.isfinite(solved)):
        raise ValueError(
            "Posterior covariance is numerically singular, so a Mahalanobis "
            "distance is undefined."
        )

    return float(
        delta @ solved
    )


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