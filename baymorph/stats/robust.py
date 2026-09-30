"""Robust summary statistics."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


MAD_TO_SIGMA = 1.0 / norm.ppf(0.75)
"""Scale factor converting a median absolute deviation to a Gaussian sigma."""


def robust_sigma(values, *, center=None):
    """Compute a robust Gaussian-equivalent sigma from a scaled MAD.

    Parameters
    ----------
    values : array-like
        Input values.
    center : float, optional
        Reference center for the absolute deviations. Defaults to the
        median of the finite values.

    Returns
    -------
    float
        ``MAD_TO_SIGMA`` times the median absolute deviation from
        ``center``, computed after discarding non-finite values. Returns
        ``NaN`` if no finite values are available.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    if center is None:
        center = np.nanmedian(values)

    return float(
        MAD_TO_SIGMA * np.nanmedian(np.abs(values - center))
    )


def truncated_variance_ratio(
    relative_threshold: float,
) -> float:
    """Return the variance ratio of a symmetrically truncated Gaussian.

    Parameters
    ----------
    relative_threshold : float
        Truncation threshold in units of the parent standard deviation.

    Returns
    -------
    float
        Ratio of the truncated variance to the parent variance, in ``(0, 1]``.

    Notes
    -----
    For a Gaussian truncated symmetrically at ``+/- c`` standard deviations::

        var_truncated / var_parent = 1 - 2 c phi(c) / (2 Phi(c) - 1)

    where ``phi`` and ``Phi`` are the standard normal density and cumulative
    distribution. The ratio tends to one as the threshold grows.
    """
    if not np.isfinite(relative_threshold):
        return 1.0

    if relative_threshold <= 0.0:
        raise ValueError(
            "relative_threshold must be positive."
        )

    mass = 2.0 * norm.cdf(
        relative_threshold
    ) - 1.0

    if mass <= 0.0:
        raise ValueError(
            "relative_threshold leaves no probability mass."
        )

    return float(
        1.0
        - 2.0
        * relative_threshold
        * norm.pdf(
            relative_threshold
        )
        / mass
    )


def truncated_gaussian_sigma(
    values,
    threshold: float,
    *,
    center=None,
) -> float:
    """Estimate the parent Gaussian sigma of a symmetrically clipped sample.

    Parameters
    ----------
    values : array-like
        Sample that survived symmetric clipping about ``center``.
    threshold : float
        Half-width of the clipping interval, in data units. Use ``inf`` when
        no clipping was applied.
    center : float, optional
        Centre of the clipping interval. Defaults to the sample median.

    Returns
    -------
    float
        Estimated standard deviation of the parent distribution. Returns
        ``NaN`` if no finite values are available.

    Raises
    ------
    ValueError
        If ``threshold`` is not positive, or if the correction fails to
        converge to a positive finite scale.

    Notes
    -----
    Clipping truncates the tails, so the standard deviation of the surviving
    sample estimates the scale of the truncated distribution rather than of
    the parent. This routine inverts that relation.

    The relation is solved rather than applied as a fixed factor, because an
    iterative clipping loop re-estimates its threshold from the already
    clipped sample: the effective truncation point is therefore not exactly
    the nominal number of parent standard deviations.

    The parent distribution is assumed Gaussian. Clipping exists to make the
    surviving core approximately Gaussian, but on data with residual sources
    or gradients that remains an assumption rather than a guarantee.
    """
    values = np.asarray(
        values,
        dtype=float,
    )
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    observed = float(
        np.std(values)
    )

    if observed <= 0.0:
        return observed

    if threshold is None or not np.isfinite(threshold):
        return observed

    threshold = float(threshold)

    if threshold <= 0.0:
        raise ValueError(
            "threshold must be positive."
        )

    if center is None:
        center = float(
            np.median(values)
        )

    # Fixed-point iteration on sigma = observed / sqrt(ratio(threshold / sigma)).
    # The map is a contraction here because the ratio varies slowly with sigma
    # at the thresholds used for clipping.
    sigma = observed

    for _ in range(100):
        ratio = truncated_variance_ratio(
            threshold / sigma
        )

        if ratio <= 0.0:
            raise ValueError(
                "Truncation correction produced a non-positive variance ratio."
            )

        updated = observed / np.sqrt(
            ratio
        )

        if abs(updated - sigma) <= 1e-12 * max(
            1.0,
            abs(sigma),
        ):
            sigma = updated
            break

        sigma = updated

    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(
            "Truncation correction did not converge to a positive scale."
        )

    return float(sigma)
