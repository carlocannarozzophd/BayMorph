"""Robust summary statistics."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def robust_rms(values):
    """Compute the root-mean-square of finite values.

    Parameters
    ----------
    values : array-like
        Input values.

    Returns
    -------
    float
        Root-mean-square computed after discarding non-finite values. Returns
        ``NaN`` if no finite values are available.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    return np.sqrt(
        np.nanmean(values**2)
    )


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


def robust_mad(values):
    """Compute the median absolute deviation of finite values.

    Parameters
    ----------
    values : array-like
        Input values.

    Returns
    -------
    float
        Median absolute deviation computed after discarding non-finite values.
        Returns ``NaN`` if no finite values are available.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    median = np.nanmedian(values)

    return np.nanmedian(
        np.abs(values - median)
    )