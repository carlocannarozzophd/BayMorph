"""Robust summary statistics."""

from __future__ import annotations

import numpy as np


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