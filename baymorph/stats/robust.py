"""Robust summary statistics."""

from __future__ import annotations

import numpy as np


def robust_rms(values):
    """Return the RMS after discarding non-finite values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    return np.sqrt(
        np.nanmean(values**2)
    )


def robust_mad(values):
    """Return the median absolute deviation after discarding non-finite values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    median = np.nanmedian(values)

    return np.nanmedian(
        np.abs(values - median)
    )