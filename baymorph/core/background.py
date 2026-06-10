"""Background and noise estimation utilities."""

from dataclasses import dataclass

import numpy as np


@dataclass
class BackgroundEstimate:
    """Background and noise estimate."""

    background: float
    sigma: float


def estimate_background_sigma(
    data: np.ndarray,
    sigma_clip: float = 3.0,
    maxiters: int = 5,
) -> BackgroundEstimate:
    """Estimate scalar background and noise using iterative sigma clipping."""
    values = np.asarray(data, dtype=float).ravel()
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("Input data contains no finite values.")

    for _ in range(maxiters):
        median = np.median(values)
        std = np.std(values)

        if std == 0:
            break

        keep = np.abs(values - median) < sigma_clip * std

        if keep.sum() == values.size:
            break

        values = values[keep]

    return BackgroundEstimate(
        background=float(np.median(values)),
        sigma=float(np.std(values)),
    )


def subtract_background(
    data: np.ndarray,
    background: float,
) -> np.ndarray:
    """Subtract scalar background from image data."""
    return np.asarray(data, dtype=float) - background