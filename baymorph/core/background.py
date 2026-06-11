"""Background and noise estimation utilities."""

from dataclasses import dataclass

import numpy as np

from photutils.background import (
    Background2D,
    MedianBackground,

)

@dataclass
class BackgroundEstimate:
    """Background and noise estimate."""

    background: float
    sigma: float

    background_map: np.ndarray | None = None
    rms_map: np.ndarray | None = None


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


def estimate_background_photutils(
    data: np.ndarray,
    box_size: int = 20,
    filter_size: int = 3,
) -> BackgroundEstimate:
    """
    Estimate background using Photutils Background2D.
    """

    bkg = Background2D(
        data,
        box_size=box_size,
        filter_size=(
            filter_size,
            filter_size,
        ),
        bkg_estimator=MedianBackground(),
    )

    return BackgroundEstimate(
        background=float(
            np.median(
                bkg.background
            )
        ),
        sigma=float(
            np.median(
                bkg.background_rms
            )
        ),
        background_map=bkg.background.copy(),
        rms_map=bkg.background_rms.copy(),
    )