"""Background and noise estimation utilities."""

from dataclasses import dataclass

import numpy as np
from photutils.background import Background2D, MedianBackground


@dataclass
class BackgroundEstimate:
    """Container for background and noise estimation results.

    Parameters
    ----------
    background : float
        Estimated scalar background level.
    sigma : float
        Estimated scalar background root-mean-square (RMS) noise.
    background_map : ndarray, optional
        Two-dimensional background model. This is available when using
        algorithms that estimate a spatially varying background.
    rms_map : ndarray, optional
        Two-dimensional background RMS map corresponding to
        ``background_map``.
    """

    background: float
    sigma: float
    background_map: np.ndarray | None = None
    rms_map: np.ndarray | None = None


def estimate_background_sigma(
    data: np.ndarray,
    sigma_clip: float = 3.0,
    maxiters: int = 5,
) -> BackgroundEstimate:
    """Estimate a scalar background and noise level using sigma clipping.

    Parameters
    ----------
    data : ndarray
        Input image.
    sigma_clip : float, default=3.0
        Sigma-clipping threshold applied during the iterative estimation.
    maxiters : int, default=5
        Maximum number of sigma-clipping iterations.

    Returns
    -------
    BackgroundEstimate
        Estimated scalar background level and RMS noise.

    Raises
    ------
    ValueError
        If the input image contains no finite values.
    """
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


def estimate_background_photutils(
    data: np.ndarray,
    box_size: int = 20,
    filter_size: int = 3,
    mask: np.ndarray | None = None,
    exclude_percentile: float = 10.0,
) -> BackgroundEstimate:
    """Estimate a spatially varying background using Photutils.

    Parameters
    ----------
    data : ndarray
        Input image.
    box_size : int, default=20
        Size of the background mesh used by
        :class:`photutils.background.Background2D`.
    filter_size : int, default=3
        Size of the median filter applied to the background meshes.
    mask : ndarray of bool, optional
        Boolean mask identifying pixels to ignore during the background
        estimation.
    exclude_percentile : float, default=10.0
        Maximum fraction of masked pixels allowed within each background
        mesh.

    Returns
    -------
    BackgroundEstimate
        Background estimate containing both scalar summary values and the
        full background and RMS maps.
    """
    bkg = Background2D(
        data,
        box_size=box_size,
        filter_size=(
            filter_size,
            filter_size,
        ),
        mask=mask,
        exclude_percentile=exclude_percentile,
        bkg_estimator=MedianBackground(),
    )

    return BackgroundEstimate(
        background=float(np.median(bkg.background)),
        sigma=float(np.median(bkg.background_rms)),
        background_map=bkg.background.copy(),
        rms_map=bkg.background_rms.copy(),
    )


def subtract_background(
    data: np.ndarray,
    background: float,
) -> np.ndarray:
    """Subtract a scalar background level from an image.

    Parameters
    ----------
    data : ndarray
        Input image.
    background : float
        Scalar background level to subtract.

    Returns
    -------
    ndarray
        Background-subtracted image.
    """
    return np.asarray(data, dtype=float) - background