"""Cutouts of empirical-PSF candidates and their local signal-to-noise.

These helpers are shared by the selection of the PSF candidates, by their
point-source screen and by the construction of the PSF, so they live in a
module that imports none of them.
"""

from __future__ import annotations

import numpy as np

from baymorph.core.image import WorkingRegion
from baymorph.stats.robust import robust_sigma


def region_hint(region: WorkingRegion | None) -> str:
    """Say where the stars were searched, when the image has a working region."""
    if region is None:
        return ""

    return (
        " The stars are searched within the working region x "
        f"{region.xmin}-{region.xmax}, y {region.ymin}-{region.ymax} of the "
        "image: a larger data.region, or data.region: whole, offers more."
    )


def extract_centered_cutout(
    image: np.ndarray, x: float, y: float, half_size: int
) -> np.ndarray:
    """Extract a square cutout centered on a source position.

    Parameters
    ----------
    image : ndarray
        Parent image.
    x, y : float
        Source position in image pixels.
    half_size : int
        Half-size of the requested cutout.

    Returns
    -------
    ndarray
        Extracted square cutout.

    Raises
    ------
    ValueError
        If the requested cutout extends outside the image boundaries.
    """
    x0 = int(round(x))
    y0 = int(round(y))

    y_min = y0 - half_size
    y_max = y0 + half_size + 1
    x_min = x0 - half_size
    x_max = x0 + half_size + 1

    if x_min < 0 or y_min < 0 or x_max > image.shape[1] or y_max > image.shape[0]:
        raise ValueError("Requested PSF cutout extends outside the image boundaries.")

    return image[y_min:y_max, x_min:x_max]


def _outer_border_pixels(cutout: np.ndarray, *, border_width: int) -> np.ndarray:
    """Return pixels in the outer border of a two-dimensional cutout."""
    data = np.asarray(cutout, dtype=float)

    if data.ndim != 2:
        raise ValueError("PSF cutout must be 2D.")

    if border_width <= 0:
        raise ValueError("border_width must be positive.")

    if 2 * border_width >= min(data.shape):
        raise ValueError("border_width is too large for this cutout.")

    return np.concatenate(
        [
            data[:border_width, :].ravel(),
            data[-border_width:, :].ravel(),
            data[:, :border_width].ravel(),
            data[:, -border_width:].ravel(),
        ]
    )


def estimate_cutout_peak_snr(
    cutout: np.ndarray, *, border_width: int = 3
) -> float | None:
    """Estimate robust local peak signal-to-noise for a PSF candidate.

    The local background and noise are estimated from the outer cutout
    border. Noise is represented by the Gaussian-equivalent MAD scale.

    Parameters
    ----------
    cutout : ndarray
        Candidate image cutout before background subtraction.
    border_width : int, default=3
        Width of the outer region used for background and noise estimation.

    Returns
    -------
    float or None
        Peak signal-to-noise ratio, or ``None`` when a finite positive
        local noise scale cannot be measured.
    """
    data = np.asarray(cutout, dtype=float)

    border = _outer_border_pixels(data, border_width=border_width)

    finite_border = border[np.isfinite(border)]

    if finite_border.size == 0:
        return None

    background = float(np.median(finite_border))

    local_sigma = robust_sigma(finite_border, center=background)

    if not np.isfinite(local_sigma) or local_sigma <= 0.0:
        return None

    finite_data = data[np.isfinite(data)]

    if finite_data.size == 0:
        return None

    peak_signal = float(np.max(finite_data) - background)

    if not np.isfinite(peak_signal) or peak_signal <= 0.0:
        return None

    return float(peak_signal / local_sigma)
