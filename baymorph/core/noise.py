"""Noise-map utilities."""

import numpy as np
from astropy.stats import sigma_clip

from baymorph.stats.robust import robust_sigma


def make_constant_noise_map(
    shape,
    sigma: float,
) -> np.ndarray:
    """Create a constant noise map.

    Parameters
    ----------
    shape : tuple of int
        Shape of the output noise map.
    sigma : float
        Constant noise level assigned to every pixel.

    Returns
    -------
    ndarray
        Constant noise map with the requested shape.

    Raises
    ------
    ValueError
        If ``sigma`` is not strictly positive.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    return np.full(
        shape,
        sigma,
        dtype=float,
    )


def estimate_noise_map(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    clip_sigma: float = 3.0,
    maxiters: int = 5,
) -> np.ndarray:
    """Estimate a constant noise map from an image.

    The noise level is a Gaussian-equivalent robust scale: finite, optionally
    masked pixels are iteratively sigma-clipped, and the scale is then the
    scaled median absolute deviation of the surviving pixels. The resulting
    scalar estimate is returned as a constant-valued noise map with the same
    shape as the input image.

    Notes
    -----
    Clipping is retained deliberately. A scaled MAD alone is less biased on
    clean Gaussian pixels, but it overestimates the scale when unmasked source
    pixels remain, whereas the clipped MAD stays stable in both regimes.

    Clipping truncates the Gaussian tails, so the surviving sample is slightly
    narrower than the parent distribution. The estimator therefore keeps a
    residual low bias of roughly 0.4% at the default ``clip_sigma``. This is a
    known and bounded limitation of the prepared noise field, not an invisible
    implementation detail; removing it would require an explicit truncation
    correction.

    The prepared noise field also excludes source Poisson variance and any
    external variance map.

    Parameters
    ----------
    image : ndarray
        Input image.
    mask : ndarray of bool, optional
        Boolean mask identifying pixels to exclude from the noise estimate.
    clip_sigma : float, default=3.0
        Sigma-clipping threshold.
    maxiters : int, default=5
        Maximum number of sigma-clipping iterations.

    Returns
    -------
    ndarray
        Constant noise map with the same shape as ``image``.
    """
    data = np.asarray(image, dtype=float)

    valid = np.isfinite(data)

    if mask is not None:
        valid &= ~np.asarray(mask, dtype=bool)

    values = data[valid]

    if values.size == 0:
        sigma = robust_sigma(data)
    else:
        clipped = sigma_clip(
            values,
            sigma=clip_sigma,
            maxiters=maxiters,
            masked=True,
        )
        sigma = robust_sigma(
            clipped.compressed()
        )

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = robust_sigma(data)

    return np.full_like(data, sigma, dtype=float)
