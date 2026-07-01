"""Noise-map utilities."""

import numpy as np
from astropy.stats import sigma_clip


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
    sigma: float | None = None,
) -> np.ndarray:
    """Estimate a constant noise map from an image.

    The noise level is estimated from finite, optionally masked pixels using
    iterative sigma clipping. The resulting scalar estimate is returned as a
    constant-valued noise map with the same shape as the input image.

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
    sigma : float, optional
        Backward-compatible alias for ``clip_sigma``. If provided, it takes
        precedence over ``clip_sigma``.

    Returns
    -------
    ndarray
        Constant noise map with the same shape as ``image``.
    """
    if sigma is not None:
        clip_sigma = sigma

    data = np.asarray(image, dtype=float)

    valid = np.isfinite(data)

    if mask is not None:
        valid &= ~np.asarray(mask, dtype=bool)

    values = data[valid]

    if values.size == 0:
        sigma = np.nanstd(data)
    else:
        clipped = sigma_clip(
            values,
            sigma=clip_sigma,
            maxiters=maxiters,
            masked=True,
        )
        sigma = np.nanstd(clipped.compressed())

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.nanstd(data)

    return np.full_like(data, sigma, dtype=float)