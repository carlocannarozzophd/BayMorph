"""Noise-map utilities."""

import numpy as np
from astropy.stats import sigma_clip

def make_constant_noise_map(
    shape,
    sigma: float,
) -> np.ndarray:
    """Create a constant noise map."""
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
    """Estimate a constant noise map from source-masked, sigma-clipped pixels."""

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