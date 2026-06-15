"""Noise-map utilities."""

import numpy as np


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
    data,
    sigma: float = 1.0,
) -> np.ndarray:
    """Estimate a simple constant noise map."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    data = np.asarray(
        data,
        dtype=float,
    )

    return make_constant_noise_map(
        data.shape,
        sigma=sigma,
    )