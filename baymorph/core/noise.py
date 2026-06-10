"""Noise map utilities."""

import numpy as np


def make_constant_noise_map(
    shape: tuple[int, int],
    sigma: float,
) -> np.ndarray:
    """Create a constant noise map."""
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    return np.ones(shape, dtype=float) * sigma


def estimate_noise_map(
    data: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Create a constant noise map with same shape as input data."""
    return make_constant_noise_map(
        shape=data.shape,
        sigma=sigma,
    )