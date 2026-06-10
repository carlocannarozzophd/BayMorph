"""Noise models for synthetic BayMorph images."""

import numpy as np


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    seed: int | None = None,
) -> np.ndarray:
    """Add Gaussian noise to an image."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape)
    return image + noise


def add_poisson_noise(
    image: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """Add Poisson noise to an image."""
    rng = np.random.default_rng(seed)
    clipped = np.clip(image, 0.0, None)
    return rng.poisson(clipped).astype(float)