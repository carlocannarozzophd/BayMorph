"""Synthetic Sersic image generation."""

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaincinv


@dataclass
class SyntheticSersic:
    """Generate a synthetic single-component Sersic galaxy."""

    flux: float
    re: float
    n: float
    q: float
    theta: float
    x0: float
    y0: float

    def render(self, shape: tuple[int, int]) -> np.ndarray:
        """Render the Sersic model on a pixel-centered grid."""
        ny, nx = shape
        y, x = np.mgrid[0:ny, 0:nx]

        dx = x - self.x0
        dy = y - self.y0

        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)

        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t

        radius = np.sqrt(x_rot**2 + (y_rot / self.q) ** 2)

        bn = gammaincinv(2.0 * self.n, 0.5)

        intensity = np.exp(-bn * ((radius / self.re) ** (1.0 / self.n) - 1.0))

        norm = intensity.sum()

        if norm <= 0:
            raise ValueError("Sersic model has non-positive normalization.")

        return self.flux * intensity / norm