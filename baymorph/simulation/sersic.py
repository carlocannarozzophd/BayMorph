"""Synthetic Sersic image generation."""

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaincinv


@dataclass
class SyntheticSersic:
    """Synthetic single-component Sérsic galaxy model.

    Parameters
    ----------
    flux : float
        Total integrated flux of the model.
    re : float
        Effective radius, in pixels.
    n : float
        Sérsic index.
    q : float
        Projected axis ratio.
    theta : float
        Position angle, in radians.
    x0 : float
        X-coordinate of the model center, in pixels.
    y0 : float
        Y-coordinate of the model center, in pixels.
    """

    flux: float
    re: float
    n: float
    q: float
    theta: float
    x0: float
    y0: float

    def render(self, shape: tuple[int, int]) -> np.ndarray:
        """Render the Sérsic model on a pixel-centered grid.

        Parameters
        ----------
        shape : tuple of int
            Output image shape as ``(ny, nx)``.

        Returns
        -------
        ndarray
            Rendered image normalized to the requested total flux.

        Raises
        ------
        ValueError
            If the rendered model has a non-positive normalization.
        """
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

        intensity = np.exp(
            -bn * ((radius / self.re) ** (1.0 / self.n) - 1.0)
        )

        norm = intensity.sum()

        if norm <= 0:
            raise ValueError("Sersic model has non-positive normalization.")

        return self.flux * intensity / norm