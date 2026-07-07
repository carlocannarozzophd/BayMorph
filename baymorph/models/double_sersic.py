"""Double-Sersic morphology model."""

from dataclasses import dataclass

import numpy as np


@dataclass
class DoubleSersicModel:
    """Two-component Sérsic model.

    Parameters
    ----------
    bulge : object
        Inner Sérsic-like component exposing a ``render(shape)`` method.
    disk : object
        Outer Sérsic-like component exposing a ``render(shape)`` method.
    sky : float, optional
        Constant sky background added to the total rendered model.
    """

    bulge: object
    disk: object
    sky: float = 0.0

    def render_components(
        self,
        shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        """Render the bulge and disk components.

        Parameters
        ----------
        shape : tuple of int
            Output image shape as ``(ny, nx)``.

        Returns
        -------
        dict of str, ndarray
            Rendered model components.
        """
        return {
            "bulge": self.bulge.render(shape=shape),
            "disk": self.disk.render(shape=shape),
        }

    def render(
        self,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Render the sum of the bulge, disk, and sky components.

        Parameters
        ----------
        shape : tuple of int
            Output image shape as ``(ny, nx)``.

        Returns
        -------
        ndarray
            Rendered double-Sérsic image.
        """
        components = self.render_components(shape=shape)
        return components["bulge"] + components["disk"] + self.sky