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
    """

    bulge: object
    disk: object

    def render(
        self,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Render the sum of the bulge and disk components.

        Parameters
        ----------
        shape : tuple of int
            Output image shape as ``(ny, nx)``.

        Returns
        -------
        ndarray
            Rendered double-Sérsic image.
        """
        return self.bulge.render(shape=shape) + self.disk.render(shape=shape)