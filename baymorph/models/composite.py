"""Composite morphological models."""

import numpy as np


class CompositeModel:
    """Composite model formed by the sum of multiple renderable components.

    Parameters
    ----------
    components : iterable
        Collection of model objects exposing a ``render(shape=...)`` method.
        The rendered image is obtained by summing the contribution of every
        component.
    """

    def __init__(
        self,
        components,
    ):
        self.components = list(
            components
        )

    def render(
        self,
        shape,
    ):
        """Render the composite model.

        Parameters
        ----------
        shape : tuple of int
            Output image shape as ``(ny, nx)``.

        Returns
        -------
        ndarray
            Image obtained by summing the rendered contribution of every
            component.
        """
        image = np.zeros(
            shape,
            dtype=float,
        )

        for component in self.components:
            image += component.render(
                shape=shape,
            )

        return image