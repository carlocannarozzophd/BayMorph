"""Composite morphological models."""

import numpy as np


class CompositeModel:
    """Sum of multiple renderable models."""

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
        """Render all components and sum them."""
        image = np.zeros(
            shape,
            dtype=float,
        )

        for component in self.components:
            image += component.render(
                shape=shape,
            )

        return image