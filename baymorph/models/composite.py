"""Composite morphological models."""

from __future__ import annotations

import numpy as np


class CompositeModel:
    """Composite model formed by the sum of multiple renderable components."""

    def __init__(
        self,
        components,
        *,
        sky: float = 0.0,
    ):
        if hasattr(components, "items"):
            self.components = dict(components)
        else:
            self.components = {}

            for index, component in enumerate(components):
                name = getattr(
                    component,
                    "name",
                    f"component_{index}",
                )

                if name in self.components:
                    raise ValueError(
                        f"duplicate component name '{name}'"
                    )

                self.components[name] = component

        self.sky = float(sky)

    def render_components(
        self,
        shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        """Render every model component."""
        return {
            name: component.render(shape=shape)
            for name, component in self.components.items()
        }

    def render(
        self,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Render the composite model."""
        image = np.zeros(shape, dtype=float)

        for component_image in self.render_components(shape).values():
            image += component_image

        if self.sky != 0.0:
            image += self.sky

        return image