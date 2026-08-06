"""Composite intrinsic morphological models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from baymorph.models.protocols import MorphologicalComponent

FloatArray = NDArray[np.float64]


class CompositeModel:
    """Additive model formed from named morphological components.

    Parameters
    ----------
    components : mapping or iterable of MorphologicalComponent
        Intrinsic components to be summed. A mapping preserves the supplied
        names. Components supplied as an iterable use their ``name`` attribute
        when available and otherwise receive names of the form
        ``component_<index>``.
    sky : float, default=0.0
        Constant additive background level.

    Notes
    -----
    ``CompositeModel`` performs intrinsic additive composition. PSF
    convolution and likelihood evaluation remain outside this class.
    """

    def __init__(
        self,
        components: (
            Mapping[str, MorphologicalComponent]
            | Iterable[MorphologicalComponent]
        ),
        *,
        sky: float = 0.0,
    ) -> None:
        self.components: dict[str, MorphologicalComponent] = {}

        if isinstance(components, Mapping):
            for name, component in components.items():
                self._add_component(
                    name=name,
                    component=component,
                )
        else:
            for index, component in enumerate(components):
                name = getattr(
                    component,
                    "name",
                    f"component_{index}",
                )
                self._add_component(
                    name=name,
                    component=component,
                )

        self.sky = float(sky)

    def _add_component(
        self,
        *,
        name: str,
        component: MorphologicalComponent,
    ) -> None:
        """Validate and register one named component."""
        if not isinstance(name, str) or not name:
            raise ValueError(
                "Composite component names must be non-empty strings."
            )

        if name in self.components:
            raise ValueError(
                f"duplicate component name '{name}'"
            )

        if not isinstance(component, MorphologicalComponent):
            raise TypeError(
                f"component '{name}' does not implement render()."
            )

        self.components[name] = component

    def render_components(
        self,
        shape: tuple[int, int],
        *,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> dict[str, FloatArray]:
        """Render every named intrinsic component."""
        return {
            name: np.asarray(
                component.render(
                    shape=shape,
                    origin=origin,
                ),
                dtype=float,
            )
            for name, component in self.components.items()
        }

    def render(
        self,
        shape: tuple[int, int],
        *,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> FloatArray:
        """Render the total intrinsic composite model."""
        image = np.zeros(shape, dtype=float)

        for component_image in self.render_components(
            shape=shape,
            origin=origin,
        ).values():
            image += component_image

        if self.sky != 0.0:
            image += self.sky

        return image
