"""Composite intrinsic morphological models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from baymorph.models.protocols import MorphologicalComponent

FloatArray = NDArray[np.float64]


def common_centre(components: Iterable) -> tuple[float, float]:
    """Return the centre shared by every component.

    The effective-PSF rendering of AD-046 places one window on the centre of
    a model, so it requires its components to share it.

    Raises
    ------
    TypeError
        If a component does not declare its centre.
    ValueError
        If the components have different centres.
    """
    centres = []

    for component in components:
        if not hasattr(component, "centre"):
            raise TypeError(
                f"{type(component).__name__} does not declare its centre."
            )

        centres.append(tuple(component.centre()))

    if not centres:
        raise ValueError("A model without components has no centre.")

    if any(centre != centres[0] for centre in centres[1:]):
        raise ValueError(
            "The components of this model have different centres; the "
            "effective-PSF rendering of AD-046 needs a single centre."
        )

    return centres[0]


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

    def centre(self) -> tuple[float, float]:
        """Return the centre shared by every component."""
        return common_centre(self.components.values())

    def scaled(self, factor: int) -> "CompositeModel":
        """Return this model on a pixel grid finer by ``factor``.

        Every component is scaled; the constant level is per pixel, so it is
        divided by the number of fine cells in a pixel.
        """
        return CompositeModel(
            {
                name: component.scaled(factor)
                for name, component in self.components.items()
            },
            sky=self.sky / factor**2,
        )

    def render_components(
        self,
        shape: tuple[int, int],
        *,
        origin: tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ) -> dict[str, FloatArray]:
        """Render every named intrinsic component."""
        return {
            name: np.asarray(
                component.render(
                    shape=shape,
                    origin=origin,
                    **kwargs,
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
        **kwargs,
    ) -> FloatArray:
        """Render the total intrinsic composite model."""
        image = np.zeros(shape, dtype=float)

        for component_image in self.render_components(
            shape=shape,
            origin=origin,
            **kwargs,
        ).values():
            image += component_image

        if self.sky != 0.0:
            image += self.sky

        return image
