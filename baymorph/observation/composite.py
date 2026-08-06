"""Observed composition of extended and unresolved source components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from baymorph.models.convolved import ConvolvedModel
from baymorph.models.point_source import PointSourceModel
from baymorph.observation.point_source import PSFPointSourceRenderer

FloatArray = NDArray[np.float64]


class ObservedCompositeModel:
    """Combine PSF-convolved extended morphology and point sources.

    Parameters
    ----------
    extended_model : object, optional
        Intrinsic extended morphology exposing ``render()`` and optionally
        ``render_components()``.
    point_sources : mapping of str to PointSourceModel, optional
        Named unresolved intrinsic source components.
    psf : object
        Observation PSF shared by all source components.
    kernel_size : int, default=25
        Kernel size used for analytic Gaussian PSFs.

    Notes
    -----
    Extended morphology is rendered on a PSF-extended domain before
    convolution. Point sources are rendered directly through the same PSF.
    This avoids representing Dirac distributions as pixelized intrinsic
    images and avoids convolving point sources twice.
    """

    def __init__(
        self,
        *,
        extended_model: Any | None = None,
        point_sources: Mapping[str, PointSourceModel] | None = None,
        psf: Any,
        kernel_size: int = 25,
    ) -> None:
        if extended_model is None and not point_sources:
            raise ValueError(
                "ObservedCompositeModel requires at least one source "
                "component."
            )

        self.extended_model = extended_model
        self.point_sources = dict(point_sources or {})

        for name, source in self.point_sources.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "Point-source component names must be non-empty strings."
                )

            if not isinstance(source, PointSourceModel):
                raise TypeError(
                    f"point-source component '{name}' must be a "
                    "PointSourceModel."
                )

        self.psf = psf
        self.kernel_size = int(kernel_size)

        self.extended_renderer = (
            ConvolvedModel(
                model=extended_model,
                psf=psf,
                kernel_size=kernel_size,
            )
            if extended_model is not None
            else None
        )

        self.point_source_renderer = PSFPointSourceRenderer(
            psf,
            kernel_size=kernel_size,
        )

    @classmethod
    def from_sersic_point_source(
        cls,
        morphology,
        *,
        psf: Any,
        kernel_size: int = 25,
    ) -> "ObservedCompositeModel":
        """Build an observed Sérsic plus point-source model.

        Parameters
        ----------
        morphology : SersicPointSourceModel
            Intrinsic named morphology containing one extended Sérsic
            component and one unresolved central source.
        psf : object
            Observation PSF shared by both source components.
        kernel_size : int, default=25
            Kernel size used for analytic Gaussian PSFs.

        Returns
        -------
        ObservedCompositeModel
            Observation-level model with independently correct treatment of
            the extended and unresolved components.
        """
        from baymorph.models.sersic_point_source import (
            SersicPointSourceModel,
        )

        if not isinstance(
            morphology,
            SersicPointSourceModel,
        ):
            raise TypeError(
                "from_sersic_point_source() requires a "
                "SersicPointSourceModel."
            )

        return cls(
            extended_model=morphology.sersic,
            point_sources={
                "point_source": morphology.point_source,
            },
            psf=psf,
            kernel_size=kernel_size,
        )

    def render_components(
        self,
        shape: tuple[int, int],
    ) -> dict[str, FloatArray]:
        """Render every named observed source component."""
        components: dict[str, FloatArray] = {}

        if self.extended_renderer is not None:
            try:
                extended_components = (
                    self.extended_renderer.render_components(
                        shape=shape,
                    )
                )
            except TypeError:
                extended_components = {
                    "extended": self.extended_renderer.render(
                        shape=shape,
                    )
                }

            components.update(extended_components)

        for name, source in self.point_sources.items():
            if name in components:
                raise ValueError(
                    f"duplicate observed component name '{name}'"
                )

            components[name] = self.point_source_renderer.render(
                source,
                shape=shape,
            )

        return components

    def render(
        self,
        shape: tuple[int, int],
    ) -> FloatArray:
        """Render the total expected observed source image."""
        image = np.zeros(
            shape,
            dtype=float,
        )

        for component in self.render_components(
            shape=shape,
        ).values():
            image += component

        return image
