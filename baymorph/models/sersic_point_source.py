"""Intrinsic Sérsic plus unresolved point-source morphology."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baymorph.models.point_source import PointSourceModel
from baymorph.models.sersic import SersicModel


@dataclass(frozen=True)
class SersicPointSourceModel:
    """Intrinsic Sérsic morphology with a central unresolved source.

    Parameters
    ----------
    sersic : SersicModel
        Extended Sérsic component.
    point_source : PointSourceModel
        Unresolved central component.

    Notes
    -----
    The baseline Sérsic plus point-source model assumes a shared centre.
    This is an explicit modelling assumption rather than a requirement of
    either component individually.

    The model owns no PSF. Observation-level rendering must convolve the
    Sérsic component and render the point source through the same observation
    PSF without convolving the point source twice.
    """

    sersic: SersicModel
    point_source: PointSourceModel

    def __post_init__(self) -> None:
        """Validate component types and the shared-centre contract."""
        if not isinstance(self.sersic, SersicModel):
            raise TypeError(
                "SersicPointSourceModel requires a SersicModel."
            )

        if not isinstance(
            self.point_source,
            PointSourceModel,
        ):
            raise TypeError(
                "SersicPointSourceModel requires a PointSourceModel."
            )

        if not np.isclose(
            self.sersic.x0,
            self.point_source.x0,
            rtol=0.0,
            atol=1.0e-12,
        ) or not np.isclose(
            self.sersic.y0,
            self.point_source.y0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "SersicPointSourceModel requires a shared centre."
            )

    @property
    def x0(self) -> float:
        """Return the shared x-coordinate."""
        return float(self.sersic.x0)

    @property
    def y0(self) -> float:
        """Return the shared y-coordinate."""
        return float(self.sersic.y0)

    @property
    def total_flux(self) -> float:
        """Return the total intrinsic source flux."""
        return float(
            self.sersic.flux
            + self.point_source.flux
        )

    @property
    def point_source_fraction(self) -> float:
        """Return the intrinsic unresolved-source flux fraction."""
        return float(
            self.point_source.flux
            / self.total_flux
        )
