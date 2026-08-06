"""Intrinsic Sérsic bulge plus exponential disc morphology."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from baymorph.models.composite import CompositeModel
from baymorph.models.exponential import ExponentialComponent
from baymorph.models.sersic import SersicModel

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BulgeDiskModel:
    """Intrinsic Sérsic bulge plus exponential disc model.

    Parameters
    ----------
    bulge : SersicModel
        Sérsic bulge component with a free Sérsic index.
    disk : ExponentialComponent
        Exponential disc component, equivalent to a Sérsic profile with
        fixed index ``n = 1``.

    Notes
    -----
    This class assigns explicit physical interpretations to its components.
    It does not require shared centres, axis ratios, or position angles.
    Such parameter couplings belong to the fitting parameterization rather
    than to the intrinsic morphological definition.

    The model owns neither a PSF nor a background component.
    """

    bulge: SersicModel
    disk: ExponentialComponent

    def __post_init__(self) -> None:
        """Validate the physical component types."""
        if not isinstance(self.bulge, SersicModel):
            raise TypeError(
                "BulgeDiskModel requires a SersicModel bulge."
            )

        if isinstance(self.bulge, ExponentialComponent):
            raise TypeError(
                "BulgeDiskModel bulge must be a general SersicModel, "
                "not an ExponentialComponent."
            )

        if not isinstance(
            self.disk,
            ExponentialComponent,
        ):
            raise TypeError(
                "BulgeDiskModel requires an ExponentialComponent disk."
            )

    def _composite(self) -> CompositeModel:
        """Return the delegated additive intrinsic composition."""
        return CompositeModel(
            {
                "bulge": self.bulge,
                "disk": self.disk,
            }
        )

    def render_components(
        self,
        shape: tuple[int, int],
        *,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> dict[str, FloatArray]:
        """Render the named bulge and disc contributions."""
        return self._composite().render_components(
            shape=shape,
            origin=origin,
        )

    def render(
        self,
        shape: tuple[int, int],
        *,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> FloatArray:
        """Render the total intrinsic bulge-disc image."""
        return self._composite().render(
            shape=shape,
            origin=origin,
        )

    @property
    def total_flux(self) -> float:
        """Return the total intrinsic bulge-plus-disc flux."""
        return float(
            self.bulge.flux
            + self.disk.flux
        )

    @property
    def bulge_to_total(self) -> float:
        """Return the intrinsic bulge-to-total flux ratio."""
        return float(
            self.bulge.flux
            / self.total_flux
        )
