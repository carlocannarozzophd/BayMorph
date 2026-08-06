"""Intrinsic unresolved point-source morphology."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PointSourceModel:
    """Intrinsic unresolved point source.

    Parameters
    ----------
    flux : float
        Total intrinsic source flux.
    x0, y0 : float
        Source centre in global pixel coordinates.

    Notes
    -----
    A point source is intrinsically represented by a Dirac delta distribution.
    It does not define a regular pixelized intrinsic image.

    Rendering its expected detector image therefore requires an observation
    model containing a PSF. The point source deliberately owns no PSF and
    exposes no ``render()`` method.
    """

    flux: float
    x0: float
    y0: float

    def __post_init__(self) -> None:
        """Validate the intrinsic point-source parameters."""
        if not np.isfinite(self.flux) or self.flux <= 0.0:
            raise ValueError(
                "Point-source flux must be finite and positive."
            )

        if not np.isfinite(self.x0) or not np.isfinite(self.y0):
            raise ValueError(
                "Point-source coordinates must be finite."
            )
