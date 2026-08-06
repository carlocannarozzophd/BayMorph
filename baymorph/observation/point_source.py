"""Observation-level rendering of unresolved point sources."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from baymorph.models.point_source import PointSourceModel
from baymorph.observation.effects import render_shifted_psf

FloatArray = NDArray[np.float64]


class PSFPointSourceRenderer:
    """Render intrinsic point sources through an observation PSF.

    Parameters
    ----------
    psf : object
        PSF object exposing a ``kernel()`` method.
    kernel_size : int, default=25
        Kernel size used for analytic Gaussian PSFs.
    """

    def __init__(
        self,
        psf: Any,
        *,
        kernel_size: int = 25,
    ) -> None:
        if not hasattr(psf, "kernel"):
            raise TypeError(
                "PSFPointSourceRenderer requires a PSF object "
                "exposing kernel()."
            )

        if (
            not isinstance(kernel_size, (int, np.integer))
            or kernel_size <= 0
            or kernel_size % 2 == 0
        ):
            raise ValueError(
                "Point-source kernel_size must be a positive odd integer."
            )

        self.psf = psf
        self.kernel_size = int(kernel_size)

    def render(
        self,
        point_source: PointSourceModel,
        shape: tuple[int, int],
        *,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> FloatArray:
        """Render the expected detector image of one point source."""
        if not isinstance(point_source, PointSourceModel):
            raise TypeError(
                "PSFPointSourceRenderer requires a PointSourceModel."
            )

        return render_shifted_psf(
            psf=self.psf,
            flux=point_source.flux,
            x0=point_source.x0,
            y0=point_source.y0,
            shape=shape,
            origin=origin,
            kernel_size=self.kernel_size,
        )
