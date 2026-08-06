"""PSF-rendered unresolved point-source model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from baymorph.observation import render_shifted_psf

FloatArray = NDArray[np.float64]


@dataclass
class PointSourceModel:
    """Unresolved source represented by a shifted observation PSF.

    Parameters
    ----------
    flux : float
        Total point-source flux before footprint clipping.
    x0, y0 : float
        Source centre in global pixel coordinates.
    psf : object
        PSF object exposing a ``kernel()`` method.
    kernel_size : int, default=25
        Kernel size used for analytic Gaussian PSFs.

    Notes
    -----
    A point source is not approximated by a narrow extended profile. Its
    expected detector image is the configured PSF shifted to the source
    position and scaled by the source flux.
    """

    flux: float
    x0: float
    y0: float
    psf: Any
    kernel_size: int = 25

    def __post_init__(self) -> None:
        """Validate point-source parameters."""
        if not np.isfinite(self.flux) or self.flux <= 0.0:
            raise ValueError(
                "Point-source flux must be finite and positive."
            )

        if not np.isfinite(self.x0) or not np.isfinite(self.y0):
            raise ValueError(
                "Point-source coordinates must be finite."
            )

        if not hasattr(self.psf, "kernel"):
            raise TypeError(
                "PointSourceModel requires a PSF object exposing kernel()."
            )

        if (
            not isinstance(self.kernel_size, (int, np.integer))
            or self.kernel_size <= 0
            or self.kernel_size % 2 == 0
        ):
            raise ValueError(
                "Point-source kernel_size must be a positive odd integer."
            )

        self.kernel_size = int(self.kernel_size)

    def render(
        self,
        shape: tuple[int, int],
        *,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> FloatArray:
        """Render the observed point-source image."""
        return render_shifted_psf(
            psf=self.psf,
            flux=self.flux,
            x0=self.x0,
            y0=self.y0,
            shape=shape,
            origin=origin,
            kernel_size=self.kernel_size,
        )

    def render_components(
        self,
        shape: tuple[int, int],
        *,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> dict[str, FloatArray]:
        """Render the named point-source contribution."""
        return {
            "point_source": self.render(
                shape=shape,
                origin=origin,
            )
        }
