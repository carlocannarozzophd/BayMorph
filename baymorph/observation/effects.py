"""Deterministic observation effects."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from baymorph.simulation.psf import convolve_with_psf

FloatArray = NDArray[np.float64]


class PSFEffect:
    """Convolve a finite expected image with a normalized PSF kernel.

    Parameters
    ----------
    psf : object
        PSF object exposing a ``kernel()`` method.
    kernel_size : int, default=25
        Kernel size used by analytic Gaussian PSF models.

    Notes
    -----
    The input image is treated as having zero surface brightness outside its
    finite footprint. Flux scattered beyond that footprint is therefore not
    retained.

    Extended-domain rendering must be handled before applying this effect when
    light entering the requested footprint from outside must be preserved.
    """

    def __init__(
        self,
        psf: Any,
        *,
        kernel_size: int = 25,
    ) -> None:
        if not hasattr(psf, "kernel"):
            raise TypeError(
                "PSFEffect requires a PSF object exposing kernel()."
            )

        if (
            not isinstance(kernel_size, (int, np.integer))
            or kernel_size <= 0
            or kernel_size % 2 == 0
        ):
            raise ValueError(
                "PSFEffect kernel_size must be a positive odd integer."
            )

        self.psf = psf
        self.kernel_size = int(kernel_size)

    def apply(
        self,
        image: ArrayLike,
    ) -> FloatArray:
        """Convolve the finite input image with the configured PSF."""
        return np.asarray(
            convolve_with_psf(
                np.asarray(image, dtype=float),
                self.psf,
                kernel_size=self.kernel_size,
            ),
            dtype=float,
        )
