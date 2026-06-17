"""PSF models and convolution utilities."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


@dataclass
class GaussianPSF:
    """
    Circular Gaussian PSF.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum in pixels.
    """

    fwhm: float

    @property
    def sigma(self) -> float:
        return self.fwhm / 2.354820045

    def kernel(
        self,
        size: int = 25,
    ) -> np.ndarray:
        """
        Generate a normalized PSF kernel.
        """

        if size % 2 == 0:
            raise ValueError(
                "Kernel size must be odd."
            )

        center = size // 2

        y, x = np.mgrid[0:size, 0:size]

        r2 = (
            (x - center) ** 2
            + (y - center) ** 2
        )

        kernel = np.exp(
            -0.5 * r2 / self.sigma**2
        )

        kernel /= kernel.sum()

        return kernel


@dataclass
class EmpiricalPSF:
    """
    Empirical PSF represented by a 2D kernel.

    Parameters
    ----------
    kernel_array : ndarray
        Empirical PSF kernel.
    """

    kernel_array: np.ndarray

    def kernel(
        self,
        size: int | None = None,
    ) -> np.ndarray:
        """
        Return a normalized empirical PSF kernel.
        """

        kernel = np.asarray(
            self.kernel_array,
            dtype=float,
        )

        if kernel.ndim != 2:
            raise ValueError(
                "Empirical PSF kernel must be 2D."
            )

        kernel = np.nan_to_num(
            kernel,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        kernel[kernel < 0] = 0.0

        total = kernel.sum()

        if total <= 0:
            raise ValueError(
                "Empirical PSF kernel must have positive flux."
            )

        return kernel / total


def convolve_with_psf(
    image: np.ndarray,
    psf,
    kernel_size: int = 25,
) -> np.ndarray:
    """
    Convolve image with PSF.
    """

    if isinstance(
        psf,
        GaussianPSF,
    ):
        kernel = psf.kernel(
            size=kernel_size,
        )

    else:
        kernel = psf.kernel()

    return fftconvolve(
        image,
        kernel,
        mode="same",
    )