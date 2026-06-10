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


def convolve_with_psf(
    image: np.ndarray,
    psf: GaussianPSF,
    kernel_size: int = 25,
) -> np.ndarray:
    """
    Convolve image with PSF.
    """

    kernel = psf.kernel(
        size=kernel_size
    )

    return fftconvolve(
        image,
        kernel,
        mode="same",
    )