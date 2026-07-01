"""PSF models and convolution utilities."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


@dataclass
class GaussianPSF:
    """Circular Gaussian point-spread function.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum, in pixels.
    """

    fwhm: float

    @property
    def sigma(self) -> float:
        """float : Gaussian standard deviation, in pixels."""
        return self.fwhm / 2.354820045

    def kernel(
        self,
        size: int = 25,
    ) -> np.ndarray:
        """Generate a normalized Gaussian PSF kernel.

        Parameters
        ----------
        size : int, default=25
            Width and height of the square kernel. Must be odd.

        Returns
        -------
        ndarray
            Unit-normalized PSF kernel.

        Raises
        ------
        ValueError
            If ``size`` is even.
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
    """Empirical point-spread function represented by a 2D kernel.

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
        """Return a normalized empirical PSF kernel.

        Parameters
        ----------
        size : int, optional
            Reserved for API compatibility with analytic PSF models. The value
            is currently ignored.

        Returns
        -------
        ndarray
            Unit-normalized empirical PSF kernel.

        Raises
        ------
        ValueError
            If the kernel is not two-dimensional or has non-positive total
            flux.
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
    """Convolve an image with a PSF using edge padding.

    Parameters
    ----------
    image : ndarray
        Input image.
    psf : object
        PSF object exposing a ``kernel()`` method. For ``GaussianPSF``,
        ``kernel_size`` is passed to ``kernel(size=...)``.
    kernel_size : int, default=25
        Kernel size used for Gaussian PSF models.

    Returns
    -------
    ndarray
        PSF-convolved image with the same shape as ``image``.

    Raises
    ------
    ValueError
        If the PSF kernel is not two-dimensional, contains non-finite values,
        or has non-positive total flux.
    """
    image = np.asarray(
        image,
        dtype=float,
    )

    if isinstance(
        psf,
        GaussianPSF,
    ):
        kernel = psf.kernel(
            size=kernel_size,
        )
    else:
        kernel = psf.kernel()

    kernel = np.asarray(
        kernel,
        dtype=float,
    )

    if kernel.ndim != 2:
        raise ValueError(
            "PSF kernel must be 2D."
        )

    if not np.all(
        np.isfinite(kernel)
    ):
        raise ValueError(
            "PSF kernel must contain only finite values."
        )

    kernel_sum = np.sum(
        kernel,
    )

    if kernel_sum <= 0:
        raise ValueError(
            "PSF kernel must have positive sum."
        )

    kernel = kernel / kernel_sum

    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2

    padded = np.pad(
        image,
        pad_width=(
            (pad_y, pad_y),
            (pad_x, pad_x),
        ),
        mode="edge",
    )

    convolved = fftconvolve(
        padded,
        kernel,
        mode="same",
    )

    return convolved[
        pad_y : pad_y + image.shape[0],
        pad_x : pad_x + image.shape[1],
    ]