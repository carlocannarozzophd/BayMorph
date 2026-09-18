"""PSF models and convolution utilities."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import fftconvolve


def validate_finite_nonnegative_kernel(
    kernel: ArrayLike,
    *,
    label: str,
) -> np.ndarray:
    """Validate PSF-kernel structural hygiene: 2D, finite, non-negative.

    Does not check normalization or centering; callers apply those with
    their own tolerance conventions. Rejects invalid kernels; never repairs
    them.

    Parameters
    ----------
    kernel : array_like
        Candidate PSF kernel.
    label : str
        Human-readable kernel identity used in raised error messages.

    Returns
    -------
    ndarray
        The kernel as a float array, unchanged.

    Raises
    ------
    ValueError
        If the kernel is not 2D, contains non-finite values, or contains
        negative values.
    """
    array = np.asarray(
        kernel,
        dtype=float,
    )

    if array.ndim != 2:
        raise ValueError(
            f"{label} kernel must be a 2D array."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{label} kernel must contain only finite values."
        )

    if np.any(array < 0.0):
        raise ValueError(
            f"{label} kernel must contain only non-negative values."
        )

    return array


@dataclass
class GaussianPSF:
    """Circular Gaussian point-spread function.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum, in pixels.
    """

    fwhm: float

    def __post_init__(self) -> None:
        """Validate the Gaussian PSF parameters."""
        if not np.isfinite(self.fwhm) or self.fwhm <= 0.0:
            raise ValueError(
                "Gaussian PSF FWHM must be finite and positive."
            )

    @property
    def includes_pixel_response(self) -> bool:
        """Return whether this kernel already includes the pixel response.

        ``GaussianPSF`` is a point-sampled analytic profile: the detector
        pixel response is not baked into the kernel, so the intrinsic model
        must still be integrated over each pixel footprint before
        convolution.
        """
        return False

    @property
    def sigma(self) -> float:
        """Return the Gaussian standard deviation in pixels."""
        return self.fwhm / 2.354820045

    def kernel(
        self,
        size: int = 25,
    ) -> np.ndarray:
        """Generate a normalized Gaussian PSF kernel.

        Parameters
        ----------
        size : int, default=25
            Width and height of the square kernel. Must be a positive odd
            integer.

        Returns
        -------
        ndarray
            Unit-normalized PSF kernel.

        Raises
        ------
        ValueError
            If ``size`` is not a positive odd integer.
        """
        if (
            not isinstance(size, (int, np.integer))
            or size <= 0
            or size % 2 == 0
        ):
            raise ValueError(
                "Gaussian PSF kernel size must be a positive odd integer."
            )

        center = size // 2
        y, x = np.mgrid[0:size, 0:size]

        radius_squared = (
            (x - center) ** 2
            + (y - center) ** 2
        )

        kernel = np.exp(
            -0.5 * radius_squared / self.sigma**2
        )

        return kernel / kernel.sum()


@dataclass
class EmpiricalPSF:
    """Empirical point-spread function represented by a 2D kernel.

    Parameters
    ----------
    kernel_array : ndarray
        Empirical PSF kernel.
    """

    kernel_array: np.ndarray

    @property
    def includes_pixel_response(self) -> bool:
        """Return whether this kernel already includes the pixel response.

        ``EmpiricalPSF`` kernels are built from real detector-pixel star
        cutouts with no deconvolution: by construction they already equal
        ``optical_PSF ⊛ pixel_response``. The intrinsic model must therefore
        be point-sampled at pixel centers rather than pixel-footprint
        integrated before convolution, or the pixel response would be
        applied twice.
        """
        return True

    def kernel(
        self,
        size: int | None = None,
    ) -> np.ndarray:
        """Return a normalized empirical PSF kernel.

        Parameters
        ----------
        size : int, optional
            Reserved for API compatibility with analytic PSF models. The value
            is ignored.

        Returns
        -------
        ndarray
            Unit-normalized empirical PSF kernel.

        Raises
        ------
        ValueError
            If the kernel is not two-dimensional, contains non-finite or
            negative values, or has non-positive total flux.
        """
        del size

        kernel = validate_finite_nonnegative_kernel(
            self.kernel_array,
            label="Empirical PSF",
        )

        total = kernel.sum()

        if total <= 0.0:
            raise ValueError(
                "Empirical PSF kernel must have positive flux."
            )

        return kernel / total


def convolve_with_psf(
    image: np.ndarray,
    psf,
    kernel_size: int = 25,
) -> np.ndarray:
    """Convolve a finite image with a normalized PSF kernel.

    The convolution assumes zero surface brightness outside the supplied image.
    Flux scattered beyond the finite image footprint is therefore not retained.
    This is preferable to edge replication, which creates artificial emission
    outside the rendered domain.

    Parameters
    ----------
    image : ndarray
        Two-dimensional finite input image.
    psf : object
        PSF object exposing a ``kernel()`` method. For ``GaussianPSF``,
        ``kernel_size`` is passed as ``kernel(size=...)``.
    kernel_size : int, default=25
        Kernel size used for Gaussian PSF models.

    Returns
    -------
    ndarray
        PSF-convolved image with the same shape as ``image``.

    Raises
    ------
    ValueError
        If the image or PSF kernel is not two-dimensional, contains
        non-finite or negative values, or if the kernel has non-positive
        total flux.
    """
    image_array = np.asarray(
        image,
        dtype=float,
    )

    if image_array.ndim != 2:
        raise ValueError(
            "Input image must be 2D."
        )

    if not np.all(np.isfinite(image_array)):
        raise ValueError(
            "Input image must contain only finite values."
        )

    if isinstance(psf, GaussianPSF):
        kernel = psf.kernel(size=kernel_size)
    else:
        kernel = psf.kernel()

    kernel_array = validate_finite_nonnegative_kernel(
        kernel,
        label="PSF",
    )

    kernel_sum = kernel_array.sum()

    if kernel_sum <= 0.0:
        raise ValueError(
            "PSF kernel must have positive sum."
        )

    normalized_kernel = kernel_array / kernel_sum

    return np.asarray(
        fftconvolve(
            image_array,
            normalized_kernel,
            mode="same",
        ),
        dtype=float,
    )
