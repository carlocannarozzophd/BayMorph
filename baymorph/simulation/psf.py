"""PSF models and convolution utilities."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import irfftn, next_fast_len, rfftn


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


_CONVOLUTION_CACHE_ATTRIBUTE = "_baymorph_convolution_cache"

_CONVOLUTION_CACHE_ENTRIES = 4


class _PaddedKernelTransform:
    """Linear FFT convolution with the kernel transform computed once.

    The padded transform shape, the transform of the zero-padded kernel and
    the crop indices depend only on the kernel values and the image shape.
    Both are fixed for the duration of a fit, so recomputing them for every
    likelihood evaluation transforms a constant array millions of times.

    This performs exactly the operations ``scipy.signal.fftconvolve`` performs
    for ``mode="same"``: the same ``next_fast_len`` shape, the same real
    transforms, the same centred crop, in the same order. The results agree
    bit for bit.
    """

    def __init__(
        self,
        kernel: np.ndarray,
        image_shape: tuple[int, int],
    ) -> None:
        self.image_shape = (
            int(image_shape[0]),
            int(image_shape[1]),
        )
        self.kernel_shape = (
            int(kernel.shape[0]),
            int(kernel.shape[1]),
        )

        self.fft_shape = tuple(
            next_fast_len(
                image + kernel_length - 1,
                real=True,
            )
            for image, kernel_length in zip(
                self.image_shape,
                self.kernel_shape,
            )
        )

        buffer = np.zeros(
            self.fft_shape,
            dtype=float,
        )
        buffer[
            : self.kernel_shape[0],
            : self.kernel_shape[1],
        ] = kernel

        self.kernel_transform = rfftn(
            buffer,
            self.fft_shape,
        )

        # The centred window of the linear convolution lies entirely inside
        # the valid region, so it can be taken from the padded inverse
        # transform without trimming first.
        self.crop = tuple(
            slice(
                (kernel_length - 1) // 2,
                (kernel_length - 1) // 2 + image,
            )
            for image, kernel_length in zip(
                self.image_shape,
                self.kernel_shape,
            )
        )

    def __call__(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Convolve one image with the cached kernel transform."""
        spectrum = rfftn(
            image,
            self.fft_shape,
        )
        spectrum *= self.kernel_transform

        return irfftn(
            spectrum,
            self.fft_shape,
        )[self.crop]


def _padded_kernel_transform(
    psf,
    kernel: np.ndarray,
    image_shape: tuple[int, int],
) -> _PaddedKernelTransform:
    """Return the cached padded transform for one PSF, kernel and shape.

    The cache lives on the PSF object because every other object in the
    render chain is rebuilt for each likelihood evaluation. A PSF object that
    does not accept attribute assignment receives an uncached transform; that
    cannot change any result, because the cached and uncached paths perform
    the same operations in the same order.
    """
    key = (
        tuple(image_shape),
        kernel.shape,
        kernel.tobytes(),
    )

    cache = getattr(
        psf,
        _CONVOLUTION_CACHE_ATTRIBUTE,
        None,
    )

    if cache is None:
        cache = {}

        try:
            setattr(
                psf,
                _CONVOLUTION_CACHE_ATTRIBUTE,
                cache,
            )
        except (AttributeError, TypeError):
            return _PaddedKernelTransform(
                kernel,
                image_shape,
            )

    transform = cache.get(key)

    if transform is None:
        if len(cache) >= _CONVOLUTION_CACHE_ENTRIES:
            cache.clear()

        transform = _PaddedKernelTransform(
            kernel,
            image_shape,
        )
        cache[key] = transform

    return transform


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

    transform = _padded_kernel_transform(
        psf,
        normalized_kernel,
        image_array.shape,
    )

    return transform(
        image_array
    )
