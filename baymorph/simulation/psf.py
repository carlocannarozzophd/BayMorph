"""PSF models and convolution utilities."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import irfftn, next_fast_len, rfftn
from scipy.ndimage import map_coordinates, spline_filter

from baymorph.rendering.effective_psf import (
    RENDERING_CONVENTION,
    SUBPIXEL_FACTOR,
    divide_cell_response,
    interleave,
)

SUBPIXEL_SUM_TOLERANCE = 1e-9
SUBPIXEL_ZERO_PHASE_TOLERANCE = 1e-12


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
        Empirical PSF kernel: the effective PSF at integer offsets.
    subpixel_kernels : ndarray, optional
        The effective PSF at the ``SUBPIXEL_FACTOR`` sub-pixel phases of
        each axis, shaped ``(f, f, size, size)`` with ``size`` the side of
        ``kernel_array``. Element ``[j, i]`` is the kernel at the phase
        ``((i - c) / f, (j - c) / f)``, ``c = (f - 1) / 2``: its pixel
        ``[b, a]`` is the effective PSF at offset ``(a - h, b - h)`` plus
        the phase, ``h = size // 2``. Every phase is non-negative with unit
        sum, and the zero phase equals the normalized ``kernel_array``.
        With them, convolved models follow the effective-PSF rendering of
        AD-046; without them, they sample the profile at pixel centres.

    Attributes
    ----------
    rendering_convention : str or None
        The rendering convention the sub-pixel kernels select, recorded in
        the checkpoint signature; ``None`` without sub-pixel kernels.
    """

    kernel_array: np.ndarray
    subpixel_kernels: np.ndarray | None = None
    rendering_convention: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.subpixel_kernels is None:
            return

        self.subpixel_kernels = validate_subpixel_kernels(
            self.subpixel_kernels,
            self.kernel(),
        )
        self.rendering_convention = RENDERING_CONVENTION

    @property
    def includes_pixel_response(self) -> bool:
        """Return whether this kernel already includes the pixel response.

        ``EmpiricalPSF`` kernels are built from real detector-pixel star
        cutouts with no deconvolution: by construction they already equal
        ``optical_PSF ⊛ pixel_response``, the effective PSF. Integrating the
        profile over pixels before convolving with it would apply the pixel
        response twice.
        """
        return True

    @property
    def has_subpixel_kernels(self) -> bool:
        """Return whether the effective PSF is known at sub-pixel offsets."""
        return self.subpixel_kernels is not None

    def fine_kernel(self) -> np.ndarray:
        """Return the sub-pixel kernels as one grid, corrected for the cell.

        The grid is finer by ``SUBPIXEL_FACTOR`` with the zero offset at its
        middle pixel, divided by the response of one fine cell. It is
        computed once and kept on the object.
        """
        if not self.has_subpixel_kernels:
            raise ValueError(
                "This empirical PSF has no sub-pixel kernels."
            )

        cached = self.__dict__.get("_fine_kernel")

        if cached is None:
            cached = divide_cell_response(
                interleave(self.subpixel_kernels),
                SUBPIXEL_FACTOR,
            )
            self.__dict__["_fine_kernel"] = cached

        return cached

    def kernel_at(
        self,
        phase_x: float,
        phase_y: float,
    ) -> np.ndarray:
        """Return the normalized effective PSF at a sub-pixel phase.

        Pixel ``[b, a]`` of the result is the effective PSF at offset
        ``(a - h + phase_x, b - h + phase_y)``, interpolated with cubic
        splines between the sub-pixel kernels, which it reproduces at their
        own phases. Negative interpolation ringing is set to zero before the
        normalization.
        """
        if not self.has_subpixel_kernels:
            raise ValueError(
                "This empirical PSF has no sub-pixel kernels."
            )

        if not (abs(phase_x) <= 0.5 and abs(phase_y) <= 0.5):
            raise ValueError(
                "Sub-pixel phases must lie within half a pixel."
            )

        coefficients = self.__dict__.get("_spline_coefficients")

        if coefficients is None:
            coefficients = spline_filter(
                interleave(self.subpixel_kernels),
                order=3,
                mode="constant",
            )
            self.__dict__["_spline_coefficients"] = coefficients

        factor = SUBPIXEL_FACTOR
        size = self.subpixel_kernels.shape[-1]
        offset = (factor - 1) / 2
        indices = factor * np.arange(size, dtype=float) + offset
        rows, columns = np.meshgrid(
            indices + factor * phase_y,
            indices + factor * phase_x,
            indexing="ij",
        )
        kernel = np.clip(
            map_coordinates(
                coefficients,
                [rows, columns],
                order=3,
                mode="constant",
                cval=0.0,
                prefilter=False,
            ),
            0.0,
            None,
        )

        return kernel / kernel.sum()

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


def validate_subpixel_kernels(
    subpixel_kernels: ArrayLike,
    native_kernel: np.ndarray,
) -> np.ndarray:
    """Validate the sub-pixel kernels of an empirical PSF.

    They must have ``SUBPIXEL_FACTOR`` phases per axis and the footprint of
    the native kernel, be finite and non-negative with unit sum at every
    phase, and equal the normalized native kernel at zero phase. Invalid
    kernels are rejected, never repaired.
    """
    kernels = np.asarray(
        subpixel_kernels,
        dtype=float,
    )
    factor = SUBPIXEL_FACTOR

    if kernels.shape != (factor, factor, *native_kernel.shape):
        raise ValueError(
            "Sub-pixel kernels must have shape "
            f"({factor}, {factor}, {native_kernel.shape[0]}, "
            f"{native_kernel.shape[1]}); got {kernels.shape}."
        )

    if not np.all(np.isfinite(kernels)):
        raise ValueError(
            "Sub-pixel kernels must contain only finite values."
        )

    if np.any(kernels < 0.0):
        raise ValueError(
            "Sub-pixel kernels must contain only non-negative values."
        )

    sums = kernels.sum(axis=(-2, -1))

    if np.any(np.abs(sums - 1.0) > SUBPIXEL_SUM_TOLERANCE):
        raise ValueError(
            "Every sub-pixel kernel must have unit sum."
        )

    centre = (factor - 1) // 2
    difference = np.abs(kernels[centre, centre] - native_kernel).max()

    if difference > SUBPIXEL_ZERO_PHASE_TOLERANCE * native_kernel.max():
        raise ValueError(
            "The zero-phase sub-pixel kernel differs from the native "
            f"kernel by {difference:.3e}."
        )

    return kernels


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
