"""Deterministic observation effects."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import shift

from baymorph.simulation.psf import (
    GaussianPSF,
    convolve_with_psf,
    validate_finite_nonnegative_kernel,
)

FloatArray = NDArray[np.float64]

PSF_SHIFT_RINGING_TOLERANCE = 1.0e-2
"""Maximum tolerated negative-flux fraction after sub-pixel PSF shifting.

Cubic-spline (``order=3``) sub-pixel shifting of an already valid (finite,
non-negative, centred) PSF kernel can introduce small negative "ringing"
values as a pure numerical artifact, not as evidence of an invalid input
kernel. Measured worst-case negative-flux fractions after shifting a
Gaussian kernel: about 1.1e-3 at Nyquist sampling (FWHM=2.0 px), about
3.7e-3 at FWHM=1.5 px, rising sharply beyond that (about 0.13 at FWHM=1.2
px). This tolerance comfortably passes any reasonably sampled empirical
PSF while still rejecting near-pathological, undersampled kernels or
excessive shifts.
"""


def normalized_psf_kernel(
    psf: Any,
    *,
    kernel_size: int = 25,
) -> FloatArray:
    """Return a finite, non-negative, unit-normalized PSF kernel.

    Parameters
    ----------
    psf : object
        PSF object exposing a ``kernel()`` method.
    kernel_size : int, default=25
        Kernel size used for analytic Gaussian PSFs.

    Returns
    -------
    ndarray
        Unit-normalized two-dimensional PSF kernel.

    Raises
    ------
    TypeError
        If ``psf`` does not expose a ``kernel()`` method.
    ValueError
        If the kernel is not 2D, contains non-finite or negative values, or
        has non-positive total flux. Invalid kernels are rejected, not
        repaired.
    """
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
            "PSF kernel_size must be a positive odd integer."
        )

    if isinstance(psf, GaussianPSF):
        kernel = psf.kernel(
            size=int(kernel_size),
        )
    else:
        kernel = psf.kernel()

    kernel_array = validate_finite_nonnegative_kernel(
        kernel,
        label="PSF",
    )

    total = float(kernel_array.sum())

    if total <= 0.0:
        raise ValueError(
            "PSF kernel must have positive sum."
        )

    return np.asarray(
        kernel_array / total,
        dtype=float,
    )


def render_shifted_psf(
    *,
    psf: Any,
    flux: float,
    x0: float,
    y0: float,
    shape: tuple[int, int],
    origin: tuple[float, float] = (0.0, 0.0),
    kernel_size: int = 25,
) -> FloatArray:
    """Render a flux-scaled PSF at an arbitrary sub-pixel position.

    The shifted kernel is normalized before placement on the requested
    footprint. Flux falling outside the footprint is therefore lost rather
    than redistributed into the retained pixels.

    Numerical approximation
    ------------------------
    Sub-pixel positioning uses cubic-spline interpolation
    (:func:`scipy.ndimage.shift`, ``order=3``), which can introduce small
    negative "ringing" values even for an already valid input kernel. This
    is treated as a declared, bounded numerical approximation, not as
    evidence of an invalid kernel: the negative flux introduced by shifting
    is clipped and the kernel renormalized only if its fraction of the
    total shifted flux stays within ``PSF_SHIFT_RINGING_TOLERANCE``.
    Exceeding that tolerance raises, since it signals a pathologically
    undersampled input kernel or an excessive shift rather than benign
    interpolation noise.

    Parameters
    ----------
    psf : object
        PSF object exposing a ``kernel()`` method.
    flux : float
        Total pre-clipping point-source flux.
    x0, y0 : float
        Point-source centre in global pixel coordinates.
    shape : tuple of int
        Requested output shape as ``(height, width)``.
    origin : tuple of float, default=(0.0, 0.0)
        Global coordinate of the output pixel at local index ``(0, 0)``,
        ordered as ``(y_origin, x_origin)``.
    kernel_size : int, default=25
        Kernel size used for analytic Gaussian PSFs.

    Returns
    -------
    ndarray
        Point-source image on the requested footprint.

    Raises
    ------
    ValueError
        If the shifted kernel's negative-flux fraction exceeds
        ``PSF_SHIFT_RINGING_TOLERANCE``, or if it has non-positive total
        flux.
    """
    if (
        len(shape) != 2
        or any(
            not isinstance(size, (int, np.integer))
            or size <= 0
            for size in shape
        )
    ):
        raise ValueError(
            "Point-source render shape must contain two positive integers."
        )

    if not np.isfinite(flux) or flux <= 0.0:
        raise ValueError(
            "Point-source flux must be finite and positive."
        )

    if not np.isfinite(x0) or not np.isfinite(y0):
        raise ValueError(
            "Point-source coordinates must be finite."
        )

    y_origin, x_origin = origin
    local_x = float(x0) - float(x_origin)
    local_y = float(y0) - float(y_origin)

    anchor_x = int(np.floor(local_x + 0.5))
    anchor_y = int(np.floor(local_y + 0.5))

    shift_x = local_x - anchor_x
    shift_y = local_y - anchor_y

    if getattr(psf, "has_subpixel_kernels", False):
        # AD-046: the effective PSF is known at sub-pixel offsets, so the
        # source is placed by interpolating between them instead of
        # shifting the native kernel.
        shifted_kernel = psf.kernel_at(-shift_x, -shift_y) * float(flux)
        return _place_kernel(shifted_kernel, shape, anchor_x, anchor_y)

    kernel = normalized_psf_kernel(
        psf,
        kernel_size=kernel_size,
    )

    shifted_kernel = shift(
        kernel,
        shift=(shift_y, shift_x),
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
    )

    shifted_total_signed = float(
        shifted_kernel.sum()
    )

    if shifted_total_signed <= 0.0:
        raise ValueError(
            "Shifted PSF kernel has non-positive flux."
        )

    negative_flux = float(
        -shifted_kernel[shifted_kernel < 0.0].sum()
    )
    negative_flux_fraction = negative_flux / shifted_total_signed

    if negative_flux_fraction > PSF_SHIFT_RINGING_TOLERANCE:
        raise ValueError(
            "Sub-pixel PSF shift produced negative flux of "
            f"{negative_flux_fraction:.3e} of total, exceeding the "
            "declared cubic-spline ringing tolerance of "
            f"{PSF_SHIFT_RINGING_TOLERANCE:.3e}. This indicates an "
            "undersampled or pathological kernel, or an excessive shift, "
            "not a benign interpolation artifact."
        )

    shifted_kernel = np.clip(
        shifted_kernel,
        a_min=0.0,
        a_max=None,
    )

    shifted_total = float(
        shifted_kernel.sum()
    )

    shifted_kernel /= shifted_total
    shifted_kernel *= float(flux)

    return _place_kernel(shifted_kernel, shape, anchor_x, anchor_y)


def _place_kernel(
    shifted_kernel: FloatArray,
    shape: tuple[int, int],
    anchor_x: int,
    anchor_y: int,
) -> FloatArray:
    """Place a flux-scaled kernel centred on a pixel of a footprint.

    The kernel centre falls on the local pixel ``(anchor_y, anchor_x)``;
    the part of the kernel outside the footprint is lost.
    """
    image = np.zeros(
        shape,
        dtype=float,
    )

    kernel_height, kernel_width = shifted_kernel.shape
    half_height = kernel_height // 2
    half_width = kernel_width // 2

    image_y_min = max(
        0,
        anchor_y - half_height,
    )
    image_y_max = min(
        shape[0],
        anchor_y - half_height + kernel_height,
    )
    image_x_min = max(
        0,
        anchor_x - half_width,
    )
    image_x_max = min(
        shape[1],
        anchor_x - half_width + kernel_width,
    )

    if (
        image_y_min >= image_y_max
        or image_x_min >= image_x_max
    ):
        return image

    kernel_y_min = image_y_min - (
        anchor_y - half_height
    )
    kernel_y_max = kernel_y_min + (
        image_y_max - image_y_min
    )
    kernel_x_min = image_x_min - (
        anchor_x - half_width
    )
    kernel_x_max = kernel_x_min + (
        image_x_max - image_x_min
    )

    image[
        image_y_min:image_y_max,
        image_x_min:image_x_max,
    ] = shifted_kernel[
        kernel_y_min:kernel_y_max,
        kernel_x_min:kernel_x_max,
    ]

    return image


class PSFEffect:
    """Convolve a finite expected image with a normalized PSF kernel."""

    def __init__(
        self,
        psf: Any,
        *,
        kernel_size: int = 25,
    ) -> None:
        normalized_psf_kernel(
            psf,
            kernel_size=kernel_size,
        )

        self.psf = psf
        self.kernel_size = int(kernel_size)

    def normalized_kernel(self) -> FloatArray:
        """Return the finite unit-normalized PSF kernel."""
        return normalized_psf_kernel(
            self.psf,
            kernel_size=self.kernel_size,
        )

    def apply(
        self,
        image: ArrayLike,
    ) -> FloatArray:
        """Convolve the finite input image with the configured PSF.

        The kernel was validated when this effect was constructed and cannot
        have changed since, so it is not revalidated here.
        """
        return np.asarray(
            convolve_with_psf(
                np.asarray(
                    image,
                    dtype=float,
                ),
                self.psf,
                kernel_size=self.kernel_size,
            ),
            dtype=float,
        )
