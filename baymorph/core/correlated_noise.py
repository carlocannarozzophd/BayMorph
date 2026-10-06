"""Stationary sky correlation of the correlated Gaussian noise model (AD-047).

AD-047 models the noise of an observed image as a Gaussian with covariance
``C = S R S``: ``S`` is diagonal, from the prepared sky scale, and ``R`` is
the stationary correlation of the sky, with unit diagonal. This module
estimates ``R`` once per observation on the sky region of the preparation:

1. one plane is fitted to the sky pixels and removed;
2. the autocovariance of the residual is computed with the sky as a mask and
   normalized at every lag by the number of sky pixel pairs, without a taper;
3. it is symmetrized and divided by its value at lag zero.

The kernel keeps the lags up to ``CORRELATION_MAX_LAG`` along each axis, the
value validated by the simulation bench of noise models. A kernel whose
spectrum is not strictly positive is rejected, never repaired. The correlation at
the lags beyond the kernel, up to ``TRUNCATION_DIAGNOSTIC_MAX_LAG``, is
reported as a diagnostic of the truncation.

That estimator, ``autocovariance``, assumes that the correlation of the sky
ends within the kernel. A sky that also holds structure on scales of tens to
hundreds of pixels, a large-scale field, breaks that assumption, and its
kernel is rejected. The estimator ``field_separated`` of AD-051 is declared
for such a sky, never chosen automatically: it fits a smooth isotropic
correlation, ``v exp(-r / l) + c``, to the correlation at the lags beyond the
kernel, subtracts it within the kernel and renormalizes the remainder at lag
zero. It was validated on sky regions whose contaminant masks were grown by
``FIELD_SEPARATED_GROWTH_PIXELS``. Every calibration it makes also records
the kernel of a high-pass estimate, which counts as noise the structure
below about 30 px that the field-separated estimate counts as field; the
ratio of their kernel sums measures how much that attribution matters on
the sky.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.fft import next_fast_len
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

FloatArray = NDArray[np.float64]

CALIBRATION_SCHEMA_VERSION = 3
"""Schema of the calibration product: version 1 held the retired estimator,
version 2 the estimator of AD-047 alone, and version 3 records which of the
declared estimators made the kernel."""

CORRELATION_ESTIMATOR = "masked_sky_autocovariance_v1"
"""Name of the estimator of AD-047, recorded in every product."""

FIELD_SEPARATED_ESTIMATOR = "field_separated_sky_autocovariance_v1"
"""Name of the field-separated estimator of AD-051, recorded in its products."""

SKY_CORRELATION_ESTIMATORS = {
    "autocovariance": CORRELATION_ESTIMATOR,
    "field_separated": FIELD_SEPARATED_ESTIMATOR,
}
"""Name of each sky-correlation estimator in the configuration, and the name
its products record."""

DEFAULT_SKY_CORRELATION = "autocovariance"
"""The estimator of a configuration that declares none, that of AD-047."""

FIELD_SEPARATED_GROWTH_PIXELS = 20
"""Growth of the contaminant masks, in pixels, of the sky regions on which
the field-separated estimator was validated (AD-051)."""

FIELD_FIT_MAX_LAG = 48
"""Largest length of the lags at which the field is fitted; the fit uses the
lags whose larger component exceeds ``CORRELATION_MAX_LAG``."""

FIELD_LENGTH_BOUNDS = (16.0, 400.0)
"""Bounds of the length of the field, in pixels. The lower bound is twice the
reach of the kernel: a shorter component cannot be told from the noise at
the lags beyond the kernel."""

FIELD_START_LENGTH = 30.0
"""Length, in pixels, from which the fit of the field starts."""

HIGH_PASS_SIGMA = 16.0
"""Width, in pixels, of the Gaussian smoothing of the high-pass diagnostic."""

HIGH_PASS_TRUNCATION = 3.0
"""The smoothing of the high-pass diagnostic is cut at this many widths and
renormalized."""

HIGH_PASS_FIT_MAX_LAG = 24
"""Largest lag of the filtered correlation fitted by the high-pass
diagnostic."""

CORRELATION_MAX_LAG = 8
"""Largest lag of the correlation kernel along each axis (AD-047)."""

TRUNCATION_DIAGNOSTIC_MAX_LAG = 16
"""Largest lag at which the correlation is measured for the diagnostic."""

SPECTRUM_GRID_SIZE = 512
"""Fourier grid on which the spectrum of the kernel must be positive."""

MINIMUM_SKY_PAIRS = 1000
"""Fewest sky pixel pairs accepted at any lag up to the diagnostic lag."""


@dataclass(frozen=True)
class CorrelatedGaussianNoiseCalibration:
    """Fixed sky correlation of one observation (AD-047).

    Parameters
    ----------
    correlation_kernel : ndarray
        Odd-sized stationary two-dimensional correlation kernel, indexed
        ``[dy, dx]`` from ``-max_lag`` to ``max_lag``, with unit zero-lag
        correlation.
    metadata : dict
        Estimator constants and diagnostics required for reproducibility.
    """

    correlation_kernel: FloatArray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate and freeze the correlation kernel."""
        kernel = np.array(self.correlation_kernel, dtype=float, copy=True)

        if kernel.ndim != 2:
            raise ValueError("correlation_kernel must be two-dimensional.")

        if any(size < 1 or size % 2 == 0 for size in kernel.shape):
            raise ValueError(
                "correlation_kernel dimensions must be positive and odd."
            )

        if not np.all(np.isfinite(kernel)):
            raise ValueError("correlation_kernel must contain only finite values.")

        centre = (kernel.shape[0] // 2, kernel.shape[1] // 2)

        if not np.isclose(kernel[centre], 1.0, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError(
                "correlation_kernel must have unit zero-lag correlation."
            )

        if not np.allclose(kernel, kernel[::-1, ::-1], rtol=1.0e-10, atol=1.0e-12):
            raise ValueError(
                "correlation_kernel must satisfy covariance symmetry."
            )

        kernel.setflags(write=False)
        object.__setattr__(self, "correlation_kernel", kernel)
        object.__setattr__(self, "metadata", dict(self.metadata))


def calibrate_correlated_gaussian_noise(
    image: NDArray[np.floating],
    sky_mask: NDArray[np.bool_],
    *,
    estimator: str = DEFAULT_SKY_CORRELATION,
) -> CorrelatedGaussianNoiseCalibration:
    """Estimate the stationary sky correlation of one observation (AD-047).

    Parameters
    ----------
    image : ndarray
        Two-dimensional cutout, before or after subtraction of a constant
        background; the plane fitted to the sky absorbs either.
    sky_mask : ndarray of bool
        ``True`` for the pixels of the sky region.
    estimator : {"autocovariance", "field_separated"}, default="autocovariance"
        ``autocovariance``, the estimator of AD-047, keeps the correlation of
        the sky up to ``CORRELATION_MAX_LAG``; ``field_separated``, that of
        AD-051, first subtracts a large-scale field fitted beyond it.

    Returns
    -------
    CorrelatedGaussianNoiseCalibration
        The correlation kernel up to ``CORRELATION_MAX_LAG`` and the
        diagnostics of the estimate.

    Raises
    ------
    ValueError
        If the inputs are invalid, if some lag up to
        ``TRUNCATION_DIAGNOSTIC_MAX_LAG`` has fewer than
        ``MINIMUM_SKY_PAIRS`` sky pixel pairs, if some lag that the
        estimator reads has no sky pixel pair, or if the spectrum of the
        kernel is not strictly positive.
    """
    if estimator not in SKY_CORRELATION_ESTIMATORS:
        raise ValueError(
            f"Unknown sky-correlation estimator {estimator!r}; expected one "
            f"of {list(SKY_CORRELATION_ESTIMATORS)}."
        )

    data = np.asarray(image, dtype=float)
    mask = np.asarray(sky_mask, dtype=bool)

    if data.ndim != 2:
        raise ValueError("image must be two-dimensional.")

    if mask.shape != data.shape:
        raise ValueError("sky_mask must match image shape.")

    if not np.all(np.isfinite(data[mask])):
        raise ValueError("Sky pixels must contain only finite image values.")

    field_separated = estimator == "field_separated"
    reach = FIELD_FIT_MAX_LAG if field_separated else TRUNCATION_DIAGNOSTIC_MAX_LAG
    correlation, pairs = masked_sky_correlation(
        data,
        mask,
        max_lag=reach,
    )
    near = slice(
        reach - TRUNCATION_DIAGNOSTIC_MAX_LAG,
        reach + TRUNCATION_DIAGNOSTIC_MAX_LAG + 1,
    )
    minimum_pairs = int(np.min(pairs[near, near]))

    if minimum_pairs < MINIMUM_SKY_PAIRS:
        raise ValueError(
            f"The sky region holds {minimum_pairs} pixel pairs at some lag up "
            f"to {TRUNCATION_DIAGNOSTIC_MAX_LAG}, fewer than the "
            f"{MINIMUM_SKY_PAIRS} the sky-correlation estimator of AD-047 "
            "requires."
        )

    inner = slice(reach - CORRELATION_MAX_LAG, reach + CORRELATION_MAX_LAG + 1)

    if field_separated:
        kernel, estimator_metadata = _field_separated_kernel(correlation)
    else:
        kernel = np.array(correlation[inner, inner], copy=True)
        estimator_metadata = {"decision": "AD-047"}

    minimum_spectrum = minimum_stationary_spectrum(
        kernel,
        grid_size=SPECTRUM_GRID_SIZE,
    )

    if minimum_spectrum <= 0.0 and field_separated:
        raise ValueError(
            "The field-separated sky correlation is not positive definite: "
            f"the spectrum of its kernel reaches {minimum_spectrum:.3e}. "
            "AD-051 rejects such a kernel instead of repairing it."
        )

    if minimum_spectrum <= 0.0:
        raise ValueError(
            "The sky correlation is not positive definite: the spectrum of "
            f"its kernel reaches {minimum_spectrum:.3e}. AD-047 rejects such a "
            "kernel instead of repairing it. A sky that holds correlated "
            "structure on scales larger than the kernel, such as a "
            "large-scale background field, needs the field-separated "
            "estimator: declare preparation.noise.sky_correlation: "
            "field_separated, which requires preparation.masking."
            f"contaminant_growth_pixels: {FIELD_SEPARATED_GROWTH_PIXELS}, the "
            "default with data.mask_mode: union, and prepare the image in a "
            "new output namespace."
        )

    if field_separated:
        estimator_metadata.update(_high_pass_attribution(data, mask, kernel))

    diagnostic = correlation[near, near]
    centre = TRUNCATION_DIAGNOSTIC_MAX_LAG
    ring = np.ones(diagnostic.shape, dtype=bool)
    ring[
        centre - CORRELATION_MAX_LAG:centre + CORRELATION_MAX_LAG + 1,
        centre - CORRELATION_MAX_LAG:centre + CORRELATION_MAX_LAG + 1,
    ] = False
    metadata = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "decision": estimator_metadata.pop("decision"),
        "model": "correlated_gaussian",
        "factorization": "C = S R S",
        "correlation_model": "stationary_sky_autocovariance",
        "estimator": SKY_CORRELATION_ESTIMATORS[estimator],
        "plane_removed": True,
        "taper": "none",
        "max_lag": CORRELATION_MAX_LAG,
        "diagnostic_max_lag": TRUNCATION_DIAGNOSTIC_MAX_LAG,
        "spectrum_grid_size": SPECTRUM_GRID_SIZE,
        "minimum_sky_pairs": MINIMUM_SKY_PAIRS,
        "sky_pixels": int(np.count_nonzero(mask)),
        "minimum_pair_count": minimum_pairs,
        "minimum_spectrum": float(minimum_spectrum),
        "kernel_sum": float(np.sum(kernel)),
        "beyond_kernel_sum": float(np.sum(diagnostic[ring])),
        "beyond_kernel_max_abs_correlation": float(
            np.max(np.abs(diagnostic[ring]))
        ),
        "correlation_along_x": [float(value) for value in diagnostic[centre, centre:]],
        "correlation_along_y": [float(value) for value in diagnostic[centre:, centre]],
        **estimator_metadata,
    }

    return CorrelatedGaussianNoiseCalibration(
        correlation_kernel=kernel,
        metadata=metadata,
    )


def _field_separated_kernel(
    correlation: FloatArray,
) -> tuple[FloatArray, dict[str, Any]]:
    """Return the kernel of the field-separated estimator of AD-051.

    ``correlation`` reaches ``FIELD_FIT_MAX_LAG``. The model of the field,
    ``v exp(-r / l) + c``, is fitted by bounded least squares to the means of
    the correlation in bins of 1 px of the lag length ``r``, over the lags
    whose larger component exceeds ``CORRELATION_MAX_LAG`` and whose length
    is at most ``FIELD_FIT_MAX_LAG``. The constant ``c`` stands for the
    deficit that the removal of the plane leaves at short lags. The fitted
    surface is subtracted within the kernel, which is divided by the part of
    the variance left to the noise, ``1 - v - c``.
    """
    reach = FIELD_FIT_MAX_LAG
    lags = np.arange(-reach, reach + 1)
    larger = np.maximum(np.abs(lags[:, None]), np.abs(lags[None, :]))
    length = np.hypot(lags[:, None], lags[None, :])
    selected = (larger > CORRELATION_MAX_LAG) & (length <= reach)
    bins = np.rint(length[selected]).astype(int)
    radii = np.unique(bins)
    values = correlation[selected]
    means = np.array([np.mean(values[bins == radius]) for radius in radii])

    def residual(parameters: FloatArray) -> FloatArray:
        amplitude, scale, offset = parameters

        return amplitude * np.exp(-radii / scale) + offset - means

    start = (
        max(float(means[0] - means[-1]), 1.0e-3),
        FIELD_START_LENGTH,
        float(means[-1]),
    )
    result = least_squares(
        residual,
        start,
        bounds=(
            [0.0, FIELD_LENGTH_BOUNDS[0], -1.0],
            [1.0, FIELD_LENGTH_BOUNDS[1], 1.0],
        ),
    )
    amplitude, scale, offset = (float(value) for value in result.x)
    noise_fraction = 1.0 - amplitude - offset

    if not noise_fraction > 0.0:
        raise ValueError(
            "The large-scale field fitted to the sky correlation leaves no "
            f"variance to the noise: 1 - v - c = {noise_fraction:.3e}."
        )

    inner = slice(reach - CORRELATION_MAX_LAG, reach + CORRELATION_MAX_LAG + 1)
    field = amplitude * np.exp(-length[inner, inner] / scale) + offset
    kernel = (correlation[inner, inner] - field) / noise_fraction
    # One by construction; set exactly, since the nearest-neighbour
    # factorization requires it and the last bit of the division can differ.
    kernel[CORRELATION_MAX_LAG, CORRELATION_MAX_LAG] = 1.0

    return kernel, {
        "decision": "AD-051",
        "field_model": "v exp(-r / l) + c",
        "field_fit_max_lag": FIELD_FIT_MAX_LAG,
        "field_length_bounds": list(FIELD_LENGTH_BOUNDS),
        "field_amplitude": amplitude,
        "field_length": scale,
        "field_offset": offset,
        "field_length_at_lower_bound": bool(
            scale <= FIELD_LENGTH_BOUNDS[0] * (1.0 + 1.0e-6)
        ),
        "field_rms_misfit": float(np.sqrt(np.mean(result.fun**2))),
        "field_fit_radii": [int(radius) for radius in radii],
        "field_fit_correlation": [float(value) for value in means],
        "noise_fraction": noise_fraction,
    }


def _high_pass_attribution(
    image: FloatArray,
    mask: NDArray[np.bool_],
    kernel: FloatArray,
) -> dict[str, Any]:
    """Return the high-pass estimate of the kernel sum and its ratio to ``kernel``.

    The high-pass filter ``delta - G``, with ``G`` a Gaussian of
    ``HIGH_PASS_SIGMA`` cut at ``HIGH_PASS_TRUNCATION`` widths and
    renormalized, acts on the autocovariance as ``H = delta - 2 G + G * G``,
    exactly for a stationary field and without the edges of the mask. The
    autocovariance, with zero at the lags without pairs, is filtered by
    ``H``, and the kernel of AD-047, zero beyond ``CORRELATION_MAX_LAG``, is
    recovered by linear least squares from the filtered correlation up to
    ``HIGH_PASS_FIT_MAX_LAG``. That kernel counts as noise the structure
    below about 30 px that the field-separated kernel counts as field. It is
    a diagnostic and never enters the likelihood.
    """
    response = _high_pass_lag_response()
    reach = response.shape[0] // 2
    covariance, pairs = _masked_sky_autocovariance(
        image,
        mask,
        max_lag=HIGH_PASS_FIT_MAX_LAG + reach,
    )
    centre = covariance.shape[0] // 2
    values = covariance / covariance[centre, centre]
    filtered = fftconvolve(values, response, mode="valid")
    sparse = fftconvolve(
        (pairs < MINIMUM_SKY_PAIRS).astype(float),
        np.abs(response),
        mode="valid",
    )
    half_plane = [
        (dy, dx)
        for dy in range(0, CORRELATION_MAX_LAG + 1)
        for dx in range(-CORRELATION_MAX_LAG, CORRELATION_MAX_LAG + 1)
        if dy > 0 or dx >= 0
    ]
    middle = response.shape[0] // 2
    fitted = np.arange(-HIGH_PASS_FIT_MAX_LAG, HIGH_PASS_FIT_MAX_LAG + 1)
    columns = []

    for dy, dx in half_plane:
        column = response[
            middle + fitted[:, None] - dy,
            middle + fitted[None, :] - dx,
        ].copy()

        if (dy, dx) != (0, 0):
            column += response[
                middle + fitted[:, None] + dy,
                middle + fitted[None, :] + dx,
            ]

        columns.append(column.ravel())

    design = np.column_stack(columns)
    solution, *_ = np.linalg.lstsq(design, filtered.ravel(), rcond=None)
    high_pass = np.zeros(kernel.shape)

    for value, (dy, dx) in zip(solution, half_plane):
        high_pass[CORRELATION_MAX_LAG + dy, CORRELATION_MAX_LAG + dx] = value
        high_pass[CORRELATION_MAX_LAG - dy, CORRELATION_MAX_LAG - dx] = value

    high_pass = high_pass / high_pass[CORRELATION_MAX_LAG, CORRELATION_MAX_LAG]
    misfit = filtered.ravel() - design @ solution
    high_pass_sum = float(np.sum(high_pass))

    return {
        "high_pass_sigma": HIGH_PASS_SIGMA,
        "high_pass_truncation": HIGH_PASS_TRUNCATION,
        "high_pass_fit_max_lag": HIGH_PASS_FIT_MAX_LAG,
        "high_pass_kernel_sum": high_pass_sum,
        "high_pass_rms_misfit": float(np.sqrt(np.mean(misfit**2))),
        "high_pass_weight_on_sparse_lags": float(np.max(sparse)),
        "attribution_ratio": high_pass_sum / float(np.sum(kernel)),
    }


def _high_pass_lag_response() -> FloatArray:
    """Return ``H = delta - 2 G + G * G`` on its own grid, centred."""
    half = int(np.ceil(HIGH_PASS_TRUNCATION * HIGH_PASS_SIGMA))
    profile = np.exp(-0.5 * (np.arange(-half, half + 1) / HIGH_PASS_SIGMA) ** 2)
    single = np.outer(profile, profile)
    single = single / single.sum()
    double = fftconvolve(single, single)
    response = double.copy()
    offset = (double.shape[0] - single.shape[0]) // 2
    response[
        offset:offset + single.shape[0],
        offset:offset + single.shape[1],
    ] -= 2.0 * single
    response[double.shape[0] // 2, double.shape[1] // 2] += 1.0

    return response


def masked_sky_correlation(
    image: NDArray[np.floating],
    sky_mask: NDArray[np.bool_],
    *,
    max_lag: int,
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Return the correlation of the sky residual and its pair counts.

    One plane is fitted to the sky pixels by least squares and removed. The
    autocovariance of the residual at each lag is the sum of the products of
    the sky pixel pairs at that lag divided by their number, computed by
    zero-padded Fourier transforms; it is symmetrized and divided by its value
    at lag zero.

    Parameters
    ----------
    image : ndarray
        Two-dimensional image.
    sky_mask : ndarray of bool
        ``True`` for the sky pixels.
    max_lag : int
        Largest lag along each axis.

    Returns
    -------
    correlation : ndarray
        Correlation indexed ``[dy, dx]`` from ``-max_lag`` to ``max_lag``.
    pairs : ndarray of int
        Number of sky pixel pairs at each lag, on the same grid.
    """
    covariance, pairs = _masked_sky_autocovariance(
        image,
        sky_mask,
        max_lag=max_lag,
    )

    if np.any(pairs < 1):
        raise ValueError(
            f"The sky region has no pixel pair at some lag up to {max_lag}."
        )

    if not covariance[max_lag, max_lag] > 0.0:
        raise ValueError("The sky residual has no variance.")

    return covariance / covariance[max_lag, max_lag], pairs


def _masked_sky_autocovariance(
    image: NDArray[np.floating],
    sky_mask: NDArray[np.bool_],
    *,
    max_lag: int,
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Return the symmetrized autocovariance of the sky residual and its pairs.

    The computation of :func:`masked_sky_correlation` before its checks and
    normalization; a lag without any sky pixel pair has zero autocovariance.
    """
    data = np.asarray(image, dtype=float)
    mask = np.asarray(sky_mask, dtype=bool)

    if max_lag < 1:
        raise ValueError("max_lag must be at least one.")

    if np.count_nonzero(mask) < 3:
        raise ValueError("At least three sky pixels are required to fit a plane.")

    yy, xx = np.indices(data.shape, dtype=float)
    design = np.column_stack(
        [np.ones(int(np.count_nonzero(mask))), xx[mask], yy[mask]]
    )

    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("The sky pixels do not determine a plane.")

    coefficients = np.linalg.lstsq(design, data[mask], rcond=None)[0]
    plane = coefficients[0] + coefficients[1] * xx + coefficients[2] * yy
    residual = np.where(mask, data - plane, 0.0)
    shape = tuple(next_fast_len(size + max_lag + 1) for size in data.shape)
    power = np.fft.irfft2(np.abs(np.fft.rfft2(residual, s=shape)) ** 2, s=shape)
    counts = np.fft.irfft2(
        np.abs(np.fft.rfft2(mask.astype(float), s=shape)) ** 2,
        s=shape,
    )
    offsets = np.arange(-max_lag, max_lag + 1)
    rows = offsets[:, None] % shape[0]
    columns = offsets[None, :] % shape[1]
    pairs = np.rint(counts[rows, columns]).astype(np.int64)
    covariance = np.divide(
        power[rows, columns],
        pairs,
        out=np.zeros(pairs.shape),
        where=pairs > 0,
    )

    return 0.5 * (covariance + covariance[::-1, ::-1]), pairs


def minimum_stationary_spectrum(
    kernel: NDArray[np.floating],
    *,
    grid_size: int,
) -> float:
    """Return the minimum of the spectrum of a finite stationary kernel."""
    values = np.asarray(kernel, dtype=float)
    lag_y = values.shape[0] // 2
    lag_x = values.shape[1] // 2

    if grid_size < max(values.shape):
        raise ValueError("grid_size must not be smaller than the kernel.")

    grid = np.zeros((grid_size, grid_size), dtype=float)
    rows = np.arange(-lag_y, lag_y + 1)[:, None] % grid_size
    columns = np.arange(-lag_x, lag_x + 1)[None, :] % grid_size
    grid[rows, columns] = values

    return float(np.min(np.fft.fft2(grid).real))
