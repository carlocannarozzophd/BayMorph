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
value validated by the noise-model bench
(``docs/research/NOISE_MODEL_BENCH_2026-10-01.md``). A kernel whose spectrum
is not strictly positive is rejected, never repaired. The correlation at
the lags beyond the kernel, up to ``TRUNCATION_DIAGNOSTIC_MAX_LAG``, is
reported as a diagnostic of the truncation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.fft import next_fast_len

FloatArray = NDArray[np.float64]

CALIBRATION_SCHEMA_VERSION = 2
"""Schema of the calibration product; version 1 held the retired estimator."""

CORRELATION_ESTIMATOR = "masked_sky_autocovariance_v1"
"""Name of the estimator of AD-047, recorded in every product."""

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
) -> CorrelatedGaussianNoiseCalibration:
    """Estimate the stationary sky correlation of one observation (AD-047).

    Parameters
    ----------
    image : ndarray
        Two-dimensional cutout, before or after subtraction of a constant
        background; the plane fitted to the sky absorbs either.
    sky_mask : ndarray of bool
        ``True`` for the pixels of the sky region.

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
        ``MINIMUM_SKY_PAIRS`` sky pixel pairs, or if the spectrum of the
        kernel is not strictly positive.
    """
    data = np.asarray(image, dtype=float)
    mask = np.asarray(sky_mask, dtype=bool)

    if data.ndim != 2:
        raise ValueError("image must be two-dimensional.")

    if mask.shape != data.shape:
        raise ValueError("sky_mask must match image shape.")

    if not np.all(np.isfinite(data[mask])):
        raise ValueError("Sky pixels must contain only finite image values.")

    correlation, pairs = masked_sky_correlation(
        data,
        mask,
        max_lag=TRUNCATION_DIAGNOSTIC_MAX_LAG,
    )
    minimum_pairs = int(np.min(pairs))

    if minimum_pairs < MINIMUM_SKY_PAIRS:
        raise ValueError(
            f"The sky region holds {minimum_pairs} pixel pairs at some lag up "
            f"to {TRUNCATION_DIAGNOSTIC_MAX_LAG}, fewer than the "
            f"{MINIMUM_SKY_PAIRS} the sky-correlation estimator of AD-047 "
            "requires."
        )

    centre = TRUNCATION_DIAGNOSTIC_MAX_LAG
    inner = slice(centre - CORRELATION_MAX_LAG, centre + CORRELATION_MAX_LAG + 1)
    kernel = np.array(correlation[inner, inner], copy=True)
    minimum_spectrum = minimum_stationary_spectrum(
        kernel,
        grid_size=SPECTRUM_GRID_SIZE,
    )

    if minimum_spectrum <= 0.0:
        raise ValueError(
            "The sky correlation is not positive definite: the spectrum of "
            f"its kernel reaches {minimum_spectrum:.3e}. AD-047 rejects such a "
            "kernel instead of repairing it."
        )

    ring = np.ones(correlation.shape, dtype=bool)
    ring[inner, inner] = False
    metadata = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "decision": "AD-047",
        "model": "correlated_gaussian",
        "factorization": "C = S R S",
        "correlation_model": "stationary_sky_autocovariance",
        "estimator": CORRELATION_ESTIMATOR,
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
        "beyond_kernel_sum": float(np.sum(correlation[ring])),
        "beyond_kernel_max_abs_correlation": float(
            np.max(np.abs(correlation[ring]))
        ),
        "correlation_along_x": [float(value) for value in correlation[centre, centre:]],
        "correlation_along_y": [float(value) for value in correlation[centre:, centre]],
    }

    return CorrelatedGaussianNoiseCalibration(
        correlation_kernel=kernel,
        metadata=metadata,
    )


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

    if np.any(pairs < 1):
        raise ValueError(
            f"The sky region has no pixel pair at some lag up to {max_lag}."
        )

    covariance = power[rows, columns] / pairs
    covariance = 0.5 * (covariance + covariance[::-1, ::-1])

    if not covariance[max_lag, max_lag] > 0.0:
        raise ValueError("The sky residual has no variance.")

    return covariance / covariance[max_lag, max_lag], pairs


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
