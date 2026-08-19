"""Calibration utilities for within-observation correlated Gaussian noise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CorrelatedGaussianNoiseCalibration:
    """Fixed calibration for a correlated Gaussian image likelihood.

    Parameters
    ----------
    marginal_sigma : ndarray
        Positive two-dimensional field of marginal standard deviations.
    correlation_kernel : ndarray
        Odd-sized two-dimensional stationary correlation kernel centred on
        unit zero-lag correlation.
    metadata : dict
        Calibration parameters and diagnostics required for reproducibility.
    """

    marginal_sigma: np.ndarray
    correlation_kernel: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate and freeze the calibration arrays."""
        sigma = np.array(
            self.marginal_sigma,
            dtype=float,
            copy=True,
        )

        kernel = np.array(
            self.correlation_kernel,
            dtype=float,
            copy=True,
        )

        if sigma.ndim != 2:
            raise ValueError(
                "marginal_sigma must be two-dimensional."
            )

        if not np.all(np.isfinite(sigma)):
            raise ValueError(
                "marginal_sigma must contain only finite values."
            )

        if np.any(sigma <= 0.0):
            raise ValueError(
                "marginal_sigma must be strictly positive."
            )

        if kernel.ndim != 2:
            raise ValueError(
                "correlation_kernel must be two-dimensional."
            )

        if any(
            size < 1 or size % 2 == 0
            for size in kernel.shape
        ):
            raise ValueError(
                "correlation_kernel dimensions must be positive and odd."
            )

        if not np.all(np.isfinite(kernel)):
            raise ValueError(
                "correlation_kernel must contain only finite values."
            )

        center = (
            kernel.shape[0] // 2,
            kernel.shape[1] // 2,
        )

        if not np.isclose(
            kernel[center],
            1.0,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError(
                "correlation_kernel must have unit zero-lag correlation."
            )

        if not np.allclose(
            kernel,
            kernel[::-1, ::-1],
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError(
                "correlation_kernel must satisfy covariance symmetry."
            )

        sigma.setflags(write=False)
        kernel.setflags(write=False)

        object.__setattr__(
            self,
            "marginal_sigma",
            sigma,
        )
        object.__setattr__(
            self,
            "correlation_kernel",
            kernel,
        )
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )


def calibrate_correlated_gaussian_noise(
    image: np.ndarray,
    sky_mask: np.ndarray,
    *,
    marginal_tile_size: int,
    marginal_min_coverage: float,
    correlation_tile_size: int,
    correlation_stride: int,
    max_lag: int,
    spectrum_grid_size: int = 512,
) -> CorrelatedGaussianNoiseCalibration:
    """Calibrate one fixed spatial Gaussian noise model from blank sky.

    The covariance factorization is

    .. math::

        C = S R S,

    where ``S`` is a smooth spatial marginal-standard-deviation field and
    ``R`` is an empirical stationary two-dimensional correlation structure.

    Parameters
    ----------
    image : ndarray
        Two-dimensional observed cutout before or after subtraction of a
        constant background.
    sky_mask : ndarray of bool
        Pixels belonging to the blank-sky calibration region.
    marginal_tile_size : int
        Non-overlapping tile size used to calibrate the marginal scale field.
    marginal_min_coverage : float
        Minimum fraction of valid sky pixels required for a marginal-scale
        tile.
    correlation_tile_size : int
        Size of fully valid sky tiles used to estimate the correlation.
    correlation_stride : int
        Stride between correlation-calibration tiles.
    max_lag : int
        Maximum retained absolute correlation lag in either detector axis.
    spectrum_grid_size : int, default=512
        Fourier grid used to verify positivity of the finite stationary
        correlation kernel.

    Returns
    -------
    CorrelatedGaussianNoiseCalibration
        Fixed marginal scale field, correlation kernel, and calibration
        diagnostics.
    """
    data = np.asarray(
        image,
        dtype=float,
    )

    mask = np.asarray(
        sky_mask,
        dtype=bool,
    )

    if data.ndim != 2:
        raise ValueError(
            "image must be two-dimensional."
        )

    if mask.shape != data.shape:
        raise ValueError(
            "sky_mask must match image shape."
        )

    if not np.all(
        np.isfinite(
            data[mask]
        )
    ):
        raise ValueError(
            "Sky pixels must contain only finite image values."
        )

    if marginal_tile_size < 2:
        raise ValueError(
            "marginal_tile_size must be at least two."
        )

    if not 0.0 < marginal_min_coverage <= 1.0:
        raise ValueError(
            "marginal_min_coverage must lie in (0, 1]."
        )

    if correlation_tile_size < 2:
        raise ValueError(
            "correlation_tile_size must be at least two."
        )

    if correlation_stride < 1:
        raise ValueError(
            "correlation_stride must be at least one."
        )

    if max_lag < 1:
        raise ValueError(
            "max_lag must be at least one."
        )

    if correlation_tile_size <= 2 * max_lag:
        raise ValueError(
            "correlation_tile_size must exceed 2 * max_lag."
        )

    marginal_sigma, marginal_metadata = (
        _fit_marginal_sigma_field(
            data,
            mask,
            tile_size=marginal_tile_size,
            min_coverage=marginal_min_coverage,
        )
    )

    correlation_kernel, correlation_metadata = (
        _estimate_correlation_kernel(
            data,
            mask,
            tile_size=correlation_tile_size,
            stride=correlation_stride,
            max_lag=max_lag,
            spectrum_grid_size=spectrum_grid_size,
        )
    )

    metadata = {
        "schema_version": 1,
        "model": "correlated_gaussian",
        "factorization": "C = S R S",
        "marginal_model": "quadratic_log_sigma",
        "marginal_tile_size": int(
            marginal_tile_size
        ),
        "marginal_min_coverage": float(
            marginal_min_coverage
        ),
        "correlation_model": "empirical_stationary_2d",
        "correlation_tile_size": int(
            correlation_tile_size
        ),
        "correlation_stride": int(
            correlation_stride
        ),
        "max_lag": int(max_lag),
        "spectrum_grid_size": int(
            spectrum_grid_size
        ),
        **marginal_metadata,
        **correlation_metadata,
    }

    return CorrelatedGaussianNoiseCalibration(
        marginal_sigma=marginal_sigma,
        correlation_kernel=correlation_kernel,
        metadata=metadata,
    )


def _mad_sigma(
    values: np.ndarray,
) -> float:
    """Return a Gaussian-equivalent MAD scale."""
    finite = np.asarray(
        values,
        dtype=float,
    )

    finite = finite[
        np.isfinite(finite)
    ]

    if finite.size == 0:
        return np.nan

    median = np.median(finite)

    return float(
        1.4826
        * np.median(
            np.abs(
                finite - median
            )
        )
    )


def _fit_planar_residual(
    data: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Remove a local planar mean from selected pixels."""
    valid = np.asarray(
        mask,
        dtype=bool,
    )

    if np.count_nonzero(valid) < 3:
        raise ValueError(
            "At least three valid pixels are required for planar detrending."
        )

    yy, xx = np.indices(
        data.shape,
        dtype=float,
    )

    design = np.column_stack(
        [
            np.ones(
                np.count_nonzero(valid)
            ),
            xx[valid],
            yy[valid],
        ]
    )

    coefficients = np.linalg.lstsq(
        design,
        data[valid],
        rcond=None,
    )[0]

    plane = (
        coefficients[0]
        + coefficients[1] * xx
        + coefficients[2] * yy
    )

    residual = np.full(
        data.shape,
        np.nan,
        dtype=float,
    )

    residual[valid] = (
        data[valid]
        - plane[valid]
    )

    return residual


def _fit_marginal_sigma_field(
    data: np.ndarray,
    sky_mask: np.ndarray,
    *,
    tile_size: int,
    min_coverage: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a quadratic spatial model for log marginal noise scale."""
    x_values: list[float] = []
    y_values: list[float] = []
    sigma_values: list[float] = []

    height, width = data.shape

    for y0 in range(
        0,
        height,
        tile_size,
    ):
        y1 = min(
            y0 + tile_size,
            height,
        )

        for x0 in range(
            0,
            width,
            tile_size,
        ):
            x1 = min(
                x0 + tile_size,
                width,
            )

            local_mask = sky_mask[
                y0:y1,
                x0:x1,
            ]

            coverage = float(
                np.mean(local_mask)
            )

            if coverage < min_coverage:
                continue

            if np.count_nonzero(
                local_mask
            ) < 20:
                continue

            local_data = data[
                y0:y1,
                x0:x1,
            ]

            residual = _fit_planar_residual(
                local_data,
                local_mask,
            )

            sigma = _mad_sigma(
                residual[local_mask]
            )

            if (
                not np.isfinite(sigma)
                or sigma <= 0.0
            ):
                continue

            x_values.append(
                x0
                + 0.5
                * (x1 - x0 - 1)
            )

            y_values.append(
                y0
                + 0.5
                * (y1 - y0 - 1)
            )

            sigma_values.append(
                sigma
            )

    if len(sigma_values) < 7:
        raise RuntimeError(
            "Too few sky tiles to fit the marginal noise-scale field."
        )

    x = np.asarray(
        x_values,
        dtype=float,
    )
    y = np.asarray(
        y_values,
        dtype=float,
    )
    sigma = np.asarray(
        sigma_values,
        dtype=float,
    )

    x_center = float(
        np.mean(x)
    )
    y_center = float(
        np.mean(y)
    )

    coordinate_scale = float(
        np.sqrt(
            np.var(x)
            + np.var(y)
        )
    )

    if (
        not np.isfinite(
            coordinate_scale
        )
        or coordinate_scale <= 0.0
    ):
        raise RuntimeError(
            "Unable to define a finite spatial scale for noise calibration."
        )

    x_normalized = (
        x - x_center
    ) / coordinate_scale

    y_normalized = (
        y - y_center
    ) / coordinate_scale

    design = np.column_stack(
        [
            np.ones(x.size),
            x_normalized,
            y_normalized,
            x_normalized**2,
            x_normalized
            * y_normalized,
            y_normalized**2,
        ]
    )

    if np.linalg.matrix_rank(
        design
    ) < design.shape[1]:
        raise RuntimeError(
            "Marginal noise-scale calibration design matrix is rank deficient."
        )

    coefficients = np.linalg.lstsq(
        design,
        np.log(sigma),
        rcond=None,
    )[0]

    yy, xx = np.indices(
        data.shape,
        dtype=float,
    )

    xn = (
        xx.ravel()
        - x_center
    ) / coordinate_scale

    yn = (
        yy.ravel()
        - y_center
    ) / coordinate_scale

    full_design = np.column_stack(
        [
            np.ones(xx.size),
            xn,
            yn,
            xn**2,
            xn * yn,
            yn**2,
        ]
    )

    marginal_sigma = np.exp(
        full_design @ coefficients
    ).reshape(
        data.shape
    )

    if not np.all(
        np.isfinite(
            marginal_sigma
        )
    ):
        raise RuntimeError(
            "Marginal noise-scale field contains non-finite values."
        )

    metadata = {
        "marginal_tile_count": int(
            sigma.size
        ),
        "marginal_sigma_median": float(
            np.median(sigma)
        ),
        "marginal_sigma_p16": float(
            np.percentile(
                sigma,
                16.0,
            )
        ),
        "marginal_sigma_p84": float(
            np.percentile(
                sigma,
                84.0,
            )
        ),
        "marginal_coordinate_center_x": x_center,
        "marginal_coordinate_center_y": y_center,
        "marginal_coordinate_scale": coordinate_scale,
        "marginal_log_sigma_coefficients": (
            coefficients.tolist()
        ),
    }

    return (
        marginal_sigma,
        metadata,
    )


def _estimate_correlation_kernel(
    data: np.ndarray,
    sky_mask: np.ndarray,
    *,
    tile_size: int,
    stride: int,
    max_lag: int,
    spectrum_grid_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate a stationary empirical correlation kernel."""
    tiles: list[np.ndarray] = []

    height, width = data.shape

    for y0 in range(
        0,
        height - tile_size + 1,
        stride,
    ):
        y1 = y0 + tile_size

        for x0 in range(
            0,
            width - tile_size + 1,
            stride,
        ):
            x1 = x0 + tile_size

            local_mask = sky_mask[
                y0:y1,
                x0:x1,
            ]

            if not np.all(
                local_mask
            ):
                continue

            tile = data[
                y0:y1,
                x0:x1,
            ]

            residual = _fit_planar_residual(
                tile,
                np.ones(
                    tile.shape,
                    dtype=bool,
                ),
            )

            sigma = _mad_sigma(
                residual
            )

            if (
                not np.isfinite(sigma)
                or sigma <= 0.0
            ):
                continue

            tiles.append(
                residual / sigma
            )

    if len(tiles) < 4:
        raise RuntimeError(
            "Too few fully valid sky tiles to estimate correlation."
        )

    average_power = np.zeros(
        (tile_size, tile_size),
        dtype=float,
    )

    for tile in tiles:
        transformed = np.fft.fft2(
            tile
        )

        average_power += (
            np.abs(
                transformed
            )
            ** 2
        )

    average_power /= len(tiles)

    covariance = np.fft.ifft2(
        average_power
    ).real

    covariance /= covariance[
        0,
        0,
    ]

    coordinates = np.arange(
        tile_size,
        dtype=int,
    )

    wrapped_lag = np.minimum(
        coordinates,
        tile_size - coordinates,
    ).astype(float)

    taper_1d = np.clip(
        1.0
        - wrapped_lag
        / (max_lag + 1.0),
        0.0,
        None,
    )

    tapered = covariance * np.outer(
        taper_1d,
        taper_1d,
    )

    spectrum = np.fft.fft2(
        tapered
    ).real

    spectrum_scale = float(
        np.max(
            np.abs(spectrum)
        )
    )

    floor = max(
        1.0e-12,
        1.0e-12
        * spectrum_scale,
    )

    clipped = np.maximum(
        spectrum,
        floor,
    )

    regularized = np.fft.ifft2(
        clipped
    ).real

    regularized /= regularized[
        0,
        0,
    ]

    support = (
        2 * max_lag + 1
    )

    kernel = np.empty(
        (support, support),
        dtype=float,
    )

    for iy, dy in enumerate(
        range(
            -max_lag,
            max_lag + 1,
        )
    ):
        for ix, dx in enumerate(
            range(
                -max_lag,
                max_lag + 1,
            )
        ):
            kernel[
                iy,
                ix,
            ] = regularized[
                dy % tile_size,
                dx % tile_size,
            ]

    kernel = 0.5 * (
        kernel
        + kernel[::-1, ::-1]
    )

    kernel /= kernel[
        max_lag,
        max_lag,
    ]

    minimum_fine_spectrum = (
        _minimum_stationary_spectrum(
            kernel,
            grid_size=spectrum_grid_size,
        )
    )

    if minimum_fine_spectrum <= 0.0:
        raise RuntimeError(
            "The calibrated finite correlation kernel is not "
            "positive definite on the verification spectrum grid."
        )

    metadata = {
        "correlation_tile_count": int(
            len(tiles)
        ),
        "correlation_minimum_raw_spectrum": float(
            np.min(spectrum)
        ),
        "correlation_clipped_spectrum_fraction": float(
            np.mean(
                spectrum < floor
            )
        ),
        "correlation_minimum_fine_spectrum": float(
            minimum_fine_spectrum
        ),
    }

    return (
        kernel,
        metadata,
    )


def _minimum_stationary_spectrum(
    kernel: np.ndarray,
    *,
    grid_size: int,
) -> float:
    """Return the minimum spectrum of a finite stationary kernel."""
    max_lag_y = (
        kernel.shape[0] // 2
    )
    max_lag_x = (
        kernel.shape[1] // 2
    )

    minimum_grid = max(
        kernel.shape
    )

    if grid_size < minimum_grid:
        raise ValueError(
            "spectrum_grid_size must not be smaller than the kernel."
        )

    covariance_grid = np.zeros(
        (grid_size, grid_size),
        dtype=float,
    )

    for iy, dy in enumerate(
        range(
            -max_lag_y,
            max_lag_y + 1,
        )
    ):
        for ix, dx in enumerate(
            range(
                -max_lag_x,
                max_lag_x + 1,
            )
        ):
            covariance_grid[
                dy % grid_size,
                dx % grid_size,
            ] = kernel[
                iy,
                ix,
            ]

    spectrum = np.fft.fft2(
        covariance_grid
    ).real

    return float(
        np.min(spectrum)
    )
