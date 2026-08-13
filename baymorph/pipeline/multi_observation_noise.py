"""Pre-fit diagnostics for cross-observation noise coupling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales

from baymorph.core.observation import Observation
from baymorph.observation.geometry import slice_wcs_to_cutout


@dataclass(frozen=True)
class CrossObservationNoisePairDiagnostic:
    """Blank-sky noise-correlation diagnostic for one observation pair."""

    observation_a: str
    observation_b: str
    status: str
    alignment: str

    n_overlap_zero_lag: int = 0

    zero_lag_covariance: float | None = None
    zero_lag_correlation: float | None = None

    peak_abs_correlation: float | None = None
    peak_lag_y: int | None = None
    peak_lag_x: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def estimate_cross_observation_noise_diagnostics(
    *,
    image_products: Mapping[str, Any],
    observation_configs: Mapping[str, Mapping[str, Any]],
    max_lag: int = 2,
    registration_tolerance_pixels: float = 0.05,
    minimum_overlap: int = 50,
) -> dict[str, Any]:
    """Estimate pre-fit blank-sky cross-observation correlations.

    The diagnostic is attempted for every observation pair.

    Correlations are evaluated only when the current reduced products can
    establish a common pixel grid. BayMorph deliberately does not resample
    images for this diagnostic because resampling can itself alter the noise
    covariance.

    The diagnostic is descriptive. It does not automatically modify the
    likelihood or declare observations statistically dependent.
    """
    names = tuple(
        image_products
    )

    if len(names) < 2:
        return {
            "status": "single_observation",
            "max_lag": int(max_lag),
            "interpretation": "diagnostic_only",
            "pairs": [],
        }

    if max_lag < 0:
        raise ValueError(
            "max_lag must be non-negative."
        )

    if registration_tolerance_pixels <= 0.0:
        raise ValueError(
            "registration_tolerance_pixels must be positive."
        )

    if minimum_overlap < 2:
        raise ValueError(
            "minimum_overlap must be at least two."
        )

    pairs = []

    for name_a, name_b in combinations(
        names,
        2,
    ):
        pairs.append(
            _estimate_pair(
                name_a=name_a,
                name_b=name_b,
                products_a=image_products[
                    name_a
                ],
                products_b=image_products[
                    name_b
                ],
                config_a=observation_configs[
                    name_a
                ],
                config_b=observation_configs[
                    name_b
                ],
                max_lag=max_lag,
                registration_tolerance_pixels=(
                    registration_tolerance_pixels
                ),
                minimum_overlap=minimum_overlap,
            )
        )

    return {
        "status": "completed",
        "max_lag": int(
            max_lag
        ),
        "interpretation": "diagnostic_only",
        "pairs": [
            pair.as_dict()
            for pair in pairs
        ],
    }


def print_cross_observation_noise_diagnostics(
    diagnostics: Mapping[str, Any],
) -> None:
    """Print a concise pre-fit cross-observation noise report."""
    pairs = diagnostics.get(
        "pairs",
        [],
    )

    if not pairs:
        return

    print()
    print(
        "Pre-fit cross-observation noise diagnostics"
    )

    for pair in pairs:
        name_a = pair[
            "observation_a"
        ]
        name_b = pair[
            "observation_b"
        ]
        status = pair[
            "status"
        ]

        if status != "estimated":
            print(
                f"  {name_a} x {name_b}: "
                f"{status} "
                f"({pair['alignment']})"
            )
            continue

        print(
            f"  {name_a} x {name_b}: "
            f"rho(0)="
            f"{pair['zero_lag_correlation']:+.4f}, "
            f"peak|rho|="
            f"{pair['peak_abs_correlation']:.4f} "
            f"at lag=("
            f"{pair['peak_lag_y']}, "
            f"{pair['peak_lag_x']}), "
            f"n={pair['n_overlap_zero_lag']}"
        )


def _estimate_pair(
    *,
    name_a: str,
    name_b: str,
    products_a: Any,
    products_b: Any,
    config_a: Mapping[str, Any],
    config_b: Mapping[str, Any],
    max_lag: int,
    registration_tolerance_pixels: float,
    minimum_overlap: int,
) -> CrossObservationNoisePairDiagnostic:
    data_a = np.asarray(
        products_a.prepared_data,
        dtype=float,
    )
    data_b = np.asarray(
        products_b.prepared_data,
        dtype=float,
    )

    if data_a.shape != data_b.shape:
        return CrossObservationNoisePairDiagnostic(
            observation_a=name_a,
            observation_b=name_b,
            status="not_estimable_shape_mismatch",
            alignment="unregistered",
        )

    mask_a = _sky_mask(
        products_a,
        data_a.shape,
    )
    mask_b = _sky_mask(
        products_b,
        data_b.shape,
    )

    if (
        mask_a is None
        or mask_b is None
    ):
        return CrossObservationNoisePairDiagnostic(
            observation_a=name_a,
            observation_b=name_b,
            status="not_estimable_missing_sky_region",
            alignment="unknown",
        )

    alignment = _grid_alignment_status(
        products_a=products_a,
        products_b=products_b,
        config_a=config_a,
        config_b=config_b,
        registration_tolerance_pixels=(
            registration_tolerance_pixels
        ),
    )

    if alignment != "registered":
        return CrossObservationNoisePairDiagnostic(
            observation_a=name_a,
            observation_b=name_b,
            status="not_estimable_unregistered_grid",
            alignment=alignment,
        )

    standardized_a = _standardized_image(
        products_a,
        data_a,
    )
    standardized_b = _standardized_image(
        products_b,
        data_b,
    )

    estimates = {}

    for lag_y in range(
        -max_lag,
        max_lag + 1,
    ):
        for lag_x in range(
            -max_lag,
            max_lag + 1,
        ):
            estimates[
                (
                    lag_y,
                    lag_x,
                )
            ] = _lagged_statistics(
                raw_a=data_a,
                raw_b=data_b,
                standardized_a=standardized_a,
                standardized_b=standardized_b,
                mask_a=mask_a,
                mask_b=mask_b,
                lag_y=lag_y,
                lag_x=lag_x,
                minimum_overlap=minimum_overlap,
            )

    zero = estimates[
        (
            0,
            0,
        )
    ]

    if zero is None:
        return CrossObservationNoisePairDiagnostic(
            observation_a=name_a,
            observation_b=name_b,
            status=(
                "not_estimable_insufficient_sky_overlap"
            ),
            alignment="registered",
        )

    valid = [
        (
            lag,
            value,
        )
        for lag, value in estimates.items()
        if value is not None
    ]

    peak_lag, peak_value = max(
        valid,
        key=lambda item: abs(
            item[1][1]
        ),
    )

    return CrossObservationNoisePairDiagnostic(
        observation_a=name_a,
        observation_b=name_b,
        status="estimated",
        alignment="registered",
        n_overlap_zero_lag=zero[2],
        zero_lag_covariance=zero[0],
        zero_lag_correlation=zero[1],
        peak_abs_correlation=abs(
            peak_value[1]
        ),
        peak_lag_y=peak_lag[0],
        peak_lag_x=peak_lag[1],
    )


def _sky_mask(
    products: Any,
    shape: tuple[int, ...],
) -> np.ndarray | None:
    sky_region = getattr(
        products,
        "sky_region",
        None,
    )

    if sky_region is None:
        return None

    mask = np.asarray(
        sky_region.mask,
        dtype=bool,
    )

    if mask.shape != shape:
        return None

    return mask


def _standardized_image(
    products: Any,
    data: np.ndarray,
) -> np.ndarray:
    noise = np.asarray(
        products.noise,
        dtype=float,
    )

    if noise.shape != data.shape:
        raise ValueError(
            "Noise map must match prepared_data "
            "for noise diagnostics."
        )

    standardized = np.full_like(
        data,
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(
            data
        )
        & np.isfinite(
            noise
        )
        & (
            noise > 0.0
        )
    )

    standardized[
        valid
    ] = (
        data[
            valid
        ]
        / noise[
            valid
        ]
    )

    return standardized


def _grid_alignment_status(
    *,
    products_a: Any,
    products_b: Any,
    config_a: Mapping[str, Any],
    config_b: Mapping[str, Any],
    registration_tolerance_pixels: float,
) -> str:
    path_a = _image_path(
        config_a
    )
    path_b = _image_path(
        config_b
    )

    if (
        path_a is not None
        and path_b is not None
        and path_a == path_b
        and getattr(
            products_a,
            "cutout_bounds",
            None,
        )
        == getattr(
            products_b,
            "cutout_bounds",
            None,
        )
    ):
        return "registered"

    try:
        wcs_a = _cutout_wcs(
            path_a,
            products_a,
        )
        wcs_b = _cutout_wcs(
            path_b,
            products_b,
        )

        if (
            wcs_a is None
            or wcs_b is None
        ):
            return "unknown"

        shape = np.asarray(
            products_a.prepared_data
        ).shape

        if len(shape) != 2:
            return "unknown"

        ny, nx = shape

        sample_x = np.asarray(
            [
                0.0,
                nx - 1.0,
                0.0,
                nx - 1.0,
                0.5 * (
                    nx - 1.0
                ),
            ]
        )

        sample_y = np.asarray(
            [
                0.0,
                0.0,
                ny - 1.0,
                ny - 1.0,
                0.5 * (
                    ny - 1.0
                ),
            ]
        )

        world_a = wcs_a.pixel_to_world(
            sample_x,
            sample_y,
        )
        world_b = wcs_b.pixel_to_world(
            sample_x,
            sample_y,
        )

        if (
            not isinstance(
                world_a,
                SkyCoord,
            )
            or not isinstance(
                world_b,
                SkyCoord,
            )
        ):
            return "unknown"

        separation = (
            world_a
            .separation(
                world_b
            )
            .to_value(
                u.arcsec
            )
        )

        scales_a = (
            np.abs(
                proj_plane_pixel_scales(
                    wcs_a.celestial
                )
            )
            * 3600.0
        )
        scales_b = (
            np.abs(
                proj_plane_pixel_scales(
                    wcs_b.celestial
                )
            )
            * 3600.0
        )

        reference_scale = float(
            min(
                np.min(
                    scales_a
                ),
                np.min(
                    scales_b
                ),
            )
        )

        tolerance = (
            registration_tolerance_pixels
            * reference_scale
        )

        if np.all(
            separation <= tolerance
        ):
            return "registered"

        return "unregistered"

    except (
        AttributeError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ):
        return "unknown"


def _image_path(
    config: Mapping[str, Any],
) -> Path | None:
    data_config = config.get(
        "data",
        {},
    )

    if not isinstance(
        data_config,
        Mapping,
    ):
        return None

    value = data_config.get(
        "image_path"
    )

    if not value:
        return None

    return Path(
        value
    ).resolve()


def _cutout_wcs(
    image_path: Path | None,
    products: Any,
):
    if image_path is None:
        return None

    bounds = getattr(
        products,
        "cutout_bounds",
        None,
    )

    if not isinstance(
        bounds,
        Mapping,
    ):
        return None

    if set(
        bounds
    ) != {
        "xmin",
        "xmax",
        "ymin",
        "ymax",
    }:
        return None

    observation = Observation.from_fits(
        image_path
    )

    wcs = observation.metadata.wcs

    if wcs is None:
        return None

    return slice_wcs_to_cutout(
        wcs,
        xmin=bounds[
            "xmin"
        ],
        xmax=bounds[
            "xmax"
        ],
        ymin=bounds[
            "ymin"
        ],
        ymax=bounds[
            "ymax"
        ],
    )


def _lagged_statistics(
    *,
    raw_a: np.ndarray,
    raw_b: np.ndarray,
    standardized_a: np.ndarray,
    standardized_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    lag_y: int,
    lag_x: int,
    minimum_overlap: int,
) -> tuple[
    float,
    float,
    int,
] | None:
    ny, nx = raw_a.shape

    a_y0 = max(
        0,
        -lag_y,
    )
    a_y1 = min(
        ny,
        ny - lag_y,
    )
    b_y0 = max(
        0,
        lag_y,
    )
    b_y1 = min(
        ny,
        ny + lag_y,
    )

    a_x0 = max(
        0,
        -lag_x,
    )
    a_x1 = min(
        nx,
        nx - lag_x,
    )
    b_x0 = max(
        0,
        lag_x,
    )
    b_x1 = min(
        nx,
        nx + lag_x,
    )

    if (
        a_y0 >= a_y1
        or a_x0 >= a_x1
    ):
        return None

    raw_view_a = raw_a[
        a_y0:a_y1,
        a_x0:a_x1,
    ]
    raw_view_b = raw_b[
        b_y0:b_y1,
        b_x0:b_x1,
    ]

    std_view_a = standardized_a[
        a_y0:a_y1,
        a_x0:a_x1,
    ]
    std_view_b = standardized_b[
        b_y0:b_y1,
        b_x0:b_x1,
    ]

    valid = (
        mask_a[
            a_y0:a_y1,
            a_x0:a_x1,
        ]
        & mask_b[
            b_y0:b_y1,
            b_x0:b_x1,
        ]
        & np.isfinite(
            raw_view_a
        )
        & np.isfinite(
            raw_view_b
        )
        & np.isfinite(
            std_view_a
        )
        & np.isfinite(
            std_view_b
        )
    )

    n_overlap = int(
        np.count_nonzero(
            valid
        )
    )

    if n_overlap < minimum_overlap:
        return None

    raw_values_a = raw_view_a[
        valid
    ]
    raw_values_b = raw_view_b[
        valid
    ]

    raw_centered_a = (
        raw_values_a
        - np.mean(
            raw_values_a
        )
    )
    raw_centered_b = (
        raw_values_b
        - np.mean(
            raw_values_b
        )
    )

    covariance = float(
        np.mean(
            raw_centered_a
            * raw_centered_b
        )
    )

    values_a = std_view_a[
        valid
    ]
    values_b = std_view_b[
        valid
    ]

    centered_a = (
        values_a
        - np.mean(
            values_a
        )
    )
    centered_b = (
        values_b
        - np.mean(
            values_b
        )
    )

    scale_a = float(
        np.sqrt(
            np.mean(
                centered_a**2
            )
        )
    )
    scale_b = float(
        np.sqrt(
            np.mean(
                centered_b**2
            )
        )
    )

    if (
        scale_a == 0.0
        or scale_b == 0.0
    ):
        return None

    correlation = (
        covariance
        / (
            float(
                np.std(
                    raw_values_a
                )
            )
            * float(
                np.std(
                    raw_values_b
                )
            )
        )
    )

    if not np.isfinite(
        correlation
    ):
        return None

    return (
        covariance,
        float(
            correlation
        ),
        n_overlap,
    )
