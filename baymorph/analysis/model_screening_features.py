"""Interpretable image features for BayMorph model screening."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    binary_dilation,
    label,
)


FloatArray = NDArray[np.float64]


SCREENING_FEATURE_NAMES = (
    "log10_positive_flux_snr",
    "log10_peak_snr",
    "r20_pixels",
    "r50_pixels",
    "r80_pixels",
    "concentration_log10_r80_r20",
    "resolution_r50_over_fwhm",
    "axis_ratio_moment",
    "centroid_offset_over_fwhm",
    "central_fraction_half_fwhm",
    "central_fraction_one_fwhm",
    "outer_fraction_beyond_two_r50",
)


@dataclass(frozen=True, slots=True)
class ModelScreeningFeatureVector:
    """One deterministic model-screening feature vector."""

    values: FloatArray
    names: tuple[str, ...] = SCREENING_FEATURE_NAMES

    def __post_init__(self) -> None:
        """Validate and freeze the feature vector."""
        values = np.asarray(
            self.values,
            dtype=float,
        )
        names = tuple(
            str(name)
            for name in self.names
        )

        if values.ndim != 1:
            raise ValueError(
                "Model-screening features must be one-dimensional."
            )

        if len(
            values
        ) != len(
            names
        ):
            raise ValueError(
                "Model-screening feature values and names must "
                "have equal lengths."
            )

        if len(
            set(
                names
            )
        ) != len(
            names
        ):
            raise ValueError(
                "Model-screening feature names must be unique."
            )

        if not np.all(
            np.isfinite(
                values
            )
        ):
            raise ValueError(
                "Model-screening features must be finite."
            )

        values.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "values",
            values,
        )
        object.__setattr__(
            self,
            "names",
            names,
        )

    def as_mapping(self) -> dict[str, float]:
        """Return the feature vector as a mutable mapping."""
        return {
            name: float(
                value
            )
            for name, value in zip(
                self.names,
                self.values,
                strict=True,
            )
        }


def _border_values(
    image: FloatArray,
) -> FloatArray:
    """Return finite pixels from a deterministic outer border."""
    border_width = max(
        1,
        min(
            image.shape
        )
        // 10,
    )

    sections = [
        image[
            :border_width,
            :,
        ].ravel(),
        image[
            -border_width:,
            :,
        ].ravel(),
    ]

    if image.shape[0] > 2 * border_width:
        sections.extend(
            [
                image[
                    border_width:-border_width,
                    :border_width,
                ].ravel(),
                image[
                    border_width:-border_width,
                    -border_width:,
                ].ravel(),
            ]
        )

    border = np.concatenate(
        sections
    )

    return border[
        np.isfinite(
            border
        )
    ]


def _resolve_background(
    border: FloatArray,
    background: float | None,
) -> float:
    """Return a supplied or robustly estimated background."""
    if background is None:
        value = float(
            np.median(
                border
            )
        )
    else:
        value = float(
            background
        )

    if not np.isfinite(
        value
    ):
        raise ValueError(
            "Screening background must be finite."
        )

    return value


def _resolve_noise(
    border: FloatArray,
    noise_sigma: float | None,
) -> float:
    """Return a supplied or robustly estimated noise scale."""
    if noise_sigma is None:
        median = float(
            np.median(
                border
            )
        )

        value = float(
            1.482602218505602
            * np.median(
                np.abs(
                    border
                    - median
                )
            )
        )

        if value <= 0.0:
            value = float(
                np.std(
                    border
                )
            )
    else:
        value = float(
            noise_sigma
        )

    if not np.isfinite(
        value
    ) or value <= 0.0:
        raise ValueError(
            "Screening noise sigma must be finite and positive."
        )

    return value


def _detect_target_mask(
    data: FloatArray,
    *,
    finite: NDArray[np.bool_],
    target_x: float,
    target_y: float,
    psf_fwhm: float,
    background: float,
    noise_sigma: float,
) -> NDArray[np.bool_]:
    """Build a temporary target-connected screening mask."""
    detection = (
        finite
        & (
            data
            - background
            > 1.5
            * noise_sigma
        )
    )

    labels, n_labels = label(
        detection
    )

    if n_labels == 0:
        raise ValueError(
            "No provisional source segment was detected near the target."
        )

    seed_x = int(
        np.rint(
            target_x
        )
    )
    seed_y = int(
        np.rint(
            target_y
        )
    )

    target_label = int(
        labels[
            seed_y,
            seed_x,
        ]
    )

    if target_label == 0:
        candidate_pixels = np.argwhere(
            labels
            > 0
        )

        squared_distances = (
            (
                candidate_pixels[
                    :,
                    1,
                ]
                - target_x
            )
            ** 2
            + (
                candidate_pixels[
                    :,
                    0,
                ]
                - target_y
            )
            ** 2
        )

        nearest_index = int(
            np.argmin(
                squared_distances
            )
        )
        nearest_distance = float(
            np.sqrt(
                squared_distances[
                    nearest_index
                ]
            )
        )
        maximum_distance = max(
            3.0,
            2.0
            * psf_fwhm,
        )

        if nearest_distance > maximum_distance:
            raise ValueError(
                "No provisional source segment was detected near "
                "the target seed."
            )

        nearest_pixel = candidate_pixels[
            nearest_index
        ]

        target_label = int(
            labels[
                nearest_pixel[0],
                nearest_pixel[1],
            ]
        )

    target_mask = (
        labels
        == target_label
    )

    target_mask = binary_dilation(
        target_mask,
        iterations=max(
            1,
            int(
                np.ceil(
                    psf_fwhm
                )
            ),
        ),
    )

    return (
        target_mask
        & finite
    )


def _enclosed_radius(
    sorted_radius: FloatArray,
    cumulative_flux: FloatArray,
    *,
    total_flux: float,
    fraction: float,
) -> float:
    """Return the circular radius enclosing one flux fraction."""
    index = int(
        np.searchsorted(
            cumulative_flux,
            fraction
            * total_flux,
            side="left",
        )
    )

    index = min(
        index,
        len(
            sorted_radius
        )
        - 1,
    )

    return max(
        float(
            sorted_radius[
                index
            ]
        ),
        np.finfo(
            float
        ).eps,
    )


def extract_model_screening_features(
    image: NDArray[np.floating],
    *,
    target_x: float,
    target_y: float,
    psf_fwhm: float,
    background: float | None = None,
    noise_sigma: float | None = None,
    target_mask: NDArray[np.bool_] | None = None,
) -> ModelScreeningFeatureVector:
    """Extract interpretable morphology features from one target image.

    This function is non-destructive. When no target mask is supplied,
    it constructs a temporary connected segment for screening only.
    It does not create or modify production preparation products.
    """
    data = np.asarray(
        image,
        dtype=float,
    )

    if data.ndim != 2:
        raise ValueError(
            "Model-screening feature extraction requires a 2D image."
        )

    finite = np.isfinite(
        data
    )

    if np.count_nonzero(
        finite
    ) < 16:
        raise ValueError(
            "Model-screening feature extraction requires at least "
            "16 finite pixels."
        )

    target_x = float(
        target_x
    )
    target_y = float(
        target_y
    )
    psf_fwhm = float(
        psf_fwhm
    )

    if not np.isfinite(
        target_x
    ) or not np.isfinite(
        target_y
    ):
        raise ValueError(
            "Target coordinates must be finite."
        )

    if not (
        0.0
        <= target_x
        <= data.shape[1]
        - 1
        and 0.0
        <= target_y
        <= data.shape[0]
        - 1
    ):
        raise ValueError(
            "Target coordinates must lie inside the image."
        )

    if not np.isfinite(
        psf_fwhm
    ) or psf_fwhm <= 0.0:
        raise ValueError(
            "PSF FWHM must be finite and positive."
        )

    border = _border_values(
        data
    )

    if border.size == 0:
        raise ValueError(
            "Cannot estimate screening scales from an empty border."
        )

    background_value = _resolve_background(
        border,
        background,
    )
    noise_value = _resolve_noise(
        border,
        noise_sigma,
    )

    if target_mask is None:
        diagnostic_mask = _detect_target_mask(
            data,
            finite=finite,
            target_x=target_x,
            target_y=target_y,
            psf_fwhm=psf_fwhm,
            background=background_value,
            noise_sigma=noise_value,
        )
    else:
        diagnostic_mask = np.asarray(
            target_mask,
            dtype=bool,
        ).copy()

        if diagnostic_mask.shape != data.shape:
            raise ValueError(
                "The provisional target mask must match the image shape."
            )

        diagnostic_mask &= finite

        if not np.any(
            diagnostic_mask
        ):
            raise ValueError(
                "The provisional target mask must contain usable pixels."
            )

    signal = np.where(
        diagnostic_mask,
        np.clip(
            data
            - background_value,
            0.0,
            None,
        ),
        0.0,
    )

    total_flux = float(
        np.sum(
            signal
        )
    )

    if not np.isfinite(
        total_flux
    ) or total_flux <= 0.0:
        raise ValueError(
            "The provisional target segment contains no positive signal."
        )

    y, x = np.indices(
        data.shape,
        dtype=float,
    )

    dx = (
        x
        - target_x
    )
    dy = (
        y
        - target_y
    )
    radius = np.hypot(
        dx,
        dy,
    )

    order = np.argsort(
        radius.ravel()
    )
    sorted_radius = radius.ravel()[
        order
    ]
    cumulative_flux = np.cumsum(
        signal.ravel()[
            order
        ]
    )

    r20 = _enclosed_radius(
        sorted_radius,
        cumulative_flux,
        total_flux=total_flux,
        fraction=0.2,
    )
    r50 = _enclosed_radius(
        sorted_radius,
        cumulative_flux,
        total_flux=total_flux,
        fraction=0.5,
    )
    r80 = _enclosed_radius(
        sorted_radius,
        cumulative_flux,
        total_flux=total_flux,
        fraction=0.8,
    )

    centroid_x = float(
        np.sum(
            signal
            * x
        )
        / total_flux
    )
    centroid_y = float(
        np.sum(
            signal
            * y
        )
        / total_flux
    )

    centered_x = (
        x
        - centroid_x
    )
    centered_y = (
        y
        - centroid_y
    )

    moment_xx = float(
        np.sum(
            signal
            * centered_x**2
        )
        / total_flux
    )
    moment_yy = float(
        np.sum(
            signal
            * centered_y**2
        )
        / total_flux
    )
    moment_xy = float(
        np.sum(
            signal
            * centered_x
            * centered_y
        )
        / total_flux
    )

    moment_matrix = np.array(
        [
            [
                moment_xx,
                moment_xy,
            ],
            [
                moment_xy,
                moment_yy,
            ],
        ]
    )

    eigenvalues = np.linalg.eigvalsh(
        moment_matrix
    )

    major_moment = max(
        float(
            eigenvalues[-1]
        ),
        np.finfo(
            float
        ).eps,
    )
    minor_moment = max(
        float(
            eigenvalues[0]
        ),
        0.0,
    )

    axis_ratio = float(
        np.sqrt(
            minor_moment
            / major_moment
        )
    )

    centroid_offset = float(
        np.hypot(
            centroid_x
            - target_x,
            centroid_y
            - target_y,
        )
    )

    central_half_flux = float(
        np.sum(
            signal[
                radius
                <= 0.5
                * psf_fwhm
            ]
        )
    )
    central_one_flux = float(
        np.sum(
            signal[
                radius
                <= psf_fwhm
            ]
        )
    )
    outer_flux = float(
        np.sum(
            signal[
                radius
                > 2.0
                * r50
            ]
        )
    )

    positive_pixels = max(
        int(
            np.count_nonzero(
                signal
                > 0.0
            )
        ),
        1,
    )

    integrated_snr = (
        total_flux
        / (
            noise_value
            * np.sqrt(
                float(
                    positive_pixels
                )
            )
        )
    )
    peak_snr = float(
        np.max(
            signal
        )
        / noise_value
    )

    values = np.array(
        [
            np.log10(
                max(
                    integrated_snr,
                    np.finfo(
                        float
                    ).tiny,
                )
            ),
            np.log10(
                max(
                    peak_snr,
                    np.finfo(
                        float
                    ).tiny,
                )
            ),
            r20,
            r50,
            r80,
            np.log10(
                r80
                / r20
            ),
            r50
            / psf_fwhm,
            axis_ratio,
            centroid_offset
            / psf_fwhm,
            central_half_flux
            / total_flux,
            central_one_flux
            / total_flux,
            outer_flux
            / total_flux,
        ],
        dtype=float,
    )

    return ModelScreeningFeatureVector(
        values=values
    )


__all__ = [
    "ModelScreeningFeatureVector",
    "SCREENING_FEATURE_NAMES",
    "extract_model_screening_features",
]
