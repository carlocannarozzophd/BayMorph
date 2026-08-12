"""Radial-profile analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RadialProfileStatistics:
    """Summary statistics measured in concentric elliptical annuli.

    Parameters
    ----------
    radius : ndarray
        Semi-major-axis radial-bin centres in pixels.
    median : ndarray
        Median pixel intensity in each annulus.
    mean : ndarray
        Mean pixel intensity in each annulus.
    p16, p84 : ndarray
        Sixteenth and eighty-fourth percentiles of the pixel intensities.
    n_pixels : ndarray
        Number of valid pixels contributing to each annulus.

    Notes
    -----
    The interval between ``p16`` and ``p84`` describes the pixel-value
    distribution within an annulus. It is not, in general, an uncertainty
    interval for the median estimator.
    """

    radius: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    p16: np.ndarray
    p84: np.ndarray
    n_pixels: np.ndarray


def _validate_axis_ratio(
    axis_ratio: float,
) -> float:
    """Validate and return an elliptical axis ratio."""
    axis_ratio = float(
        axis_ratio
    )

    if (
        not np.isfinite(
            axis_ratio
        )
        or axis_ratio <= 0.0
        or axis_ratio > 1.0
    ):
        raise ValueError(
            "axis_ratio must be finite and satisfy 0 < axis_ratio <= 1."
        )

    return axis_ratio


def elliptical_radius_map(
    shape: tuple[int, int],
    *,
    center_x: float,
    center_y: float,
    axis_ratio: float = 1.0,
    position_angle: float = 0.0,
) -> np.ndarray:
    """Return the semi-major-axis elliptical radius of every image pixel.

    Parameters
    ----------
    shape : tuple of int
        Image shape as ``(ny, nx)``.
    center_x, center_y : float
        Ellipse centre in detector pixel coordinates.
    axis_ratio : float, default=1
        Minor-to-major axis ratio ``q = b / a``.
    position_angle : float, default=0
        Counter-clockwise major-axis angle from the positive detector x-axis,
        in radians.

    Returns
    -------
    ndarray
        Elliptical radius map in pixels.

    Notes
    -----
    A constant-radius contour is an ellipse whose semi-major axis equals the
    returned radius and whose semi-minor axis is ``q`` times that radius.
    """
    if (
        len(
            shape
        )
        != 2
        or int(
            shape[0]
        )
        <= 0
        or int(
            shape[1]
        )
        <= 0
    ):
        raise ValueError(
            "shape must contain two positive dimensions."
        )

    center_x = float(
        center_x
    )
    center_y = float(
        center_y
    )
    position_angle = float(
        position_angle
    )
    axis_ratio = _validate_axis_ratio(
        axis_ratio
    )

    if not (
        np.isfinite(
            center_x
        )
        and np.isfinite(
            center_y
        )
    ):
        raise ValueError(
            "Ellipse centre coordinates must be finite."
        )

    if not np.isfinite(
        position_angle
    ):
        raise ValueError(
            "position_angle must be finite."
        )

    yy, xx = np.indices(
        shape,
        dtype=float,
    )

    dx = xx - center_x
    dy = yy - center_y

    cos_theta = np.cos(
        position_angle
    )
    sin_theta = np.sin(
        position_angle
    )

    major_coordinate = (
        dx * cos_theta
        + dy * sin_theta
    )

    minor_coordinate = (
        -dx * sin_theta
        + dy * cos_theta
    )

    return np.sqrt(
        major_coordinate**2
        + (
            minor_coordinate
            / axis_ratio
        )
        ** 2
    )


def radial_profile_statistics_from_image(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radii: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    axis_ratio: float = 1.0,
    position_angle: float = 0.0,
) -> RadialProfileStatistics:
    """Measure pixel statistics in concentric elliptical annuli.

    Parameters
    ----------
    image : ndarray
        Input image.
    center_x, center_y : float
        Ellipse centre in detector pixel coordinates.
    radii : ndarray
        Semi-major-axis radial-bin edges in pixels.
    mask : ndarray of bool, optional
        Pixels excluded from the profile measurement.
    axis_ratio : float, default=1
        Minor-to-major axis ratio of the annuli.
    position_angle : float, default=0
        Counter-clockwise major-axis angle from the positive detector x-axis,
        in radians.

    Returns
    -------
    RadialProfileStatistics
        Median, mean, percentile range, and valid-pixel count in each annulus.
    """
    image = np.asarray(
        image,
        dtype=float,
    )

    radii = np.asarray(
        radii,
        dtype=float,
    )

    if image.ndim != 2:
        raise ValueError(
            "image must be 2D."
        )

    if radii.ndim != 1:
        raise ValueError(
            "radii must be 1D."
        )

    if radii.size < 2:
        raise ValueError(
            "radii must contain at least two edges."
        )

    if not np.all(
        np.isfinite(
            radii
        )
    ):
        raise ValueError(
            "radii must contain only finite values."
        )

    if np.any(
        np.diff(
            radii
        )
        <= 0.0
    ):
        raise ValueError(
            "radii must be strictly increasing."
        )

    radius_map = elliptical_radius_map(
        image.shape,
        center_x=center_x,
        center_y=center_y,
        axis_ratio=axis_ratio,
        position_angle=position_angle,
    )

    valid = np.isfinite(
        image
    )

    if mask is not None:
        mask = np.asarray(
            mask,
            dtype=bool,
        )

        if mask.shape != image.shape:
            raise ValueError(
                "mask must have the same shape as image."
            )

        valid &= ~mask

    n_bins = radii.size - 1

    median = np.full(
        n_bins,
        np.nan,
        dtype=float,
    )
    mean = np.full(
        n_bins,
        np.nan,
        dtype=float,
    )
    p16 = np.full(
        n_bins,
        np.nan,
        dtype=float,
    )
    p84 = np.full(
        n_bins,
        np.nan,
        dtype=float,
    )
    n_pixels = np.zeros(
        n_bins,
        dtype=int,
    )

    for index in range(
        n_bins
    ):
        annulus = (
            (
                radius_map
                >= radii[
                    index
                ]
            )
            & (
                radius_map
                < radii[
                    index + 1
                ]
            )
            & valid
        )

        values = image[
            annulus
        ]

        n_pixels[
            index
        ] = values.size

        if values.size == 0:
            continue

        median[
            index
        ] = np.median(
            values
        )

        mean[
            index
        ] = np.mean(
            values
        )

        p16[
            index
        ], p84[
            index
        ] = np.percentile(
            values,
            [
                16.0,
                84.0,
            ],
        )

    radius = 0.5 * (
        radii[:-1]
        + radii[1:]
    )

    return RadialProfileStatistics(
        radius=radius,
        median=median,
        mean=mean,
        p16=p16,
        p84=p84,
        n_pixels=n_pixels,
    )


def radial_profile_from_image(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radii: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    axis_ratio: float = 1.0,
    position_angle: float = 0.0,
) -> np.ndarray:
    """Compute the median profile in concentric elliptical annuli.

    This compatibility interface returns only the median profile. Use
    :func:`radial_profile_statistics_from_image` when the mean, percentile
    range, or contributing-pixel count is also required.
    """
    return radial_profile_statistics_from_image(
        image,
        center_x=center_x,
        center_y=center_y,
        radii=radii,
        mask=mask,
        axis_ratio=axis_ratio,
        position_angle=position_angle,
    ).median


def radial_bins(
    max_radius,
    *,
    mode="fixed",
    dr=1.0,
    min_pixels=50,
    log_n_bins=40,
    axis_ratio=1.0,
):
    """Construct semi-major-axis radial-bin edges.

    Parameters
    ----------
    max_radius : float
        Maximum semi-major-axis radius in pixels.
    mode : {"fixed", "adaptive", "logarithmic"}, default="fixed"
        Radial-binning strategy.
    dr : float, default=1
        Minimum radial-bin width in pixels.
    min_pixels : int, default=50
        Approximate minimum number of pixels in a complete elliptical annulus
        for adaptive binning.
    log_n_bins : int, default=40
        Number of logarithmically spaced bins.
    axis_ratio : float, default=1
        Minor-to-major axis ratio used to estimate elliptical-annulus areas
        in adaptive binning.

    Returns
    -------
    ndarray
        Semi-major-axis radial-bin edges in pixels.
    """
    max_radius = float(
        max_radius
    )
    dr = float(
        dr
    )
    axis_ratio = _validate_axis_ratio(
        axis_ratio
    )

    if (
        not np.isfinite(
            max_radius
        )
        or max_radius <= 0.0
    ):
        raise ValueError(
            "max_radius must be finite and positive."
        )

    if (
        not np.isfinite(
            dr
        )
        or dr <= 0.0
    ):
        raise ValueError(
            "dr must be finite and positive."
        )

    if mode == "fixed":
        edges = np.arange(
            0.0,
            max_radius,
            dr,
            dtype=float,
        )

        return np.unique(
            np.concatenate(
                (
                    edges,
                    [
                        max_radius,
                    ],
                )
            )
        )

    if mode == "logarithmic":
        inner = max(
            dr,
            1.0,
        )

        if inner >= max_radius:
            return np.asarray(
                [
                    0.0,
                    max_radius,
                ],
                dtype=float,
            )

        bins = np.geomspace(
            inner,
            max_radius,
            int(
                log_n_bins
            ),
        )

        return np.unique(
            np.concatenate(
                (
                    [
                        0.0,
                    ],
                    bins,
                )
            )
        )

    if mode == "adaptive":
        edges = [
            0.0,
        ]
        radius = 0.0

        while radius < max_radius:
            circumference_scale = (
                2.0
                * np.pi
                * axis_ratio
                * max(
                    radius,
                    1.0,
                )
            )

            dr_current = max(
                dr,
                float(
                    min_pixels
                )
                / max(
                    circumference_scale,
                    1.0,
                ),
            )

            radius = min(
                radius
                + dr_current,
                max_radius,
            )

            edges.append(
                radius
            )

        return np.asarray(
            edges,
            dtype=float,
        )

    raise ValueError(
        "Unknown radial binning mode "
        f"'{mode}'. Expected one of "
        "'fixed', 'adaptive', or 'logarithmic'."
    )
