"""Atomic steps for intelligent image preparation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baymorph.core.sky import (
    PlateauResult,
    RadialProfile,
    compute_elliptical_radial_profile,
    detect_profile_plateau,
)
from baymorph.pipeline.cutout import CutoutDefinition


@dataclass(slots=True)
class TargetGeometry:
    """Initial geometric estimate of a target source.

    Parameters
    ----------
    x0, y0 : float
        Estimated target centre in image coordinates.
    q : float
        Estimated minor-to-major axis ratio.
    theta : float
        Position angle in radians, measured counter-clockwise from the
        positive x-axis.
    """

    x0: float
    y0: float
    q: float
    theta: float


@dataclass(slots=True)
class AutoCutoutResult:
    """Products generated while resolving an automatic target cutout.

    Parameters
    ----------
    cutout : CutoutDefinition
        Final plateau-based cutout definition.
    preliminary_cutout : ndarray
        Preliminary adaptive cutout used to estimate the target profile.
    preliminary_segmentation : ndarray
        Segmentation map measured on the preliminary cutout.
    geometry : TargetGeometry
        Target geometry estimated on the preliminary cutout.
    radial_profile : RadialProfile
        Elliptical radial profile measured on the preliminary cutout.
    plateau : PlateauResult
        Detected sky-profile plateau.
    """

    cutout: CutoutDefinition
    preliminary_cutout: np.ndarray
    preliminary_segmentation: np.ndarray
    geometry: TargetGeometry
    radial_profile: RadialProfile
    plateau: PlateauResult


def estimate_target_geometry_from_segment(target_segment: np.ndarray) -> TargetGeometry:
    """Estimate target geometry from a boolean segmentation mask.

    Parameters
    ----------
    target_segment : ndarray of bool
        Boolean mask identifying the target source.

    Returns
    -------
    TargetGeometry
        Initial centre, axis ratio, and position-angle estimates.

    Raises
    ------
    ValueError
        If the segment is not two-dimensional or contains fewer than three
        pixels.
    """
    segment = np.asarray(target_segment, dtype=bool)

    if segment.ndim != 2:
        raise ValueError("'target_segment' must be a two-dimensional array.")

    y, x = np.nonzero(segment)

    if x.size < 3:
        raise ValueError(
            "At least three target-segment pixels are required "
            "to estimate the source geometry."
        )

    x0 = float(np.mean(x))
    y0 = float(np.mean(y))

    dx = x.astype(float) - x0
    dy = y.astype(float) - y0

    covariance = np.array(
        [[np.mean(dx * dx), np.mean(dx * dy)], [np.mean(dx * dy), np.mean(dy * dy)]],
        dtype=float,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    major_variance = float(eigenvalues[0])
    minor_variance = float(eigenvalues[1])

    if major_variance <= 0.0:
        q = 1.0
        theta = 0.0
    else:
        q = float(np.sqrt(max(minor_variance, 0.0) / major_variance))
        q = float(np.clip(q, np.finfo(float).eps, 1.0))

        major_axis = eigenvectors[:, 0]
        theta = float(np.arctan2(major_axis[1], major_axis[0]))

    return TargetGeometry(x0=x0, y0=y0, q=q, theta=theta)


def estimate_target_geometry_from_image(
    image: np.ndarray, target_segment: np.ndarray
) -> TargetGeometry:
    """Estimate target geometry using image-weighted source moments.

    Parameters
    ----------
    image : ndarray
        Two-dimensional source image.
    target_segment : ndarray of bool
        Boolean mask identifying the target source.

    Returns
    -------
    TargetGeometry
        Flux-weighted centre, axis ratio, and position-angle estimates.

    Raises
    ------
    ValueError
        If the image and segment are invalid, have different shapes, or
        contain insufficient positive source signal.
    """
    data = np.asarray(image, dtype=float)
    segment = np.asarray(target_segment, dtype=bool)

    if data.ndim != 2 or segment.ndim != 2:
        raise ValueError("'image' and 'target_segment' must be two-dimensional arrays.")

    if data.shape != segment.shape:
        raise ValueError("'image' and 'target_segment' must have the same shape.")

    finite_source = segment & np.isfinite(data)

    if np.count_nonzero(finite_source) < 3:
        raise ValueError(
            "At least three finite target pixels are required "
            "to estimate the source geometry."
        )

    source_values = data[finite_source]
    baseline = float(np.median(source_values))
    weights = np.clip(data - baseline, 0.0, None)
    weights = np.where(finite_source, weights, 0.0)

    total_weight = float(np.sum(weights))

    if not np.isfinite(total_weight) or total_weight <= 0.0:
        return estimate_target_geometry_from_segment(segment)

    y, x = np.indices(data.shape, dtype=float)

    x0 = float(np.sum(weights * x) / total_weight)
    y0 = float(np.sum(weights * y) / total_weight)

    dx = x - x0
    dy = y - y0

    covariance = np.array(
        [
            [
                np.sum(weights * dx * dx) / total_weight,
                np.sum(weights * dx * dy) / total_weight,
            ],
            [
                np.sum(weights * dx * dy) / total_weight,
                np.sum(weights * dy * dy) / total_weight,
            ],
        ],
        dtype=float,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    major_variance = float(eigenvalues[0])
    minor_variance = float(eigenvalues[1])

    if major_variance <= 0.0:
        q = 1.0
        theta = 0.0
    else:
        q = float(np.sqrt(max(minor_variance, 0.0) / major_variance))
        q = float(np.clip(q, np.finfo(float).eps, 1.0))

        major_axis = eigenvectors[:, 0]
        theta = float(np.arctan2(major_axis[1], major_axis[0]))

    return TargetGeometry(x0=x0, y0=y0, q=q, theta=theta)


def build_target_radial_profile(
    image: np.ndarray,
    geometry: TargetGeometry,
    *,
    mask: np.ndarray | None = None,
    statistic: str = "median",
    min_pixels: int = 1,
) -> RadialProfile:
    """Build an elliptical radial profile from target geometry.

    Parameters
    ----------
    image : ndarray
        Input image.
    geometry : TargetGeometry
        Initial target geometry.
    mask : ndarray of bool, optional
        Pixels excluded from the radial profile.
    statistic : {"median", "mean", "sum"}, default="median"
        Statistic computed within each elliptical annulus.
    min_pixels : int, default=1
        Minimum number of valid pixels required in each annulus.

    Returns
    -------
    RadialProfile
        Elliptical radial profile of the target.
    """
    return compute_elliptical_radial_profile(
        image,
        x0=geometry.x0,
        y0=geometry.y0,
        q=geometry.q,
        theta=geometry.theta,
        mask=mask,
        statistic=statistic,
        min_pixels=min_pixels,
    )


def estimate_target_sky(
    image: np.ndarray,
    target_segment: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    statistic: str = "median",
    min_pixels: int = 1,
    min_radius: float = 0.0,
    window_size: int = 5,
    sigma_threshold: float = 1.0,
    outer_fraction: float = 0.25,
    min_coverage: float = 0.5,
) -> tuple[TargetGeometry, RadialProfile, PlateauResult]:
    """Estimate target geometry, radial profile, and sky plateau."""
    geometry = estimate_target_geometry_from_image(image, target_segment)

    radial_profile = build_target_radial_profile(
        image, geometry, mask=mask, statistic=statistic, min_pixels=min_pixels
    )

    plateau = detect_profile_plateau(
        radial_profile,
        min_radius=min_radius,
        window_size=window_size,
        sigma_threshold=sigma_threshold,
        outer_fraction=outer_fraction,
        min_coverage=min_coverage,
    )

    return geometry, radial_profile, plateau
