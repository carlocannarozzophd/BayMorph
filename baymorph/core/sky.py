"""Sky-region and elliptical-geometry utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baymorph.stats.robust import robust_sigma


@dataclass(slots=True)
class RadialProfile:
    """Elliptical radial profile."""

    radius: np.ndarray
    profile: np.ndarray
    n_pixels: np.ndarray
    scatter: np.ndarray | None = None
    area: np.ndarray | None = None


@dataclass(slots=True)
class PlateauResult:
    """Result of the radial-profile plateau detection.

    Parameters
    ----------
    radius : float
        Radius at which the first stable plateau window begins.
    index : int
        Index of the first radial bin belonging to the plateau.
    background_level : float
        Median profile value within the detected plateau window.
    noise_sigma : float
        Robust scatter estimated from the outer radial profile.
    window_start : int
        First index of the accepted plateau window.
    window_stop : int
        Exclusive final index of the accepted plateau window.
    """

    radius: float
    index: int
    background_level: float
    noise_sigma: float
    window_start: int
    window_stop: int


@dataclass(slots=True)
class SkyRegion:
    """Automatically identified sky region.

    Parameters
    ----------
    mask : ndarray of bool
        Boolean mask selecting sky pixels.
    radius_map : ndarray
        Elliptical radius map used to define the sky region.
    plateau : PlateauResult
        Plateau-detection result.
    inner_radius : float
        Inner elliptical radius of the sky region.
    outer_radius : float
        Outer elliptical radius of the sky region.
    n_pixels : int
        Number of selected sky pixels.
    coverage : float
        Fraction of geometrically available pixels retained as sky.
    """

    mask: np.ndarray
    radius_map: np.ndarray
    plateau: PlateauResult
    inner_radius: float
    outer_radius: float
    n_pixels: int
    coverage: float


def build_elliptical_radius_map(
    shape: tuple[int, int], x0: float, y0: float, q: float, theta: float
) -> np.ndarray:
    """Compute the elliptical radius map of an image.

    The returned array contains, for every pixel, the semi-major-axis
    distance from the reference centre assuming elliptical isophotes with
    axis ratio ``q`` and position angle ``theta``.

    Parameters
    ----------
    shape : tuple of int
        Image shape as ``(ny, nx)``.
    x0 : float
        X-coordinate of the ellipse centre.
    y0 : float
        Y-coordinate of the ellipse centre.
    q : float
        Minor-to-major axis ratio (``0 < q <= 1``).
    theta : float
        Position angle in radians, measured counter-clockwise from the
        positive x-axis.

    Returns
    -------
    ndarray
        Two-dimensional elliptical radius map.

    Raises
    ------
    ValueError
        If ``q`` is not in the interval ``(0, 1]``.
    """
    if not (0.0 < q <= 1.0):
        raise ValueError("Axis ratio 'q' must satisfy 0 < q <= 1.")

    yy, xx = np.indices(shape, dtype=float)

    dx = xx - x0
    dy = yy - y0

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_rot = dx * cos_theta + dy * sin_theta
    y_rot = -dx * sin_theta + dy * cos_theta

    return np.sqrt(x_rot**2 + (y_rot / q) ** 2)


def compute_elliptical_radial_profile(
    image: np.ndarray,
    x0: float,
    y0: float,
    q: float,
    theta: float,
    *,
    mask: np.ndarray | None = None,
    statistic: str = "median",
    min_pixels: int = 1,
) -> RadialProfile:
    """Compute an elliptical radial profile.

    Parameters
    ----------
    image : ndarray
        Input image.
    x0, y0 : float
        Centre of the elliptical isophotes.
    q : float
        Minor-to-major axis ratio.
    theta : float
        Position angle in radians.
    mask : ndarray of bool, optional
        Pixels to exclude from the profile.
    statistic : {"median", "mean", "sum"}, default="median"
        Statistic computed in each annulus.
    min_pixels : int, default=1
        Minimum number of valid pixels required to compute the statistic.
        Annuli with fewer valid pixels are assigned ``NaN``.

    Returns
    -------
    RadialProfile
        Elliptical radial profile.
    """
    image = np.asarray(image, dtype=float)

    radius_map = build_elliptical_radius_map(
        image.shape, x0=x0, y0=y0, q=q, theta=theta
    )

    radius_index = np.floor(radius_map).astype(int)

    valid = np.isfinite(image)

    if mask is not None:
        valid &= ~mask

    max_radius = radius_index.max()

    radius = []
    profile = []
    n_pixels = []
    scatter = []
    area = []

    for r in range(max_radius + 1):

        full_annulus = radius_index == r
        annulus = valid & full_annulus

        npix = np.count_nonzero(annulus)

        radius.append(float(r))
        n_pixels.append(npix)
        area.append(np.count_nonzero(full_annulus))

        if npix < min_pixels:
            profile.append(np.nan)
            scatter.append(np.nan)
            continue

        values = image[annulus]

        if statistic == "median":
            value = np.median(values)

        elif statistic == "mean":
            value = np.mean(values)

        elif statistic == "sum":
            value = np.sum(values)

        else:
            raise ValueError(f"Unknown statistic: {statistic!r}")

        profile.append(float(value))

        if npix > 1:
            scatter.append(float(np.std(values, ddof=1)))
        else:
            scatter.append(0.0)

    return RadialProfile(
        radius=np.asarray(radius, dtype=float),
        profile=np.asarray(profile, dtype=float),
        n_pixels=np.asarray(n_pixels, dtype=int),
        scatter=np.asarray(scatter, dtype=float),
        area=np.asarray(area, dtype=int),
    )


def detect_profile_plateau(
    radial_profile: RadialProfile,
    *,
    min_radius: float = 0.0,
    window_size: int = 5,
    sigma_threshold: float = 1.0,
    outer_fraction: float = 0.25,
    min_coverage: float = 0.5,
) -> PlateauResult:
    """Detect the first stable plateau in an elliptical radial profile.

    The noise scale is estimated robustly from the outer part of the
    profile. A plateau is identified as the first sequence of consecutive
    radial bins whose values are mutually consistent within
    ``sigma_threshold * noise_sigma``.

    Parameters
    ----------
    radial_profile : RadialProfile
        Elliptical radial profile to analyse.
    min_radius : float, default=0.0
        Minimum radius from which plateau detection is allowed.
    window_size : int, default=5
        Number of consecutive radial bins required for a plateau.
    sigma_threshold : float, default=1.0
        Maximum allowed deviation from the window median, in units of the
        robust outer-profile scatter.
    outer_fraction : float, default=0.25
        Fraction of the outer valid radial bins used to estimate the noise.
    min_coverage : float, default=0.5
        Minimum accepted ratio between valid and geometrically available
        pixels in each radial bin.

    Returns
    -------
    PlateauResult
        Detected plateau properties.

    Raises
    ------
    ValueError
        If the configuration is invalid or no plateau can be detected.
    """
    if window_size < 2:
        raise ValueError("'window_size' must be at least 2.")

    if sigma_threshold <= 0.0:
        raise ValueError("'sigma_threshold' must be positive.")

    if not (0.0 < outer_fraction <= 1.0):
        raise ValueError("'outer_fraction' must satisfy 0 < value <= 1.")

    if not (0.0 <= min_coverage <= 1.0):
        raise ValueError("'min_coverage' must satisfy 0 <= value <= 1.")

    radius = np.asarray(radial_profile.radius, dtype=float)
    values = np.asarray(radial_profile.profile, dtype=float)
    n_pixels = np.asarray(radial_profile.n_pixels, dtype=float)

    if radial_profile.area is None:
        coverage = np.ones_like(values, dtype=float)
    else:
        area = np.asarray(radial_profile.area, dtype=float)
        coverage = np.divide(
            n_pixels, area, out=np.zeros_like(n_pixels, dtype=float), where=area > 0
        )

    valid = (
        np.isfinite(radius)
        & np.isfinite(values)
        & np.isfinite(coverage)
        & (radius >= min_radius)
        & (coverage >= min_coverage)
    )

    valid_indices = np.flatnonzero(valid)

    if valid_indices.size < window_size:
        raise ValueError("Not enough valid radial bins for plateau detection.")

    n_outer = max(window_size, int(np.ceil(outer_fraction * valid_indices.size)))

    outer_indices = valid_indices[-n_outer:]
    outer_values = values[outer_indices]

    outer_median = float(np.median(outer_values))
    noise_sigma = robust_sigma(outer_values, center=outer_median)

    if not np.isfinite(noise_sigma):
        raise ValueError("Unable to estimate a finite outer-profile noise level.")

    numerical_floor = 10.0 * np.finfo(float).eps * max(1.0, abs(outer_median))

    tolerance = max(sigma_threshold * noise_sigma, numerical_floor)

    for start in range(values.size - window_size + 1):
        stop = start + window_size
        window_indices = np.arange(start, stop)

        if not np.all(valid[window_indices]):
            continue

        window_values = values[window_indices]
        window_median = float(np.median(window_values))

        deviations = np.abs(window_values - window_median)

        window_x = np.arange(window_size, dtype=float)
        slope = float(np.polyfit(window_x, window_values, deg=1)[0])
        total_trend = abs(slope) * (window_size - 1)

        is_stable = np.all(deviations <= tolerance)
        is_flat = total_trend <= tolerance
        is_background_consistent = abs(window_median - outer_median) <= tolerance

        if is_stable and is_flat and is_background_consistent:
            return PlateauResult(
                radius=float(radius[start]),
                index=start,
                background_level=window_median,
                noise_sigma=noise_sigma,
                window_start=start,
                window_stop=stop,
            )

    raise ValueError("No stable radial-profile plateau was detected.")


def build_sky_region(
    shape: tuple[int, int],
    geometry,
    plateau: PlateauResult,
    *,
    margin: float = 5.0,
    outer_radius: float | None = None,
    mask: np.ndarray | None = None,
) -> SkyRegion:
    """Build an elliptical sky region from a detected plateau.

    Parameters
    ----------
    shape : tuple of int
        Image shape as ``(ny, nx)``.
    geometry : object
        Object exposing ``x0``, ``y0``, ``q``, and ``theta`` attributes.
    plateau : PlateauResult
        Detected radial-profile plateau.
    margin : float, default=5.0
        Additional radial offset applied beyond the plateau radius.
    outer_radius : float, optional
        Maximum elliptical radius included in the sky region. If omitted,
        all available pixels beyond ``inner_radius`` are considered.
    mask : ndarray of bool, optional
        Pixels excluded from the sky region.

    Returns
    -------
    SkyRegion
        Selected sky region and its geometric metadata.

    Raises
    ------
    ValueError
        If the margin is negative or the selected region contains no pixels.
    """
    if margin < 0.0:
        raise ValueError("'margin' must be non-negative.")

    inner_radius = float(plateau.radius + margin)

    radius_map = build_elliptical_radius_map(
        shape, x0=geometry.x0, y0=geometry.y0, q=geometry.q, theta=geometry.theta
    )

    if outer_radius is None:
        resolved_outer_radius = float(np.max(radius_map))
    else:
        resolved_outer_radius = float(outer_radius)

    if resolved_outer_radius <= inner_radius:
        raise ValueError("'outer_radius' must be greater than the inner sky radius.")

    geometrical_region = (radius_map >= inner_radius) & (
        radius_map <= resolved_outer_radius
    )

    sky_mask = geometrical_region.copy()

    if mask is not None:
        sky_mask &= ~np.asarray(mask, dtype=bool)

    n_pixels = int(np.count_nonzero(sky_mask))
    available_pixels = int(np.count_nonzero(geometrical_region))

    if n_pixels == 0:
        raise ValueError("The selected sky region contains no valid pixels.")

    coverage = float(n_pixels / available_pixels) if available_pixels > 0 else 0.0

    return SkyRegion(
        mask=sky_mask,
        radius_map=radius_map,
        plateau=plateau,
        inner_radius=inner_radius,
        outer_radius=resolved_outer_radius,
        n_pixels=n_pixels,
        coverage=coverage,
    )
