"""Sky-region and elliptical-geometry utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RadialProfile:
    """Elliptical radial profile."""

    radius: np.ndarray
    profile: np.ndarray
    n_pixels: np.ndarray
    scatter: np.ndarray | None = None
    area: np.ndarray | None = None


def build_elliptical_radius_map(
    shape: tuple[int, int],
    x0: float,
    y0: float,
    q: float,
    theta: float,
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

    return np.sqrt(
        x_rot**2 +
        (y_rot / q) ** 2,
    )


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
        image.shape,
        x0=x0,
        y0=y0,
        q=q,
        theta=theta,
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
            raise ValueError(
                f"Unknown statistic: {statistic!r}"
            )

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