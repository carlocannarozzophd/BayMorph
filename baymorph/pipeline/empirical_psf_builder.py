"""Empirical PSF builder of AD-046: phase-regression core and measured wings.

The builder turns the stars selected for the PSF into the effective PSF at
integer offsets and at the ``SUBPIXEL_FACTOR`` sub-pixel phases of each axis.

* The sky of each star is the sigma-clipped median of a ring far beyond its
  light, with the other detected sources masked.
* The core is a weighted least-squares quadratic in the sub-pixel phase of
  the stars, fitted pixel by pixel to the stars normalized by their flux in
  the common footprint and moved by whole pixels only, so no star is
  interpolated.
* The wings are the inverse-variance mean of the stars' azimuthal profiles,
  in rings about their windowed centroids out to the support of the kernel,
  with the other sources masked and negative rings set to zero.
* At every phase the core, evaluated there, is blended into the wings,
  evaluated at the radius from the displaced centre, between ``WING_START``
  and ``WING_START + WING_BLEND``; negative pixels are set to zero and the
  kernel is normalized.

The algorithm is the candidate W1 of
``docs/research/PSF_EXTENDED_WINGS_2026-10-01.md``, validated with the
rendering of AD-046 on synthetic fields; its fixed constants are those of
that validation, and its radii are configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy.stats import sigma_clipped_stats
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation

from baymorph.rendering.effective_psf import SUBPIXEL_FACTOR

FloatArray = NDArray[np.float64]

WINDOW_SIGMA = 1.2
CENTROID_MAX_ITERATIONS = 100
CENTROID_TOLERANCE = 1e-5
WING_START = 6.0
WING_BLEND = 2.0
SIGMA_CLIP = 3.0
MIN_RING_PIXELS = 3


@dataclass(frozen=True)
class BuilderSettings:
    """Radii of the builder, in pixels.

    Attributes
    ----------
    sky_inner_radius, sky_outer_radius : float
        Ring in which the sky of each star is measured. It must lie beyond
        the light of the star.
    mask_growth_pixels : int
        Growth of the detected segments of other sources before masking.
    support_half_size : int
        Half-size of the kernel support; the wings are measured out to it.
    """

    sky_inner_radius: float = 80.0
    sky_outer_radius: float = 160.0
    mask_growth_pixels: int = 5
    support_half_size: int = 64

    def __post_init__(self) -> None:
        if not (
            0 < self.support_half_size < self.sky_inner_radius
            < self.sky_outer_radius
        ):
            raise ValueError(
                "The PSF builder needs 0 < support_half_size < "
                "sky_inner_radius < sky_outer_radius."
            )

        if int(self.mask_growth_pixels) != self.mask_growth_pixels or (
            self.mask_growth_pixels < 0
        ):
            raise ValueError(
                "mask_growth_pixels must be a non-negative integer."
            )

    @property
    def ring_edges(self) -> FloatArray:
        """Return the ring edges: finer near the core, wider in the wings."""
        return np.concatenate(
            [
                np.arange(0.0, 8.0, 0.5),
                np.arange(8.0, 20.0, 1.0),
                np.arange(20.0, 40.0, 2.0),
                np.arange(40.0, self.support_half_size + 4.0, 4.0),
            ]
        )


@dataclass(frozen=True)
class EmpiricalPSFBuild:
    """Result of the builder.

    Attributes
    ----------
    native : ndarray
        The effective PSF at integer offsets, the zero phase.
    subpixel_kernels : ndarray
        The effective PSF at every sub-pixel phase, shaped
        ``(f, f, size, size)`` as :class:`baymorph.simulation.psf.EmpiricalPSF`
        expects.
    wing_profile : dict
        Ring radius, value and standard error of the combined wing profile,
        before negative rings are set to zero.
    stars : list of dict
        For each star used: position, centroid, sky, noise, flux in the
        common footprint and sub-pixel phase.
    windows : list of ndarray
        The normalized star windows on the common footprint.
    """

    native: FloatArray
    subpixel_kernels: FloatArray
    wing_profile: dict = field(repr=False)
    stars: list = field(repr=False)
    windows: list = field(repr=False)


def window_centroid(
    stamp: FloatArray,
    x: float,
    y: float,
    *,
    sigma: float = WINDOW_SIGMA,
) -> tuple[float, float]:
    """Iterate a Gaussian-windowed centroid on a sky-subtracted stamp.

    The window follows the estimate, so for a profile symmetric about its
    centre the fixed point is that centre; there is no pedestal.
    """
    data = np.asarray(stamp, dtype=float)
    yy, xx = np.indices(data.shape, dtype=float)

    for _ in range(CENTROID_MAX_ITERATIONS):
        weights = data * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2)
        )
        total = weights.sum()

        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("The windowed flux of a star is not positive.")

        new_x = float((weights * xx).sum() / total)
        new_y = float((weights * yy).sum() / total)
        converged = np.hypot(new_x - x, new_y - y) < CENTROID_TOLERANCE
        x, y = new_x, new_y

        if converged:
            return x, y

    raise RuntimeError("The windowed centroid of a star did not converge.")


def _disk(radius: int) -> NDArray[np.bool_]:
    """Return a boolean disk of ``radius`` pixels."""
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return xx**2 + yy**2 <= radius**2


def other_sources(
    labels: NDArray[np.integer],
    x: float,
    y: float,
    reach: float,
    growth: int,
) -> tuple[slice, slice, NDArray[np.bool_]]:
    """Return a region about ``(x, y)`` and the mask of other sources in it.

    The detected segment under the pixel nearest ``(x, y)`` is the star; every
    other segment is grown by ``growth`` pixels.
    """
    rows, columns = labels.shape
    top = max(0, int(np.floor(y - reach)) - growth)
    bottom = min(rows, int(np.ceil(y + reach)) + growth + 1)
    left = max(0, int(np.floor(x - reach)) - growth)
    right = min(columns, int(np.ceil(x + reach)) + growth + 1)
    region = labels[top:bottom, left:right]
    own = labels[int(round(y)), int(round(x))]
    others = (region > 0) & (region != own)

    if growth > 0:
        others = binary_dilation(others, structure=_disk(growth))

    return slice(top, bottom), slice(left, right), others


def far_sky(
    image: FloatArray,
    labels: NDArray[np.integer],
    x: float,
    y: float,
    settings: BuilderSettings,
) -> tuple[float, float]:
    """Return the sigma-clipped sky and noise in the far ring of a star."""
    rows, columns, masked = other_sources(
        labels,
        x,
        y,
        settings.sky_outer_radius,
        settings.mask_growth_pixels,
    )
    yy, xx = np.mgrid[rows, columns]
    radius = np.hypot(xx - x, yy - y)
    ring = (
        (radius >= settings.sky_inner_radius)
        & (radius < settings.sky_outer_radius)
        & ~masked
    )

    if ring.sum() < MIN_RING_PIXELS:
        raise ValueError(
            f"The sky ring of the star at ({x:.1f}, {y:.1f}) has no free "
            "pixels."
        )

    _, median, deviation = sigma_clipped_stats(
        np.asarray(image[rows, columns], dtype=float)[ring],
        sigma=SIGMA_CLIP,
    )

    return float(median), float(deviation)


def star_window(
    image: FloatArray,
    candidate: dict,
    half: int,
    sky: tuple[float, float],
) -> dict | None:
    """Return one star on the common footprint, or ``None`` when unusable.

    The sky-subtracted window of ``2 * half + 1`` pixels is centred on the
    pixel nearest the windowed centroid, found in the star's own cutout of
    ``2 * half_size + 1`` pixels. ``None`` means the window leaves that
    cutout or holds no positive flux.
    """
    centre = int(candidate["half_size"])
    x_pixel = int(round(candidate["x"]))
    y_pixel = int(round(candidate["y"]))

    if (
        x_pixel - centre < 0
        or y_pixel - centre < 0
        or x_pixel + centre >= image.shape[1]
        or y_pixel + centre >= image.shape[0]
    ):
        return None

    stamp = np.asarray(
        image[
            y_pixel - centre:y_pixel + centre + 1,
            x_pixel - centre:x_pixel + centre + 1,
        ],
        dtype=float,
    ) - sky[0]
    x, y = window_centroid(
        stamp,
        centre + candidate["x"] - x_pixel,
        centre + candidate["y"] - y_pixel,
    )
    window_x = int(round(x))
    window_y = int(round(y))

    if (
        min(window_x, window_y) - half < 0
        or max(window_x, window_y) + half >= stamp.shape[0]
    ):
        return None

    window = stamp[
        window_y - half:window_y + half + 1,
        window_x - half:window_x + half + 1,
    ]
    flux = float(window.sum())

    if flux <= 0.0:
        return None

    return {
        "window": window,
        "flux": flux,
        "noise": float(sky[1]),
        "sky": float(sky[0]),
        "phase": (window_x - x, window_y - y),
        "x": x_pixel - centre + x,
        "y": y_pixel - centre + y,
    }


def phase_coefficients(
    phases: FloatArray,
    weights: FloatArray,
) -> FloatArray:
    """Return the star weights of the six coefficients of a local quadratic.

    Row ``k`` combines the stars into the coefficient of
    ``(1, dx, dy, dx**2, dx * dy, dy**2)[k]`` of the weighted least-squares
    quadratic in the sub-pixel phase.
    """
    dx = phases[:, 0]
    dy = phases[:, 1]
    design = np.column_stack(
        [np.ones_like(dx), dx, dy, dx**2, dx * dy, dy**2]
    )
    normal = design.T @ (weights[:, None] * design)

    return np.linalg.solve(normal, design.T * weights)


def ring_profile(
    image: FloatArray,
    labels: NDArray[np.integer],
    star: dict,
    settings: BuilderSettings,
) -> dict[str, FloatArray]:
    """Return one star's normalized azimuthal profile.

    Each ring of ``settings.ring_edges`` holds the sigma-clipped mean of the
    sky-subtracted pixels divided by the star's flux in the common
    footprint, the mean radius of its pixels and the standard error of the
    mean; a ring with fewer than ``MIN_RING_PIXELS`` free pixels is NaN.
    """
    edges = settings.ring_edges
    x = float(star["x"])
    y = float(star["y"])
    rows, columns, masked = other_sources(
        labels,
        x,
        y,
        edges[-1],
        settings.mask_growth_pixels,
    )
    yy, xx = np.mgrid[rows, columns]
    radius = np.hypot(xx - x, yy - y)
    values = (
        np.asarray(image[rows, columns], dtype=float) - star["sky"]
    ) / star["flux"]
    noise = star["noise"] / star["flux"]
    count = len(edges) - 1
    result = {
        "radius": np.full(count, np.nan),
        "value": np.full(count, np.nan),
        "error": np.full(count, np.nan),
    }

    for index in range(count):
        ring = (
            (radius >= edges[index])
            & (radius < edges[index + 1])
            & ~masked
        )

        if ring.sum() < MIN_RING_PIXELS:
            continue

        mean, _, _ = sigma_clipped_stats(values[ring], sigma=SIGMA_CLIP)
        result["radius"][index] = radius[ring].mean()
        result["value"][index] = mean
        result["error"][index] = noise / np.sqrt(ring.sum())

    return result


def combine_profiles(profiles: list[dict]) -> dict[str, FloatArray]:
    """Return the inverse-variance mean of star profiles.

    Rings that no star measures, the innermost ones, are left out.
    """
    radius = np.array([profile["radius"] for profile in profiles])
    value = np.array([profile["value"] for profile in profiles])
    measured = np.isfinite(value)
    weight = np.where(
        measured,
        1.0 / np.array([profile["error"] for profile in profiles]) ** 2,
        0.0,
    )
    total = weight.sum(axis=0)
    rings = total > 0.0
    radius = np.where(measured, radius, 0.0)[:, rings]
    value = np.where(measured, value, 0.0)[:, rings]
    weight = weight[:, rings]
    total = total[rings]

    return {
        "radius": (weight * radius).sum(axis=0) / total,
        "value": (weight * value).sum(axis=0) / total,
        "error": 1.0 / np.sqrt(total),
    }


def subpixel_kernels(
    coefficients: FloatArray,
    profile: dict[str, FloatArray],
    settings: BuilderSettings,
    *,
    factor: int = SUBPIXEL_FACTOR,
) -> FloatArray:
    """Return the kernels at every sub-pixel phase.

    At each phase the core is the phase quadratic evaluated there,
    normalized, on the ``2 * support_half_size + 1`` px grid; it is blended
    into the wing profile, interpolated at the radius from the displaced
    centre, between ``WING_START`` and ``WING_START + WING_BLEND`` px;
    negative pixels are set to zero and the kernel is normalized. The wing
    profile is set to zero where negative and beyond its last ring.
    """
    core_size = coefficients.shape[1]
    core_half = core_size // 2
    half = int(settings.support_half_size)
    size = 2 * half + 1

    if core_half > half:
        raise ValueError(
            "The common footprint of the stars exceeds the kernel support."
        )

    start = half - core_half
    yy, xx = np.indices((size, size), dtype=float)
    centre = (factor - 1) // 2
    ring_radius = profile["radius"]
    ring_value = np.clip(profile["value"], 0.0, None)
    end = settings.ring_edges[-1]
    kernels = np.empty((factor, factor, size, size))

    for j in range(factor):
        dy = (j - centre) / factor

        for i in range(factor):
            dx = (i - centre) / factor
            quadratic = (
                coefficients[0]
                + dx * coefficients[1]
                + dy * coefficients[2]
                + dx**2 * coefficients[3]
                + dx * dy * coefficients[4]
                + dy**2 * coefficients[5]
            )
            total = quadratic.sum()

            if not np.isfinite(total) or total <= 0.0:
                raise ValueError(
                    "The phase regression has no positive flux at phase "
                    f"({dx:+.3f}, {dy:+.3f})."
                )

            core = np.zeros((size, size))
            core[start:start + core_size, start:start + core_size] = (
                quadratic / total
            )
            distance = np.hypot(xx - half + dx, yy - half + dy)
            wings = np.where(
                distance < end,
                np.interp(distance, ring_radius, ring_value),
                0.0,
            )
            blend = np.clip((distance - WING_START) / WING_BLEND, 0.0, 1.0)
            kernel = np.clip((1.0 - blend) * core + blend * wings, 0.0, None)
            kernels[j, i] = kernel / kernel.sum()

    return kernels


def build_empirical_psf(
    image: FloatArray,
    candidates: list[dict],
    labels: NDArray[np.integer],
    settings: BuilderSettings,
) -> EmpiricalPSFBuild:
    """Build the effective PSF from the stars selected for it.

    Parameters
    ----------
    image : ndarray
        The science image, sky included.
    candidates : list of dict
        Stars with ``x``, ``y`` and ``half_size``, the half-size of their
        cutout; the smallest half-size sets the common footprint.
    labels : ndarray of int
        Segmentation of the detected sources of the image, zero where
        nothing is detected.
    settings : BuilderSettings
        Radii of the builder.

    Raises
    ------
    ValueError
        If fewer than six stars are usable, too few to fit the quadratic.
    """
    data = np.asarray(image, dtype=float)

    if not candidates:
        raise ValueError("No stars are available for the empirical PSF.")

    half = min(int(candidate["half_size"]) for candidate in candidates)
    stars = []
    windows = []
    profiles = []

    for candidate in candidates:
        sky = far_sky(
            data,
            labels,
            float(candidate["x"]),
            float(candidate["y"]),
            settings,
        )
        star = star_window(data, candidate, half, sky)

        if star is None:
            continue

        windows.append(star["window"] / star["flux"])
        profiles.append(ring_profile(data, labels, star, settings))
        stars.append(
            {
                "label": candidate.get("label"),
                "x": float(candidate["x"]),
                "y": float(candidate["y"]),
                "centroid_x": float(star["x"]),
                "centroid_y": float(star["y"]),
                "sky": star["sky"],
                "noise": star["noise"],
                "flux": star["flux"],
                "phase": [float(value) for value in star["phase"]],
            }
        )

    if len(stars) < 6:
        raise ValueError(
            f"Only {len(stars)} stars are usable for the empirical PSF; the "
            "phase regression needs at least six."
        )

    phases = np.array([star["phase"] for star in stars])
    weights = np.array(
        [(star["flux"] / star["noise"]) ** 2 for star in stars]
    )
    coefficients = np.tensordot(
        phase_coefficients(phases, weights),
        np.asarray(windows),
        axes=1,
    )
    profile = combine_profiles(profiles)
    kernels = subpixel_kernels(coefficients, profile, settings)
    centre = (SUBPIXEL_FACTOR - 1) // 2

    return EmpiricalPSFBuild(
        native=kernels[centre, centre],
        subpixel_kernels=kernels,
        wing_profile=profile,
        stars=stars,
        windows=windows,
    )
