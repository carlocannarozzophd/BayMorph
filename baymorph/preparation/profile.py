"""Where the sky begins: the light of the galaxy against a far sky (D1).

The sky begins where the light of the galaxy falls below a declared fraction
of the sky scale ``s``. This module measures that light and that radius:

- the annuli are ellipses about the target position, with the axis ratio and
  the position angle of the target segment;
- an annulus counts only when it is complete: at least
  :data:`COMPLETE_FRACTION` of its pixels on an unbounded grid lie in the
  image and hold data, so the cut corners of an image and the areas without
  data of a mosaic never enter the profile;
- the reference sky is a plane fitted with sigma clipping to the pixels far
  from the galaxy, beyond a radius, with every source masked; its level
  carries an error from the scatter among angular sectors, which holds the
  large-scale field of the sky, and its inner and outer halves must agree;
- the profile is the clipped mean of each annulus less the reference plane,
  in units of the scale of the reference sky;
- the radius at which the profile falls below a fraction ``f`` is where the
  non-increasing least-squares fit of the profile, weighted by its errors,
  crosses ``f``.

Under a large-scale field of the sky, galaxy light left on the reference sky
at the level of the field is invisible to every check on the reference sky
itself; the products that use it record this limitation.

Every position is in pixels of the array the functions receive, ``x`` along
the columns; the centre of the pixel ``[j, i]`` is ``(i, j)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import distance_transform_edt

from baymorph.stats.robust import truncated_gaussian_sigma

COMPLETE_FRACTION = 0.98
"""Smallest fraction of the pixels of an annulus on an unbounded grid that
must lie in the image for the annulus to count."""

LINEAR_BINS_UNTIL = 50
"""Radius, in pixels, up to which the annuli are 1 pixel wide."""

BIN_GROWTH = 1.02
"""Ratio of the outer to the inner radius of an annulus beyond
:data:`LINEAR_BINS_UNTIL`."""

CLIP_SIGMA = 3.0
"""Clipping, in standard deviations, of the means of the profile and of the
fit of the reference plane."""

CLIP_ITERATIONS = 10
"""Largest number of clipping passes."""

SECTORS = 8
"""Number of angular sectors that give the errors of the profile and of the
reference sky."""

MINIMUM_SECTOR_PIXELS = 3
"""Smallest number of pixels of a sector that enters an error."""

MINIMUM_REFERENCE_PIXELS = 10_000
"""Smallest number of pixels of a reference sky."""

AGREEMENT_SIGMA = 3.0
"""Largest difference, in combined errors, between the inner and the outer
halves of the reference sky."""


# --- The ellipse of the annuli ------------------------------------------------


@dataclass(frozen=True)
class Ellipse:
    """The centre, axis ratio and position angle of the annuli.

    Parameters
    ----------
    x0, y0 : float
        The centre, in pixels of the array.
    q : float
        The axis ratio, ``0 < q <= 1``.
    theta : float
        The position angle of the major axis, in radians, counter-clockwise
        from the ``x`` axis.
    """

    x0: float
    y0: float
    q: float
    theta: float

    def __post_init__(self) -> None:
        """Refuse an ellipse that is not valid."""
        values = (self.x0, self.y0, self.q, self.theta)

        if not all(math.isfinite(value) for value in values):
            raise ValueError("The centre and the shape of an ellipse are finite.")

        if not 0.0 < self.q <= 1.0:
            raise ValueError(f"The axis ratio must lie in (0, 1]; got {self.q!r}.")

    def coordinates(
        self, shape: tuple[int, int]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the radius and the angle of every pixel of an array.

        The radius is the semi-major axis of the ellipse through the pixel;
        the angle, in ``(-pi, pi]``, is measured in the frame in which the
        ellipse is a circle.
        """
        rows, columns = np.indices(shape, dtype=float)
        dx = columns - self.x0
        dy = rows - self.y0
        cosine, sine = math.cos(self.theta), math.sin(self.theta)
        along = dx * cosine + dy * sine
        across = (-dx * sine + dy * cosine) / self.q

        return np.hypot(along, across), np.arctan2(across, along)

    def half_extent(self, radius: float) -> tuple[float, float]:
        """Return the half-widths along ``x`` and ``y`` of the ellipse of a
        semi-major axis."""
        cosine, sine = math.cos(self.theta), math.sin(self.theta)

        return (
            radius * math.sqrt(cosine**2 + (self.q * sine) ** 2),
            radius * math.sqrt(sine**2 + (self.q * cosine) ** 2),
        )

    def inscribed_radius(self, shape: tuple[int, int]) -> float:
        """Return the largest semi-major axis whose ellipse lies in an array."""
        height, width = shape
        unit_x, unit_y = self.half_extent(1.0)

        return max(
            min(
                (self.x0 + 0.5) / unit_x,
                (width - 0.5 - self.x0) / unit_x,
                (self.y0 + 0.5) / unit_y,
                (height - 0.5 - self.y0) / unit_y,
            ),
            0.0,
        )


def ellipse_of_segment(
    image: ArrayLike, segment: ArrayLike, *, x0: float, y0: float
) -> Ellipse:
    """Return the ellipse of a segment about a given centre.

    The axis ratio and the position angle come from the second moments of the
    segment about ``(x0, y0)``, weighted by the light above the median of the
    segment, as in 0.6; without such light, from the moments of the segment
    alone.

    Parameters
    ----------
    image : array-like
        The image.
    segment : array-like of bool
        The pixels of the segment.
    x0, y0 : float
        The centre, the target position in pixels of the array (AD-055).

    Returns
    -------
    Ellipse
        The ellipse about ``(x0, y0)``.

    Raises
    ------
    ValueError
        If the segment holds fewer than three finite pixels.
    """
    data = np.asarray(image, dtype=float)
    pixels = np.asarray(segment, dtype=bool) & np.isfinite(data)
    rows, columns = np.nonzero(pixels)

    if rows.size < 3:
        raise ValueError("The ellipse of a segment needs at least three pixels.")

    values = data[rows, columns]
    weights = np.clip(values - np.median(values), 0.0, None)

    if not np.sum(weights) > 0.0:
        weights = np.ones_like(values)

    dx = columns - float(x0)
    dy = rows - float(y0)
    total = float(np.sum(weights))
    xx = float(np.sum(weights * dx * dx)) / total
    yy = float(np.sum(weights * dy * dy)) / total
    xy = float(np.sum(weights * dx * dy)) / total
    eigenvalues, eigenvectors = np.linalg.eigh(np.array([[xx, xy], [xy, yy]]))
    major, minor = float(eigenvalues[1]), float(eigenvalues[0])

    if major <= 0.0:
        return Ellipse(x0=float(x0), y0=float(y0), q=1.0, theta=0.0)

    q = float(np.clip(math.sqrt(max(minor, 0.0) / major), 1.0e-3, 1.0))
    axis = eigenvectors[:, 1]

    return Ellipse(
        x0=float(x0), y0=float(y0), q=q, theta=float(math.atan2(axis[1], axis[0]))
    )


# --- The annuli and the profile ----------------------------------------------


def annulus_edges(maximum: float) -> NDArray[np.float64]:
    """Return the edges of the annuli out to at least ``maximum`` pixels.

    The annuli are 1 pixel wide out to :data:`LINEAR_BINS_UNTIL` pixels and
    then grow by :data:`BIN_GROWTH` in radius.
    """
    edges = [float(value) for value in range(LINEAR_BINS_UNTIL + 1)]

    while edges[-1] < maximum:
        edges.append(edges[-1] * BIN_GROWTH)

    return np.asarray(edges)


@dataclass(frozen=True, eq=False)
class Profile:
    """The light of each annulus, less the reference plane.

    Parameters
    ----------
    inner, outer : ndarray
        The edges of the annuli, in pixels.
    value : ndarray
        The clipped mean of each annulus less the reference plane, in the
        units of the image; NaN where no pixel is used.
    error : ndarray
        Its error, from the scatter of the means of the angular sectors; NaN
        where fewer than four sectors hold pixels.
    used : ndarray of int
        The pixels each mean uses, after the masks and the clipping.
    in_image, unbounded : ndarray of int
        The pixels of each annulus in the image that hold data, and on an
        unbounded grid.
    complete : ndarray of bool
        The annuli that count: complete, and inside every incomplete one.
    """

    inner: NDArray[np.float64] = field(repr=False)
    outer: NDArray[np.float64] = field(repr=False)
    value: NDArray[np.float64] = field(repr=False)
    error: NDArray[np.float64] = field(repr=False)
    used: NDArray[np.int64] = field(repr=False)
    in_image: NDArray[np.int64] = field(repr=False)
    unbounded: NDArray[np.int64] = field(repr=False)
    complete: NDArray[np.bool_] = field(repr=False)

    @property
    def complete_radius(self) -> float:
        """float : The outer edge of the last annulus that counts."""
        indices = np.flatnonzero(self.complete)

        return 0.0 if indices.size == 0 else float(self.outer[indices[-1]])


def elliptical_profile(
    image: ArrayLike,
    masked: ArrayLike,
    ellipse: Ellipse,
    *,
    plane: tuple[float, float, float] = (0.0, 0.0, 0.0),
    no_data: ArrayLike | None = None,
) -> Profile:
    """Return the profile of an image on complete elliptical annuli.

    Parameters
    ----------
    image : array-like
        The image.
    masked : array-like of bool
        Pixels left out of the means, ``True`` = excluded; non-finite pixels
        must be among them, or among ``no_data``.
    ellipse : Ellipse
        The annuli.
    plane : tuple of float, default=(0, 0, 0)
        The reference plane ``(level, gradient_x, gradient_y)`` about the
        centre of the ellipse, subtracted from every pixel.
    no_data : array-like of bool, optional
        Pixels without data. They are left out of the means, and they count
        as outside the image when the completeness of an annulus is
        measured, so an annulus that crosses an area without data stops the
        profile as an edge of the image does.

    Returns
    -------
    Profile
        The profile out to the last annulus that holds pixels of the image.
    """
    data = np.asarray(image, dtype=float)
    out = np.asarray(masked, dtype=bool)

    if out.shape != data.shape:
        raise ValueError("The mask of a profile has the shape of the image.")

    missing = _no_data(no_data, data.shape)
    out = out | missing

    if np.any(~out & ~np.isfinite(data)):
        raise ValueError("A profile reads only finite pixels.")

    radius, angle = ellipse.coordinates(data.shape)
    edges, index, in_image, unbounded, complete = _completeness(
        ellipse, radius, missing
    )
    count = edges.size - 1
    inside = index < count
    level, gradient_x, gradient_y = plane
    rows, columns = np.nonzero(inside & ~out & complete[np.minimum(index, count - 1)])
    values = (
        data[rows, columns]
        - level
        - gradient_x * (columns - ellipse.x0)
        - gradient_y * (rows - ellipse.y0)
    )
    bins = index[rows, columns]
    sectors = _sector(angle[rows, columns])
    value = np.full(count, np.nan)
    error = np.full(count, np.nan)
    used = np.zeros(count, dtype=np.int64)
    order = np.argsort(bins, kind="stable")
    starts = np.searchsorted(bins[order], np.arange(count + 1))

    for number in range(count):
        members = order[starts[number] : starts[number + 1]]

        if members.size == 0:
            continue

        keep = _clipped(values[members])
        kept = values[members][keep]
        value[number] = float(np.mean(kept))
        used[number] = int(kept.size)
        error[number] = _sector_error(kept, sectors[members][keep])

    return Profile(
        inner=edges[:-1].copy(),
        outer=edges[1:].copy(),
        value=value,
        error=error,
        used=used,
        in_image=in_image.astype(np.int64),
        unbounded=unbounded.astype(np.int64),
        complete=complete,
    )


def complete_radius(
    shape: tuple[int, int], ellipse: Ellipse, *, no_data: ArrayLike | None = None
) -> float:
    """Return the outer edge of the last complete annulus of an array.

    Parameters
    ----------
    shape : tuple of int
        The shape of the array, ``(ny, nx)``.
    ellipse : Ellipse
        The annuli.
    no_data : array-like of bool, optional
        Pixels without data, which count as outside the array.

    Returns
    -------
    float
        The radius within which every annulus is complete; 0 when none is.
    """
    radius, _ = ellipse.coordinates(shape)
    edges, _, _, _, complete = _completeness(
        ellipse, radius, _no_data(no_data, tuple(shape))
    )
    counted = np.flatnonzero(complete)

    return 0.0 if counted.size == 0 else float(edges[counted[-1] + 1])


def _no_data(value: ArrayLike | None, shape: tuple[int, int]) -> NDArray[np.bool_]:
    """Return the validated pixels without data, none by default."""
    if value is None:
        return np.zeros(shape, dtype=bool)

    missing = np.asarray(value, dtype=bool)

    if missing.shape != tuple(shape):
        raise ValueError(
            f"The pixels without data have the shape {missing.shape}, the image "
            f"{tuple(shape)}."
        )

    return missing


def _completeness(
    ellipse: Ellipse, radius: NDArray[np.float64], no_data: NDArray[np.bool_]
) -> tuple:
    """Return the annuli of an array and which of them are complete.

    Returns the edges, the annulus of every pixel, the pixels of each annulus
    in the array that hold data and on an unbounded grid, and the annuli that
    count: complete, and inside every incomplete one.
    """
    reach = min(1.1 * ellipse.inscribed_radius(radius.shape) + 2.0, float(radius.max()))
    edges = annulus_edges(max(reach, 1.0))
    count = edges.size - 1
    index = np.searchsorted(edges, radius, side="right") - 1
    held = index[(index < count) & ~no_data]
    in_image = np.bincount(held, minlength=count)[:count]
    unbounded = _unbounded_counts(ellipse, edges)
    complete = (unbounded > 0) & (in_image >= COMPLETE_FRACTION * unbounded)
    first_incomplete = np.flatnonzero(~complete)

    if first_incomplete.size:
        complete[first_incomplete[0] :] = False

    return edges, index, in_image, unbounded, complete


def _unbounded_counts(ellipse: Ellipse, edges: NDArray[np.float64]) -> NDArray:
    """Return the pixels of each annulus on an unbounded grid.

    The grid has its pixel centres on the integers, as the image, so the
    counts follow the same sub-pixel position of the centre.
    """
    half_x, half_y = ellipse.half_extent(float(edges[-1]))
    first_column = math.floor(ellipse.x0 - half_x) - 1
    first_row = math.floor(ellipse.y0 - half_y) - 1
    width = math.ceil(ellipse.x0 + half_x) + 2 - first_column
    height = math.ceil(ellipse.y0 + half_y) + 2 - first_row
    shifted = Ellipse(
        x0=ellipse.x0 - first_column,
        y0=ellipse.y0 - first_row,
        q=ellipse.q,
        theta=ellipse.theta,
    )
    radius, _ = shifted.coordinates((height, width))
    count = edges.size - 1
    index = np.searchsorted(edges, radius, side="right") - 1

    return np.bincount(index[index < count], minlength=count)[:count]


def _clipped(values: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return the values kept by iterative clipping about the median."""
    keep = np.ones(values.size, dtype=bool)

    for _ in range(CLIP_ITERATIONS):
        kept = values[keep]

        if kept.size < 3:
            break

        scale = float(np.std(kept))

        if scale == 0.0:
            break

        updated = np.abs(values - float(np.median(kept))) < CLIP_SIGMA * scale

        if np.array_equal(updated, keep) or np.count_nonzero(updated) < 3:
            break

        keep = updated

    return keep


def _sector(angle: NDArray[np.float64]) -> NDArray[np.int64]:
    """Return the angular sector of each angle."""
    return np.minimum(
        ((angle + math.pi) / (2.0 * math.pi) * SECTORS).astype(np.int64), SECTORS - 1
    )


def _sector_error(values: NDArray[np.float64], sectors: NDArray[np.int64]) -> float:
    """Return the error of a mean from the scatter of its sector means."""
    counts = np.bincount(sectors, minlength=SECTORS)
    sums = np.bincount(sectors, weights=values, minlength=SECTORS)
    held = counts >= MINIMUM_SECTOR_PIXELS

    if np.count_nonzero(held) < 4:
        return math.nan

    means = sums[held] / counts[held]

    return float(np.std(means, ddof=1) / math.sqrt(means.size))


# --- The reference sky --------------------------------------------------------


@dataclass(frozen=True)
class ReferenceSky:
    """The sky far from the galaxy, against which its light is measured.

    Parameters
    ----------
    radius : float
        The radius beyond which the pixels belong to the reference sky.
    plane : tuple of float
        The plane ``(level, gradient_x, gradient_y)`` about the centre of the
        ellipse, in the units of the image.
    error : float
        The error of the level, from the scatter of the plane-subtracted
        means of the angular sectors.
    scale : float
        The scale of the sky about the plane, corrected for the clipping.
    pixels : int
        The pixels of the fit, after the masks.
    inner_minus_outer : float
        The plane-subtracted mean of the inner half of the pixels, split by
        radius, less that of the outer half.
    inner_outer_error : float
        Its error, from the sectors.
    """

    radius: float
    plane: tuple[float, float, float]
    error: float
    scale: float
    pixels: int
    inner_minus_outer: float
    inner_outer_error: float

    @property
    def consistent(self) -> bool:
        """bool : Whether the two halves agree within :data:`AGREEMENT_SIGMA`."""
        return bool(
            math.isfinite(self.inner_outer_error)
            and abs(self.inner_minus_outer) <= AGREEMENT_SIGMA * self.inner_outer_error
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the reference sky for a record."""
        return {
            "radius": self.radius,
            "plane": list(self.plane),
            "error": self.error,
            "scale": self.scale,
            "pixels": self.pixels,
            "inner_minus_outer": self.inner_minus_outer,
            "inner_outer_error": self.inner_outer_error,
            "consistent": self.consistent,
        }


def reference_sky(
    image: ArrayLike, masked: ArrayLike, ellipse: Ellipse, *, radius: float
) -> ReferenceSky:
    """Fit the reference sky beyond a radius.

    Parameters
    ----------
    image : array-like
        The image.
    masked : array-like of bool
        Pixels left out, ``True`` = excluded: every source with its growth,
        the target's included, and the pixels without data.
    ellipse : Ellipse
        The ellipse whose radius selects the pixels.
    radius : float
        The pixels at this radius and beyond belong to the reference sky.

    Returns
    -------
    ReferenceSky
        The plane, its error and its checks.

    Raises
    ------
    ValueError
        If fewer than :data:`MINIMUM_REFERENCE_PIXELS` pixels are left.
    """
    data = np.asarray(image, dtype=float)
    out = np.asarray(masked, dtype=bool)
    radii, angle = ellipse.coordinates(data.shape)
    rows, columns = np.nonzero((radii >= float(radius)) & ~out)

    if rows.size < MINIMUM_REFERENCE_PIXELS:
        raise ValueError(
            f"The reference sky beyond a radius of {radius:.0f} pixels holds "
            f"{rows.size} unmasked pixels, fewer than {MINIMUM_REFERENCE_PIXELS}."
        )

    values = data[rows, columns]
    dx = columns - ellipse.x0
    dy = rows - ellipse.y0
    design = np.column_stack([np.ones(values.size), dx, dy])
    keep = np.ones(values.size, dtype=bool)
    threshold = math.inf
    coefficients = np.zeros(3)

    for _ in range(CLIP_ITERATIONS):
        coefficients, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
        residual = values - design @ coefficients
        scale = float(np.std(residual[keep]))

        if scale == 0.0:
            break

        updated = np.abs(residual) < CLIP_SIGMA * scale

        if np.array_equal(updated, keep) or np.count_nonzero(updated) < 3:
            break

        keep = updated
        threshold = CLIP_SIGMA * scale

    residual = values - design @ coefficients
    kept = residual[keep]
    sectors = _sector(angle[rows, columns])[keep]
    outer = radii[rows, columns][keep] >= float(np.median(radii[rows, columns][keep]))
    differences = []

    for sector in range(SECTORS):
        inner_part = kept[(sectors == sector) & ~outer]
        outer_part = kept[(sectors == sector) & outer]

        if (
            inner_part.size >= MINIMUM_SECTOR_PIXELS
            and outer_part.size >= MINIMUM_SECTOR_PIXELS
        ):
            differences.append(float(np.mean(inner_part) - np.mean(outer_part)))

    if len(differences) >= 4:
        difference = float(np.mean(differences))
        difference_error = float(
            np.std(differences, ddof=1) / math.sqrt(len(differences))
        )
    else:
        difference, difference_error = math.nan, math.nan

    return ReferenceSky(
        radius=float(radius),
        plane=(float(coefficients[0]), float(coefficients[1]), float(coefficients[2])),
        error=_sector_error(kept, sectors),
        scale=float(truncated_gaussian_sigma(kept, threshold, center=0.0)),
        pixels=int(kept.size),
        inner_minus_outer=difference,
        inner_outer_error=difference_error,
    )


# --- The radius at which the sky begins --------------------------------------


def monotone_fit(profile: Profile, excess: ArrayLike) -> NDArray[np.float64]:
    """Return the non-increasing fit of a profile on its complete annuli.

    The fit is the weighted least-squares non-increasing sequence through the
    complete annuli with a finite profile, with weights from their errors;
    an annulus without a finite error takes the median weight.

    Parameters
    ----------
    profile : Profile
        The profile, whose complete annuli are read.
    excess : array-like
        The profile in units of the sky scale, one value per annulus.

    Returns
    -------
    ndarray
        The fitted value of each annulus; NaN on the annuli that do not count.

    Raises
    ------
    ValueError
        If ``excess`` does not hold one value per annulus.
    """
    values = np.asarray(excess, dtype=float)

    if values.shape != profile.value.shape:
        raise ValueError(
            f"The excess holds {values.size} values for {profile.value.size} annuli."
        )

    fitted = np.full(values.shape, np.nan)
    indices = _usable(profile, values)

    if indices.size == 0:
        return fitted

    errors = profile.error[indices]
    finite = np.isfinite(errors) & (errors > 0.0)
    typical = float(np.median(errors[finite])) if np.any(finite) else 1.0
    weights = 1.0 / np.where(finite, errors, typical) ** 2
    fitted[indices] = nonincreasing_fit(values[indices], weights)

    return fitted


def monotone_radius(profile: Profile, excess: ArrayLike, level: float) -> float | None:
    """Return the radius at which the non-increasing fit crosses ``level``.

    Parameters
    ----------
    profile : Profile
        The profile, whose complete annuli are read.
    excess : array-like
        The profile in units of the sky scale.
    level : float
        The fraction of the sky scale.

    Returns
    -------
    float or None
        The inner edge of the first annulus whose fitted value
        (:func:`monotone_fit`) lies below ``level``, or None when the fit at
        the last complete annulus does not.
    """
    below = np.flatnonzero(monotone_fit(profile, excess) < float(level))

    if below.size == 0:
        return None

    return float(profile.inner[below[0]])


def nonincreasing_fit(values: ArrayLike, weights: ArrayLike) -> NDArray[np.float64]:
    """Return the weighted least-squares non-increasing fit of a sequence.

    The pool-adjacent-violators algorithm merges neighbouring blocks that
    increase until none does.
    """
    means: list[float] = []
    totals: list[float] = []
    sizes: list[int] = []

    for value, weight in zip(np.asarray(values, float), np.asarray(weights, float)):
        means.append(float(value))
        totals.append(float(weight))
        sizes.append(1)

        while len(means) > 1 and means[-2] < means[-1]:
            weight_sum = totals[-2] + totals[-1]
            means[-2] = (means[-2] * totals[-2] + means[-1] * totals[-1]) / weight_sum
            totals[-2] = weight_sum
            sizes[-2] += sizes[-1]
            del means[-1], totals[-1], sizes[-1]

    return np.repeat(np.asarray(means), sizes)


def _usable(profile: Profile, excess: ArrayLike) -> NDArray[np.int64]:
    """Return the complete annuli with a finite profile."""
    values = np.asarray(excess, dtype=float)

    return np.flatnonzero(profile.complete & np.isfinite(values))


# --- Masks --------------------------------------------------------------------


def grown(mask: ArrayLike, pixels: float) -> NDArray[np.bool_]:
    """Return a mask grown by ``pixels`` in every direction."""
    base = np.asarray(mask, dtype=bool)

    if not np.any(base) or pixels <= 0:
        return base.copy()

    return distance_transform_edt(~base) <= float(pixels)


__all__ = [
    "AGREEMENT_SIGMA",
    "BIN_GROWTH",
    "CLIP_ITERATIONS",
    "CLIP_SIGMA",
    "COMPLETE_FRACTION",
    "LINEAR_BINS_UNTIL",
    "MINIMUM_REFERENCE_PIXELS",
    "MINIMUM_SECTOR_PIXELS",
    "SECTORS",
    "Ellipse",
    "Profile",
    "ReferenceSky",
    "annulus_edges",
    "complete_radius",
    "elliptical_profile",
    "ellipse_of_segment",
    "grown",
    "monotone_fit",
    "monotone_radius",
    "nonincreasing_fit",
    "reference_sky",
]
