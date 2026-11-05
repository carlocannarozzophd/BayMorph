"""The target stage of the preparation: the position of the galaxy.

The target position, ``data.target_position``, is the centre of the galaxy in
pixels of the whole image (AD-055). The priors of the centre stay about it,
and the cutout is taken about it. This stage checks it against the image,
before any detection:

- a missing position stops the preparation;
- ``"center"`` is the centre of the whole image;
- a position outside the image, outside the working region that the opening
  read, or on a pixel without data stops the preparation, with the remedy.

Whether a source lies at the position needs a detection. It is checked where a
detection first selects the target segment, the cutout stage, which detects
the sources of the working region whether the cutout is automatic or fixed,
through :func:`select_target_segment`:

- a position on a segment selects it;
- a position on no segment selects the nearest segment, with a warning, when
  the nearest of its pixels lies within ``max_distance`` pixels, and stops
  beyond, naming that segment and its distance;
- a segment that the deblending split from a brighter source of the same
  detection is selected with a warning that names the brighter source.

Positions are zero-based pixels of the whole image, ``x`` along the columns:
the centre of the first pixel is ``(0, 0)``, and an image of ``n`` pixels on
an axis covers ``-0.5`` to ``n - 0.5`` on it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from baymorph.preparation.identity import stage_identity
from baymorph.preparation.opening import OpenedObservation, seed_position
from baymorph.preparation.store import StageProduct

TARGET_STAGE = "target"
"""Name of the stage in the product store."""

TARGET_VERSION = 1
"""Version of the rules of the stage; raising it changes every identity."""

MAX_SEED_DISTANCE = 8.0
"""Largest distance, in pixels, from a target position on no source to the
nearest pixel of the source taken as the target: the half-width of the
default window of the centre priors, uniform within 8 pixels of the target
position."""


def _pixel_index(position: float, start: int, stop: int) -> int:
    """Return the pixel nearest a position along one axis, within a range.

    The pixel ``i`` covers the positions from ``i - 0.5`` to ``i + 0.5``. A
    position halfway between two pixels goes to the even one, as Python's
    ``round`` gives it and as the working region and the centre check of the
    fit take it; a position on the outer edge of the range goes to its last
    pixel.
    """
    return min(max(int(round(position)), start), stop - 1)


def _inside(position: float, start: int, stop: int) -> bool:
    """Return whether a position lies in the pixels ``start`` to ``stop - 1``."""
    return start - 0.5 <= position <= stop - 0.5


# --- The target position ------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Target:
    """The product of the target stage: the checked target position.

    Parameters
    ----------
    x, y : float
        The target position, in pixels of the whole image.
    column, row : int
        The pixel nearest the position, in pixels of the whole image.
    source : {"declared", "image_centre"}
        The position declared in ``data.target_position``, or the centre of
        the whole image, for ``"center"``.
    image_identity : str
        Identity of the pixels against which the position was checked, from
        the opening.
    identity : str
        Identity of the product.
    record : Mapping
        What the stage checked, JSON-serializable.
    """

    x: float
    y: float
    column: int
    row: int
    source: str
    image_identity: str
    identity: str
    record: Mapping[str, Any] = field(repr=False)

    @property
    def position(self) -> tuple[float, float]:
        """tuple of float : The target position ``(x, y)``."""
        return (self.x, self.y)

    def product(self) -> StageProduct:
        """Return the product of the stage, for the product store."""
        return StageProduct(
            stage=TARGET_STAGE,
            version=TARGET_VERSION,
            identity=self.identity,
            record=dict(self.record),
        )


def resolve_target(opened: OpenedObservation, target_position: Any) -> Target:
    """Check the target position against the image, as the target stage.

    Parameters
    ----------
    opened : OpenedObservation
        The product of the opening.
    target_position : None, "center" or mapping
        ``data.target_position`` as the schema returns it: ``"center"``, the
        centre of the whole image, or ``{x: ..., y: ...}`` in pixels of the
        whole image.

    Returns
    -------
    Target
        The checked position, with its identity and record.

    Raises
    ------
    TypeError
        If ``opened`` is not the product of the opening.
    ValueError
        If the position is missing or malformed, lies outside the image or
        outside the working region, or falls on a pixel without data.
    """
    if not isinstance(opened, OpenedObservation):
        raise TypeError(
            "resolve_target takes the OpenedObservation of the opening stage, not "
            f"{type(opened).__name__}."
        )

    height, width = opened.image_shape
    position = seed_position(target_position, opened.image_shape)

    if position is None:
        raise ValueError(
            "data.target_position is required: give the centre of the galaxy as "
            "{x: ..., y: ...} in pixels of the whole image, or 'center' when the "
            "galaxy lies at the centre of the image."
        )

    x, y = position
    declared: str | dict[str, float] = (
        "center" if target_position == "center" else {"x": x, "y": y}
    )

    if not (_inside(x, 0, width) and _inside(y, 0, height)):
        raise ValueError(
            f"data.target_position ({x:g}, {y:g}) lies outside the image of "
            f"{width} x {height} pixels, which covers x from -0.5 to "
            f"{width - 0.5:g} and y from -0.5 to {height - 0.5:g}. BayMorph counts "
            "pixels from zero, x along the columns, so the centre of the first "
            "pixel is (0, 0): a position read in DS9, which counts from one, is "
            "larger by one on each axis, and x and y may have been swapped."
        )

    region = opened.region
    xmin, xmax, ymin, ymax = (
        (0, width, 0, height)
        if region is None
        else (region.xmin, region.xmax, region.ymin, region.ymax)
    )

    if not (_inside(x, xmin, xmax) and _inside(y, ymin, ymax)):
        raise ValueError(
            f"data.target_position ({x:g}, {y:g}) lies outside the working "
            f"region x {xmin}-{xmax}, y {ymin}-{ymax} that the opening read from "
            f"the image of {width} x {height} pixels (data.region): open the "
            "image about this target position."
        )

    column = _pixel_index(x, xmin, xmax)
    row = _pixel_index(y, ymin, ymax)

    if opened.no_data[row - ymin, column - xmin]:
        raise ValueError(
            f"data.target_position ({x:g}, {y:g}) falls on the pixel ({column}, "
            f"{row}), which holds no data: its value is not finite, or it lies in "
            "an area of exact zeros, which mosaics leave where no exposure covers "
            "the sky (AD-053). The target position must be the centre of the "
            "galaxy: check it."
        )

    identity = stage_identity(
        stage=TARGET_STAGE,
        version=TARGET_VERSION,
        settings={"target_position": declared},
        upstream={"image": opened.image_identity},
    )
    record = {
        "stage": TARGET_STAGE,
        "version": TARGET_VERSION,
        "declared": declared,
        "source": "image_centre" if declared == "center" else "declared",
        "position": {"x": x, "y": y},
        "pixel": {"x": column, "y": row},
        "image": {"width": width, "height": height},
        "region": None if region is None else region.as_dict(),
        "checks": [
            "inside the image",
            "inside the working region",
            "on a pixel with data",
        ],
        "sources": "checked by the first detection, which selects the target segment",
        "image_identity": opened.image_identity,
    }

    return Target(
        x=x,
        y=y,
        column=column,
        row=row,
        source=record["source"],
        image_identity=opened.image_identity,
        identity=identity,
        record=record,
    )


# --- The target segment -------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """One segment of a detection, as a record and a message name it.

    Parameters
    ----------
    label : int
        Its label in the map of segments.
    pixels : int
        Its area, in pixels.
    centre : tuple of float
        The centre of its pixels, in pixels of the whole image.
    peak : tuple of int or None
        Its brightest pixel in the flux image, in pixels of the whole image,
        or None without a flux image. It marks the centre of a galaxy better
        than the centre of the pixels, which the deblending can pull away from
        the light.
    flux : float or None
        Its flux, the sum of the flux image over its pixels, or None without a
        flux image.
    """

    label: int
    pixels: int
    centre: tuple[float, float]
    peak: tuple[int, int] | None = None
    flux: float | None = None

    def describe(self) -> str:
        """Return how a message names the segment."""
        where = (
            f"centred at ({self.centre[0]:.1f}, {self.centre[1]:.1f})"
            if self.peak is None
            else f"brightest at ({self.peak[0]}, {self.peak[1]})"
        )

        return f"segment {self.label} ({self.pixels} pixels, {where})"

    def as_dict(self) -> dict[str, Any]:
        """Return the segment for a record."""
        return {
            "label": self.label,
            "pixels": self.pixels,
            "centre": {"x": self.centre[0], "y": self.centre[1]},
            "peak": (
                None if self.peak is None else {"x": self.peak[0], "y": self.peak[1]}
            ),
            "flux": self.flux,
        }


@dataclass(frozen=True)
class TargetSegment:
    """The segment of a detection that the target position selects.

    Parameters
    ----------
    segment : Segment
        The selected segment.
    seed : tuple of float
        The target position, in pixels of the whole image.
    on_source : bool
        Whether the pixel nearest the target position belongs to a segment.
    distance : float
        Distance, in pixels, from the target position to the centre of the
        nearest pixel of the segment; 0 when the position lies on it.
    nearest_pixel : tuple of int
        That pixel ``(x, y)``, in pixels of the whole image.
    max_distance : float
        The largest distance at which the nearest segment is taken.
    equally_near : tuple of int
        Other segments whose nearest pixel lies at the same distance.
    parent : int or None
        The label of the detection before the deblending that holds the
        segment, or None when the siblings were not checked.
    siblings : int or None
        How many segments the deblending made of that detection, the selected
        one included.
    brighter : Segment or None
        The brightest segment of that detection, when it is brighter than the
        selected one.
    brighter_count : int
        How many segments of that detection are brighter than the selected
        one.
    warnings : tuple of str
        What the stage that calls the selection records and shows.
    """

    segment: Segment
    seed: tuple[float, float]
    on_source: bool
    distance: float
    nearest_pixel: tuple[int, int]
    max_distance: float
    equally_near: tuple[int, ...] = ()
    parent: int | None = None
    siblings: int | None = None
    brighter: Segment | None = None
    brighter_count: int = 0
    warnings: tuple[str, ...] = ()

    @property
    def label(self) -> int:
        """int : The label of the selected segment."""
        return self.segment.label

    def as_dict(self) -> dict[str, Any]:
        """Return the selection for a record, JSON-serializable."""
        siblings = (
            None
            if self.parent is None
            else {
                "parent": self.parent,
                "segments": self.siblings,
                "brighter": None if self.brighter is None else self.brighter.as_dict(),
                "brighter_count": self.brighter_count,
            }
        )

        return {
            "segment": self.segment.as_dict(),
            "seed": {"x": self.seed[0], "y": self.seed[1]},
            "on_source": self.on_source,
            "distance": self.distance,
            "nearest_pixel": {"x": self.nearest_pixel[0], "y": self.nearest_pixel[1]},
            "max_distance": self.max_distance,
            "equally_near": list(self.equally_near),
            "siblings": siblings,
            "warnings": list(self.warnings),
        }


def select_target_segment(
    labels: ArrayLike,
    seed: tuple[float, float],
    *,
    parents: ArrayLike | None = None,
    flux: ArrayLike | None = None,
    origin: tuple[int, int] = (0, 0),
    max_distance: float = MAX_SEED_DISTANCE,
) -> TargetSegment:
    """Select the segment of a detection that holds the target.

    A target position on a segment selects it. A position on no segment
    selects the segment with the pixel nearest to it, measured to the centre
    of the pixel, with a warning, when that pixel lies within
    ``max_distance``; beyond, the selection stops and names the segment.
    Among pixels at the same distance the first by row, then by column, is
    taken, and the other segments at that distance are named.

    The nearest pixel is searched in square windows about the position that
    double in size from ``max_distance``: the search ends at the first window
    whose nearest labelled pixel lies within its half-side, which makes it
    exact, and it reads only the pixels of that window, so its cost grows
    with the distance to the nearest source, not with the number of sources.

    When the map before the deblending and the flux are given, a segment that
    the deblending split from a brighter source of the same detection is
    selected with a warning that names the brightest such source: a position
    on a compact companion, such as a star near the nucleus, would otherwise
    make the companion the target without a word.

    Parameters
    ----------
    labels : array-like of int
        The map of segments after the deblending, 0 where there is no source.
    seed : tuple of float
        The target position ``(x, y)``, in pixels of the whole image; it must
        lie on the map.
    parents : array-like of int, optional
        The map of segments before the deblending, of the same shape; given
        together with ``flux``.
    flux : array-like of float, optional
        The flux above the sky of each pixel, of the same shape and finite on
        every pixel of a segment, such as the detection image less its
        background level; the flux of a segment is its sum.
    origin : tuple of int, default=(0, 0)
        The pixel ``(x, y)`` of the whole image at the first element of the
        map, such as the lower bounds of a cutout or of the working region.
    max_distance : float, default=MAX_SEED_DISTANCE
        The largest distance, in pixels, at which the nearest segment is taken
        when the position lies on no segment.

    Returns
    -------
    TargetSegment
        The selected segment, how it was selected and the warnings.

    Raises
    ------
    ValueError
        If an input is not valid, the position lies outside the map, the map
        holds no segment, or the nearest segment lies beyond
        ``max_distance``.
    """
    labels = _label_map(labels, "labels")
    origin = _origin(origin)
    x, y = _finite_pair(seed, "The target position")
    limit = _max_distance(max_distance)
    height, width = labels.shape
    local_x, local_y = x - origin[0], y - origin[1]
    bounds = (
        f"x {origin[0]}-{origin[0] + width}, y {origin[1]}-{origin[1] + height} of "
        "the whole image"
    )

    if not (_inside(local_x, 0, width) and _inside(local_y, 0, height)):
        raise ValueError(
            f"The target position ({x:g}, {y:g}) lies outside the map of "
            f"detected sources, {bounds}."
        )

    if (parents is None) != (flux is None):
        raise ValueError(
            "The map before the deblending and the flux image are given "
            "together, for the check of a brighter source of the same detection."
        )

    if parents is not None:
        parents = _label_map(parents, "parents")
        flux = np.asarray(flux)

        if parents.shape != labels.shape or flux.shape != labels.shape:
            raise ValueError(
                f"The map before the deblending, of shape {parents.shape}, and the "
                f"flux image, of shape {flux.shape}, must have the shape of the map "
                f"of segments, {labels.shape}."
            )

        if flux.dtype == bool or not np.issubdtype(flux.dtype, np.number):
            raise ValueError(f"The flux image must be numeric; got dtype {flux.dtype}.")

    column = _pixel_index(local_x, 0, width)
    row = _pixel_index(local_y, 0, height)
    label = int(labels[row, column])
    equally_near: tuple[int, ...] = ()

    if label > 0:
        on_source, distance, near = True, 0.0, (row, column)
    else:
        found = _nearest_labelled_pixel(labels, local_x, local_y, max(limit, 1.0))

        if found is None:
            raise ValueError(
                f"No source was detected in the map {bounds}, so no segment can "
                "hold the target: check the image, or the detection settings, "
                "preparation.detection."
            )

        near, squared, tied = found
        label = int(labels[near])
        on_source, distance = False, math.sqrt(squared)
        equally_near = tuple(int(other) for other in tied if int(other) != label)

    segment = _segment(labels, label, flux, origin)
    nearest = (near[1] + origin[0], near[0] + origin[1])
    ties = (
        ""
        if not equally_near
        else (
            f" Segments {', '.join(str(other) for other in equally_near)} lie at "
            "the same distance; the one whose nearest pixel comes first by row, "
            "then by column, is chosen."
        )
    )

    if not on_source and distance > limit:
        raise ValueError(
            f"No source was detected at the target position ({x:g}, {y:g}) in "
            f"pixels of the whole image. The nearest is {segment.describe()}, "
            f"whose nearest pixel ({nearest[0]}, {nearest[1]}) lies "
            f"{distance:.2f} pixels away, beyond the {limit:g} pixels within which "
            f"the nearest source is taken as the target.{ties} The target "
            "position must be the centre of the galaxy (AD-055): check "
            "data.target_position, in zero-based pixels of the whole image with x "
            "along the columns, or, if the galaxy is fainter than the detection "
            "threshold, lower preparation.detection.n_sigma or n_pixels."
        )

    warnings: list[str] = []

    if not on_source:
        warnings.append(
            f"No source was detected at the target position ({x:g}, {y:g}). The "
            f"nearest is {segment.describe()}, whose nearest pixel ({nearest[0]}, "
            f"{nearest[1]}) lies {distance:.2f} pixels away, within {limit:g} "
            f"pixels, so it is taken as the target.{ties} Check that it is the "
            "galaxy: the target position should be its centre (AD-055)."
        )

    selection = TargetSegment(
        segment=segment,
        seed=(x, y),
        on_source=on_source,
        distance=distance,
        nearest_pixel=nearest,
        max_distance=limit,
        equally_near=equally_near,
    )

    if parents is None:
        return replace(selection, warnings=tuple(warnings))

    parent, siblings, brighter, count = _siblings(
        labels, parents, flux, segment=segment, origin=origin
    )

    if brighter is not None:
        warnings.append(_sibling_warning(segment, brighter, count))

    return replace(
        selection,
        parent=parent,
        siblings=siblings,
        brighter=brighter,
        brighter_count=count,
        warnings=tuple(warnings),
    )


def _nearest_labelled_pixel(
    labels: NDArray[np.integer], x: float, y: float, radius: float
) -> tuple[tuple[int, int], float, NDArray[np.integer]] | None:
    """Return the labelled pixel nearest a position, or None on an empty map.

    Returns the pixel as ``(row, column)`` of the map, its squared distance to
    the position, and the labels of every pixel at that distance.
    """
    height, width = labels.shape

    while True:
        # One pixel more on each side keeps every pixel within the radius in
        # the window, whatever the rounding of x - radius.
        first_column = max(math.ceil(x - radius) - 1, 0)
        last_column = min(math.floor(x + radius) + 1, width - 1)
        first_row = max(math.ceil(y - radius) - 1, 0)
        last_row = min(math.floor(y + radius) + 1, height - 1)
        whole = (
            first_column == 0
            and first_row == 0
            and last_column == width - 1
            and last_row == height - 1
        )
        window = labels[first_row : last_row + 1, first_column : last_column + 1]
        rows, columns = np.nonzero(window)

        if rows.size:
            dx = columns + (first_column - x)
            dy = rows + (first_row - y)
            squared = dx * dx + dy * dy
            nearest = int(np.argmin(squared))
            least = float(squared[nearest])

            # Every pixel outside the window lies farther than the radius, so
            # a nearest pixel within it is the nearest of the whole map.
            if whole or least <= radius * radius:
                tied = squared == least

                return (
                    (
                        int(rows[nearest]) + first_row,
                        int(columns[nearest]) + first_column,
                    ),
                    least,
                    np.unique(window[rows[tied], columns[tied]]),
                )

        if whole:
            return None

        radius *= 2.0


def _segment(
    labels: NDArray[np.integer],
    label: int,
    flux: NDArray[Any] | None,
    origin: tuple[int, int],
) -> Segment:
    """Return a segment with its area, the centre of its pixels and its light."""
    rows, columns = np.nonzero(labels == label)
    centre = (float(columns.mean()) + origin[0], float(rows.mean()) + origin[1])

    if flux is None:
        return Segment(label=label, pixels=int(rows.size), centre=centre)

    values = np.asarray(flux[rows, columns], dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "The flux image must be finite on every pixel of a segment; it is not "
            f"on segment {label}."
        )

    brightest = int(np.argmax(values))

    return Segment(
        label=label,
        pixels=int(rows.size),
        centre=centre,
        peak=(int(columns[brightest]) + origin[0], int(rows[brightest]) + origin[1]),
        flux=float(values.sum()),
    )


def _siblings(
    labels: NDArray[np.integer],
    parents: NDArray[np.integer],
    flux: NDArray[Any],
    *,
    segment: Segment,
    origin: tuple[int, int],
) -> tuple[int, int, Segment | None, int]:
    """Return the detection of a segment and its brightest brighter sibling.

    Returns the label of the detection before the deblending, the number of
    its segments, the brightest of them when it is brighter than ``segment``,
    and how many are brighter.
    """
    owners = np.unique(parents[labels == segment.label])

    if owners.size != 1 or owners[0] <= 0:
        raise ValueError(
            f"The map before the deblending must hold segment {segment.label} "
            f"inside one of its sources; its pixels lie in {owners.tolist()}."
        )

    parent = int(owners[0])
    inside = (parents == parent) & (labels > 0)
    members = labels[inside].astype(np.intp, copy=False)
    member_flux = np.asarray(flux[inside], dtype=float)

    if not np.all(np.isfinite(member_flux)):
        raise ValueError(
            "The flux image must be finite on every pixel of a segment; it is not "
            f"on the segments of the detection {parent}."
        )

    counts = np.bincount(members)
    sums = np.bincount(members, weights=member_flux, minlength=counts.size)
    present = np.flatnonzero(counts)
    brighter = present[
        (present != segment.label) & (sums[present] > sums[segment.label])
    ]

    if brighter.size == 0:
        return parent, int(present.size), None, 0

    brightest = int(brighter[np.argmax(sums[brighter])])

    return (
        parent,
        int(present.size),
        _segment(labels, brightest, flux, origin),
        int(brighter.size),
    )


def _sibling_warning(segment: Segment, brighter: Segment, count: int) -> str:
    """Return the warning that names a brighter source of the same detection."""
    own = float(segment.flux or 0.0)
    other = float(brighter.flux or 0.0)
    amount = (
        f"{other / own:.3g} times its flux"
        if own > 0.0
        else f"a flux of {other:.6g}, against {own:.6g} for the target segment"
    )
    more = (
        ""
        if count == 1
        else f"; {count} segments of that detection are brighter than the target"
    )

    return (
        f"The deblending split the target segment {segment.label} from a brighter "
        f"source of the same detection: {brighter.describe()}, with {amount}"
        f"{more}. If that source is the galaxy, the target position lies on a "
        "companion, such as a star or a compact knot: move it to the centre of "
        "the galaxy (AD-055)."
    )


# --- Checks of the inputs -----------------------------------------------------


def _label_map(value: ArrayLike, name: str) -> NDArray[np.integer]:
    """Return a validated two-dimensional map of non-negative integer labels."""
    array = np.asarray(value)

    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"'{name}' must be a non-empty two-dimensional map of labels.")

    if array.dtype == bool or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"'{name}' must hold integer labels, 0 where there is no source; got "
            f"dtype {array.dtype}."
        )

    if np.issubdtype(array.dtype, np.signedinteger) and array.min() < 0:
        raise ValueError(f"'{name}' must not hold negative labels.")

    return array


def _origin(value: Any) -> tuple[int, int]:
    """Return a validated origin of a map in pixels of the whole image."""
    if (
        isinstance(value, (str, bytes))
        or not hasattr(value, "__len__")
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, Integral) or item < 0
            for item in value
        )
    ):
        raise ValueError(
            "The origin of a map is two non-negative integers, the pixel (x, y) of "
            f"the whole image at its first element; got {value!r}."
        )

    return int(value[0]), int(value[1])


def _finite_pair(value: Any, what: str) -> tuple[float, float]:
    """Return two finite coordinates ``(x, y)``."""
    if (
        isinstance(value, (str, bytes))
        or not hasattr(value, "__len__")
        or len(value) != 2
        or any(
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, Real)
            or not math.isfinite(item)
            for item in value
        )
    ):
        raise ValueError(
            f"{what} must be two finite coordinates (x, y); got {value!r}."
        )

    return float(value[0]), float(value[1])


def _max_distance(value: Any) -> float:
    """Return a validated largest distance, in pixels."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(
            "max_distance must be a finite non-negative number of pixels; got "
            f"{value!r}."
        )

    return float(value)


__all__ = [
    "MAX_SEED_DISTANCE",
    "TARGET_STAGE",
    "TARGET_VERSION",
    "Segment",
    "Target",
    "TargetSegment",
    "resolve_target",
    "select_target_segment",
]
