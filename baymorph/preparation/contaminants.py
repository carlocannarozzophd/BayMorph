"""Which detected segments hold a source above the light of the target (AD-050).

The detection compares the smoothed image with one threshold above the sky.
Over the wings of a bright galaxy its light lifts peaks of the noise above
that threshold, and the deblending splits the galaxy itself into pieces; each
becomes a segment, and masking them all takes light of the galaxy and positive
noise out of the measurement. This module repeats the test of the detection
inside each segment with the threshold raised by the light of the target:

- the light level is the median of the smoothed image in elliptical annuli
  1 pixel wide about the target position, without the masked pixels; annuli
  without pixels take the value interpolated in radius from their neighbours.
  It is the sky plus the smooth light of the target, and far from the target
  the sky alone, so there the test is the detection itself;
- a segment holds a source when at least ``n_pixels`` of its pixels,
  connected along edges or corners as in the detection, exceed the level by
  the excess of the detection threshold; those pixels are its footprint. A
  segment that holds no source is released.

It ports :func:`baymorph.core.detection.segments_above_level` and the light
level of 0.6, with the annuli about the target position (AD-055), and adds
three options to the release:

- the light level of the first pass may leave out only the segments of the
  other detections, so that the pieces of the target's own detection enter
  its medians: when the target segment is a small piece of the galaxy, the
  annuli that its other pieces cover are otherwise left empty, and the level
  interpolated across them falls below the light of the galaxy;
- beyond the last complete annulus the level may be the plane of the far sky
  instead of the median of the short arcs of the annuli that the edges of the
  image cut, which follow the large-scale field of the sky in the corners;
- a second pass: the ellipse is measured again on the target segment together
  with the released pieces of its own detection, the level again with only the
  segments that the first pass kept, and the test is repeated once.

Every position is in pixels of the arrays the functions receive, ``x`` along
the columns.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import find_objects, label

from baymorph.preparation.detection import Detection
from baymorph.preparation.profile import Ellipse, ReferenceSky, ellipse_of_segment

CONNECTIVITY = np.ones((3, 3), dtype=bool)
"""Pixels connect along edges or corners, as in the detection."""

EVERY_OTHER_SEGMENT = "every other segment"
"""The light level of the first pass leaves out every segment but the target,
as in 0.6."""

OTHER_DETECTIONS = "the other detections"
"""The light level of the first pass leaves out only the segments of the other
detections: the pieces of the target's own detection enter its medians."""

LEVEL_MASKS = (EVERY_OTHER_SEGMENT, OTHER_DETECTIONS)
"""What the light level of the first pass may leave out, besides the excluded
pixels."""


# --- The light level -----------------------------------------------------------


def light_level(
    smoothed: ArrayLike,
    ellipse: Ellipse,
    masked: ArrayLike,
    *,
    sky: tuple[float, float, float] | None = None,
    sky_radius: float | None = None,
) -> NDArray[np.float64]:
    """Return the light level of every pixel about the centre of an ellipse.

    The annuli are 1 pixel wide in the semi-major axis of ``ellipse``, the
    annulus of a pixel being the integer part of its elliptical radius. Each
    annulus takes the median of the unmasked pixels of ``smoothed``; an
    annulus without such pixels takes the value interpolated linearly in
    radius from the nearest annuli that have them, and the nearest value
    beyond them. With ``sky`` and ``sky_radius``, the pixels at
    ``sky_radius`` and beyond take the plane ``sky`` instead, and only the
    pixels inside it enter the medians.

    Parameters
    ----------
    smoothed : array-like
        The smoothed image of the detection, less its sky plane.
    ellipse : Ellipse
        The annuli, about the target position.
    masked : array-like of bool
        Pixels left out of the medians, ``True`` = excluded.
    sky : tuple of float, optional
        A plane ``(level, gradient_x, gradient_y)`` about the centre of
        ``ellipse``, in the units of ``smoothed``: the sky beyond
        ``sky_radius``, such as the plane of the far sky less that of the
        detection.
    sky_radius : float, optional
        The semi-major axis from which the level is ``sky``, such as the
        outer edge of the last complete annulus. Given with ``sky`` only.

    Returns
    -------
    ndarray
        The level of the annulus of every pixel, of the shape of ``smoothed``.

    Raises
    ------
    ValueError
        If the arrays do not share one two-dimensional shape, an unmasked
        pixel is not finite, every pixel is masked, or ``sky`` and
        ``sky_radius`` are not given together, finite, with a radius of at
        least 0.
    """
    data = np.asarray(smoothed, dtype=float)
    out = np.asarray(masked, dtype=bool)

    if data.ndim != 2:
        raise ValueError("The image of a light level must be two-dimensional.")

    if out.shape != data.shape:
        raise ValueError(
            f"The masked pixels have the shape {out.shape}, the image {data.shape}."
        )

    if not np.all(np.isfinite(data[~out])):
        raise ValueError("A light level reads only finite pixels.")

    if (sky is None) != (sky_radius is None):
        raise ValueError("The sky of a light level needs its plane and its radius.")

    plane = None if sky is None else tuple(float(value) for value in sky)

    if plane is not None and (
        len(plane) != 3
        or not all(math.isfinite(value) for value in plane)
        or not math.isfinite(float(sky_radius))
        or float(sky_radius) < 0.0
    ):
        raise ValueError(
            "The sky of a light level is a finite plane (level, gradient_x, "
            "gradient_y) and a finite radius of at least 0; got "
            f"{sky!r} and {sky_radius!r}."
        )

    radius, _ = ellipse.coordinates(data.shape)
    beyond = (
        np.zeros(data.shape, dtype=bool)
        if plane is None
        else radius >= float(sky_radius)
    )
    index = np.floor(radius).astype(np.int64)
    count = int(index.max()) + 1
    read = ~out & ~beyond
    bins = index[read]

    if bins.size == 0 and not np.all(beyond):
        raise ValueError("The light level has no unmasked pixel to read.")

    levels = np.zeros(count)

    if bins.size:
        values = data[read]
        order = np.lexsort((values, bins))
        bins = bins[order]
        values = values[order]
        starts = np.searchsorted(bins, np.arange(count + 1))
        sizes = np.diff(starts)
        held = np.flatnonzero(sizes > 0)
        first = starts[held]
        # The median of each sorted annulus, as numpy.median gives it.
        medians = 0.5 * (
            values[first + (sizes[held] - 1) // 2] + values[first + sizes[held] // 2]
        )
        levels = np.interp(np.arange(count), held, medians)

    level = levels[index]

    if plane is not None and np.any(beyond):
        rows, columns = np.nonzero(beyond)
        level[rows, columns] = (
            plane[0]
            + plane[1] * (columns - ellipse.x0)
            + plane[2] * (rows - ellipse.y0)
        )

    return level


# --- The test of a segment ------------------------------------------------------


def segments_above_level(
    smoothed: ArrayLike,
    labels: ArrayLike,
    tested: ArrayLike,
    *,
    level: ArrayLike,
    excess: float,
    n_pixels: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return which segments still hold a source above a level, and where.

    A segment holds a source when at least ``n_pixels`` of its pixels,
    connected along edges or corners as in the detection, exceed ``level``
    by more than ``excess``. Those groups of pixels are its footprint.

    Parameters
    ----------
    smoothed : array-like
        The smoothed image of the detection.
    labels : array-like of int
        The map of segments, 0 where there is no source.
    tested : array-like of int
        Positive labels of the segments to test.
    level : array-like
        The level each pixel is compared with, such as the sky plus the
        smooth light of the target; the shape of ``smoothed``.
    excess : float
        The excess above ``level`` that a pixel must exceed: the detection
        threshold above its sky.
    n_pixels : int
        The fewest connected pixels above the level, as in the detection.

    Returns
    -------
    holds : ndarray of bool
        For each entry of ``tested``, whether the segment holds a source.
    footprint : ndarray of bool
        The pixels of the footprints of the tested segments that hold one.

    Raises
    ------
    ValueError
        If the arrays do not share one two-dimensional shape, an array is not
        finite, ``excess`` is not finite and positive, ``n_pixels`` is not a
        positive integer, or a label is not a segment of ``labels``.
    """
    image = np.asarray(smoothed, dtype=float)
    segments = np.asarray(labels)
    local_level = np.asarray(level, dtype=float)
    candidates = np.asarray(tested)

    if image.ndim != 2:
        raise ValueError("The smoothed image must be two-dimensional.")

    if segments.shape != image.shape or local_level.shape != image.shape:
        raise ValueError(
            "The smoothed image, the map of segments and the level must have the "
            "same shape."
        )

    if not np.issubdtype(segments.dtype, np.integer):
        raise ValueError("The map of segments must hold integer labels.")

    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(local_level)):
        raise ValueError(
            "The smoothed image and the level must contain only finite values."
        )

    if (
        isinstance(excess, bool)
        or not isinstance(excess, Real)
        or not math.isfinite(excess)
        or excess <= 0.0
    ):
        raise ValueError(f"The excess must be finite and positive; got {excess!r}.")

    if isinstance(n_pixels, bool) or not isinstance(n_pixels, Integral) or n_pixels < 1:
        raise ValueError(f"n_pixels must be a positive integer; got {n_pixels!r}.")

    if candidates.ndim != 1 or not (
        candidates.size == 0 or np.issubdtype(candidates.dtype, np.integer)
    ):
        raise ValueError("The tested labels must be a one-dimensional integer array.")

    holds = np.zeros(candidates.shape, dtype=bool)
    footprint = np.zeros(image.shape, dtype=bool)

    if candidates.size == 0:
        return holds, footprint

    windows = find_objects(segments)
    above = (image - local_level) > float(excess)

    for position, value in enumerate(candidates):
        number = int(value)

        if number < 1 or number > len(windows) or windows[number - 1] is None:
            raise ValueError(f"Label {number} is not a segment of the map.")

        window = windows[number - 1]
        inside = above[window] & (segments[window] == number)

        if np.count_nonzero(inside) < n_pixels:
            continue

        groups, count = label(inside, structure=CONNECTIVITY)
        sizes = np.bincount(groups.ravel(), minlength=count + 1)
        large = sizes >= n_pixels
        large[0] = False

        if np.any(large):
            holds[position] = True
            footprint[window] |= large[groups]

    return holds, footprint


# --- The release ------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Release:
    """Which segments hold a source above the light of the target.

    Parameters
    ----------
    target : int
        The label of the target segment, which is never tested.
    kept : tuple of int
        The other segments that hold a source, in increasing order.
    released : tuple of int
        The other segments that hold none, in increasing order.
    footprint : ndarray of bool
        The footprints of the kept segments: their pixels above the light
        level by the excess, in connected groups of at least ``n_pixels``.
    ellipse : Ellipse
        The annuli of the light level of the last pass.
    first_released : tuple of int
        The segments that the first pass released.
    siblings : tuple of int
        The other segments of the detection of the target before the
        deblending.
    passes : int
        The number of passes, 1 or 2.
    excess : float
        The excess of the detection threshold above its sky.
    n_pixels : int
        The fewest connected pixels of a source.
    level_masks : str, default=EVERY_OTHER_SEGMENT
        What the light level of the first pass left out, one of
        :data:`LEVEL_MASKS`.
    sky_radius : float or None, default=None
        The radius beyond which the light level of the last pass was the
        plane of the far sky; None when the level was the median of the
        annuli everywhere.
    """

    target: int
    kept: tuple[int, ...]
    released: tuple[int, ...]
    footprint: NDArray[np.bool_] = field(repr=False)
    ellipse: Ellipse
    first_released: tuple[int, ...]
    siblings: tuple[int, ...]
    passes: int
    excess: float
    n_pixels: int
    level_masks: str = EVERY_OTHER_SEGMENT
    sky_radius: float | None = None

    @property
    def changed(self) -> bool:
        """bool : Whether the second pass changed the released segments."""
        return self.passes == 2 and self.first_released != self.released

    @property
    def released_siblings(self) -> tuple[int, ...]:
        """tuple of int : The released segments of the detection of the target."""
        siblings = set(self.siblings)

        return tuple(number for number in self.released if number in siblings)

    def kept_segments(self, labels: ArrayLike) -> NDArray[np.bool_]:
        """Return the pixels of the kept segments of a map of segments."""
        return np.isin(np.asarray(labels), np.asarray(self.kept, dtype=np.int64))

    def as_dict(self) -> dict[str, Any]:
        """Return the release for a record, JSON-serializable."""
        return {
            "target": self.target,
            "kept": len(self.kept),
            "released": len(self.released),
            "released_labels": list(self.released),
            "siblings": len(self.siblings),
            "siblings_released": len(self.released_siblings),
            "footprint_pixels": int(np.count_nonzero(self.footprint)),
            "passes": self.passes,
            "second_pass_changed": self.changed,
            "ellipse": {
                "q": self.ellipse.q,
                "theta": self.ellipse.theta,
                "from": (
                    "the target segment"
                    if self.passes == 1
                    else "the target segment and its released siblings"
                ),
            },
            "excess": self.excess,
            "n_pixels": self.n_pixels,
            "level": {
                "first_pass_leaves_out": self.level_masks,
                "beyond_the_last_complete_annulus": (
                    "the median of the annuli"
                    if self.sky_radius is None
                    else "the plane of the far sky"
                ),
                "sky_radius": self.sky_radius,
            },
            "rule": (
                "a segment holds a source when at least n_pixels of its pixels, "
                "connected along edges or corners, exceed the light level of the "
                "target by the excess of the detection threshold"
            ),
        }


def release_segments(
    detection: Detection,
    target_label: int,
    image: ArrayLike,
    *,
    x0: float,
    y0: float,
    passes: int = 2,
    level_masks: str = EVERY_OTHER_SEGMENT,
    far_sky: Callable[[Ellipse], ReferenceSky] | None = None,
) -> Release:
    """Test every segment but the target against the light of the target.

    The first pass measures the ellipse on the target segment and the light
    level without the segments that ``level_masks`` names, and tests every
    other segment. The second pass measures the ellipse on the target segment
    together with the released segments of its own detection, the light level
    with only the segments that the first pass kept masked, and tests every
    other segment again; its result is final. With ``far_sky``, the level of
    each pass is the plane of the far sky of its ellipse beyond the radius of
    that far sky, the last complete annulus.

    Parameters
    ----------
    detection : Detection
        The sources of the image, with the smoothed image, its threshold and
        the map before the deblending.
    target_label : int
        The label of the target segment.
    image : array-like
        The image whose moments give the ellipse.
    x0, y0 : float
        The target position, in pixels of the arrays, the centre of the
        annuli (AD-055).
    passes : {1, 2}, default=2
        The number of passes.
    level_masks : str, default=EVERY_OTHER_SEGMENT
        What the level of the first pass leaves out besides the excluded
        pixels: :data:`EVERY_OTHER_SEGMENT`, as in 0.6, or
        :data:`OTHER_DETECTIONS`.
    far_sky : callable, optional
        A function that returns the far sky of an ellipse, a
        :class:`~baymorph.preparation.profile.ReferenceSky` whose radius is
        the outer edge of the last complete annulus and whose plane is given
        about the centre of the ellipse, in the units of ``image``. A plane
        is unchanged by the smoothing of the detection, up to its edges, so
        the level beyond that radius is the plane less that of the
        detection.

    Returns
    -------
    Release
        The kept and released segments, the footprints and the ellipse.

    Raises
    ------
    ValueError
        If ``passes`` is not 1 or 2, ``level_masks`` is not one of
        :data:`LEVEL_MASKS`, the target label is not a segment, or the image
        does not have the shape of the detection.
    """
    if passes not in (1, 2) or isinstance(passes, bool):
        raise ValueError(f"The release makes 1 or 2 passes; got {passes!r}.")

    if level_masks not in LEVEL_MASKS:
        raise ValueError(
            f"The light level of the first pass leaves out one of {LEVEL_MASKS}; "
            f"got {level_masks!r}."
        )

    data = np.asarray(image, dtype=float)
    labels = detection.labels

    if data.shape != labels.shape:
        raise ValueError(
            f"The image has the shape {data.shape}, the detection {labels.shape}."
        )

    target = labels == int(target_label)

    if not np.any(target):
        raise ValueError(f"Label {target_label} is not a segment of the detection.")

    n_pixels = int(detection.settings["n_pixels"])
    excess = float(detection.threshold)
    others = np.unique(labels[(labels > 0) & ~target]).astype(np.int64)
    parent = int(detection.parents[target][0])
    siblings = np.unique(
        labels[(detection.parents == parent) & (labels > 0) & ~target]
    ).astype(np.int64)
    ellipse = ellipse_of_segment(data, target, x0=x0, y0=y0)

    if others.size == 0:
        return Release(
            target=int(target_label),
            kept=(),
            released=(),
            footprint=np.zeros(labels.shape, dtype=bool),
            ellipse=ellipse,
            first_released=(),
            siblings=(),
            passes=passes,
            excess=excess,
            n_pixels=n_pixels,
            level_masks=level_masks,
            sky_radius=None,
        )

    if level_masks == EVERY_OTHER_SEGMENT:
        masked = ((labels > 0) & ~target) | detection.excluded
    else:
        masked = ((labels > 0) & (detection.parents != parent)) | detection.excluded

    sky, sky_radius = _sky_of(far_sky, ellipse, detection)
    level = light_level(
        detection.smoothed, ellipse, masked, sky=sky, sky_radius=sky_radius
    )
    holds, footprint = segments_above_level(
        detection.smoothed,
        labels,
        others,
        level=level,
        excess=excess,
        n_pixels=n_pixels,
    )
    first_released = others[~holds]

    if passes == 2:
        body = target | np.isin(labels, np.intersect1d(siblings, first_released))
        ellipse = ellipse_of_segment(data, body, x0=x0, y0=y0)
        masked = np.isin(labels, others[holds]) | detection.excluded
        sky, sky_radius = _sky_of(far_sky, ellipse, detection)
        level = light_level(
            detection.smoothed, ellipse, masked, sky=sky, sky_radius=sky_radius
        )
        holds, footprint = segments_above_level(
            detection.smoothed,
            labels,
            others,
            level=level,
            excess=excess,
            n_pixels=n_pixels,
        )

    footprint.flags.writeable = False

    return Release(
        target=int(target_label),
        kept=tuple(int(number) for number in others[holds]),
        released=tuple(int(number) for number in others[~holds]),
        footprint=footprint,
        ellipse=ellipse,
        first_released=tuple(int(number) for number in first_released),
        siblings=tuple(int(number) for number in siblings),
        passes=passes,
        excess=excess,
        n_pixels=n_pixels,
        level_masks=level_masks,
        sky_radius=sky_radius,
    )


def _sky_of(
    far_sky: Callable[[Ellipse], ReferenceSky] | None,
    ellipse: Ellipse,
    detection: Detection,
) -> tuple[tuple[float, float, float] | None, float | None]:
    """Return the plane of the far sky of an ellipse in the units of the
    smoothed image, about the centre of the ellipse, and its radius."""
    if far_sky is None:
        return None, None

    reference = far_sky(ellipse)

    if not isinstance(reference, ReferenceSky):
        raise TypeError(
            "far_sky returns the ReferenceSky of an ellipse, not "
            f"{type(reference).__name__}."
        )

    height, width = detection.labels.shape
    level, gradient_x, gradient_y = detection.plane
    # The plane of the detection is given about the centre of the image.
    at_centre = (
        level
        + gradient_x * (ellipse.x0 - (width - 1) / 2.0)
        + gradient_y * (ellipse.y0 - (height - 1) / 2.0)
    )
    sky = (
        reference.plane[0] - at_centre,
        reference.plane[1] - gradient_x,
        reference.plane[2] - gradient_y,
    )

    return sky, float(reference.radius)


__all__ = [
    "CONNECTIVITY",
    "EVERY_OTHER_SEGMENT",
    "LEVEL_MASKS",
    "OTHER_DETECTIONS",
    "Release",
    "light_level",
    "release_segments",
    "segments_above_level",
]
