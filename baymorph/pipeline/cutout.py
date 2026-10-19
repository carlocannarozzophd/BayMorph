"""Cutout-definition utilities for BayMorph image preparation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CutoutDefinition:
    """Definition of a resolved image cutout.

    Parameters
    ----------
    xmin, xmax : int
        Horizontal pixel bounds of the cutout.
    ymin, ymax : int
        Vertical pixel bounds of the cutout.
    width : int
        Width of the cutout, in pixels.
    height : int
        Height of the cutout, in pixels.
    mode : str
        Name of the cutout strategy used to generate the cutout.
    """

    xmin: int
    xmax: int

    ymin: int
    ymax: int

    width: int
    height: int

    mode: str


def _resolve_centered_cutout(
    *,
    image_shape: tuple[int, int],
    center_x: float,
    center_y: float,
    size: int,
    mode: str,
) -> CutoutDefinition:
    """Resolve and clip a centered square cutout."""
    if len(image_shape) != 2:
        raise ValueError("'image_shape' must contain exactly two dimensions.")

    if not np.isfinite(center_x) or not np.isfinite(center_y):
        raise ValueError("Cutout centre coordinates must be finite.")

    if size <= 0:
        raise ValueError("Cutout size must be positive.")

    ny, nx = image_shape

    if ny <= 0 or nx <= 0:
        raise ValueError("Image dimensions must be positive.")

    size = min(int(size), ny, nx)

    cx = int(round(center_x))
    cy = int(round(center_y))

    half = size // 2

    xmin = cx - half
    ymin = cy - half

    xmax = xmin + size
    ymax = ymin + size

    if xmin < 0:
        xmax -= xmin
        xmin = 0

    if ymin < 0:
        ymax -= ymin
        ymin = 0

    if xmax > nx:
        xmin -= xmax - nx
        xmax = nx

    if ymax > ny:
        ymin -= ymax - ny
        ymax = ny

    xmin = max(xmin, 0)
    ymin = max(ymin, 0)

    return CutoutDefinition(
        xmin=int(xmin),
        xmax=int(xmax),
        ymin=int(ymin),
        ymax=int(ymax),
        width=int(xmax - xmin),
        height=int(ymax - ymin),
        mode=mode,
    )


def _apply_size_limits(
    size: int, *, min_size: int | None = None, max_size: int | None = None
) -> int:
    """Apply optional lower and upper limits to a cutout size."""
    resolved_size = int(size)

    if min_size is not None:
        if min_size <= 0:
            raise ValueError("'min_size' must be positive.")

        resolved_size = max(resolved_size, int(min_size))

    if max_size is not None:
        if max_size <= 0:
            raise ValueError("'max_size' must be positive.")

        resolved_size = min(resolved_size, int(max_size))

    if min_size is not None and max_size is not None and min_size > max_size:
        raise ValueError("'min_size' cannot be greater than 'max_size'.")

    return resolved_size


def resolve_fixed_cutout(
    *, image_shape: tuple[int, int], center_x: float, center_y: float, size: int
) -> CutoutDefinition:
    """Resolve a fixed-size cutout centered on a target position.

    Parameters
    ----------
    image_shape : tuple of int
        Shape of the parent image as ``(ny, nx)``.
    center_x, center_y : float
        Coordinates of the target centre.
    size : int
        Requested square cutout size.

    Returns
    -------
    CutoutDefinition
        Fixed cutout clipped to the parent-image boundaries.
    """
    return _resolve_centered_cutout(
        image_shape=image_shape,
        center_x=center_x,
        center_y=center_y,
        size=int(size),
        mode="fixed",
    )


def resolve_preliminary_auto_cutout(
    *,
    image_shape: tuple[int, int],
    target_segment: np.ndarray,
    center_x: float,
    center_y: float,
    initial_factor: float = 4.0,
    min_size: int = 100,
    max_size: int | None = None,
) -> CutoutDefinition:
    """Resolve a preliminary adaptive cutout anchored to a target position.

    The preliminary cutout size is derived from the largest dimension of
    the target-segment bounding box multiplied by ``initial_factor``. Its
    centre remains fixed at the user-supplied target position rather than
    being inferred from the segment centroid.

    Anchoring the cutout to the supplied position prevents a bright nearby
    contaminant, or a blended target-contaminant segment, from displacing
    the preliminary cutout away from the intended source.

    Parameters
    ----------
    image_shape : tuple of int
        Shape of the parent image as ``(ny, nx)``.
    target_segment : ndarray of bool
        Boolean target-segment mask in parent-image coordinates.
    center_x, center_y : float
        User-supplied target position in parent-image coordinates.
    initial_factor : float, default=4.0
        Multiplicative expansion of the target bounding-box size.
    min_size : int, default=100
        Minimum preliminary cutout size.
    max_size : int, optional
        Maximum preliminary cutout size.

    Returns
    -------
    CutoutDefinition
        Preliminary adaptive cutout centred on the supplied target position.

    Raises
    ------
    ValueError
        If the segment, coordinates, or configuration are invalid.
    """
    if not np.isfinite(initial_factor):
        raise ValueError("'initial_factor' must be finite.")

    if initial_factor <= 1.0:
        raise ValueError("'initial_factor' must be greater than 1.")

    if not np.isfinite(center_x) or not np.isfinite(center_y):
        raise ValueError("Target coordinates must be finite.")

    segment = np.asarray(target_segment, dtype=bool)

    if segment.shape != tuple(image_shape):
        raise ValueError(
            "'target_segment' must have the same shape as the parent image."
        )

    y, x = np.nonzero(segment)

    if x.size == 0:
        raise ValueError("'target_segment' contains no target pixels.")

    xmin_segment = int(np.min(x))
    xmax_segment = int(np.max(x))
    ymin_segment = int(np.min(y))
    ymax_segment = int(np.max(y))

    segment_width = xmax_segment - xmin_segment + 1
    segment_height = ymax_segment - ymin_segment + 1

    segment_size = max(segment_width, segment_height)

    requested_size = int(np.ceil(initial_factor * segment_size))

    requested_size = _apply_size_limits(
        requested_size, min_size=min_size, max_size=max_size
    )

    return _resolve_centered_cutout(
        image_shape=image_shape,
        center_x=float(center_x),
        center_y=float(center_y),
        size=requested_size,
        mode="auto_preliminary",
    )


def resolve_plateau_cutout(
    *,
    image_shape: tuple[int, int],
    center_x: float,
    center_y: float,
    plateau_radius: float,
    sky_margin: float = 20.0,
    min_size: int | None = None,
    max_size: int | None = None,
) -> CutoutDefinition:
    """Resolve the final cutout from a detected profile plateau.

    The final half-size is the plateau radius plus an additional sky
    margin. The resulting square cutout therefore has an approximate size

    ``2 * (plateau_radius + sky_margin)``.

    Parameters
    ----------
    image_shape : tuple of int
        Shape of the parent image as ``(ny, nx)``.
    center_x, center_y : float
        Target centre in parent-image coordinates.
    plateau_radius : float
        Elliptical radius at which the target profile reaches the sky.
    sky_margin : float, default=20.0
        Additional radial margin beyond the plateau.
    min_size : int, optional
        Minimum final cutout size.
    max_size : int, optional
        Maximum final cutout size.

    Returns
    -------
    CutoutDefinition
        Final plateau-based cutout.
    """
    if not np.isfinite(plateau_radius):
        raise ValueError("'plateau_radius' must be finite.")

    if plateau_radius < 0.0:
        raise ValueError("'plateau_radius' must be non-negative.")

    if not np.isfinite(sky_margin):
        raise ValueError("'sky_margin' must be finite.")

    if sky_margin < 0.0:
        raise ValueError("'sky_margin' must be non-negative.")

    half_size = plateau_radius + sky_margin

    requested_size = 2 * int(np.ceil(half_size)) + 1

    requested_size = _apply_size_limits(
        requested_size, min_size=min_size, max_size=max_size
    )

    return _resolve_centered_cutout(
        image_shape=image_shape,
        center_x=center_x,
        center_y=center_y,
        size=requested_size,
        mode="auto",
    )


def validate_plateau_cutout_containment(
    *,
    image_shape: tuple[int, int],
    center_x: float,
    center_y: float,
    plateau_radius: float,
    sky_margin: float,
) -> None:
    """Require a cutout to contain its plateau and sky margin."""
    if len(image_shape) != 2:
        raise ValueError("'image_shape' must contain exactly two dimensions.")

    ny, nx = (int(image_shape[0]), int(image_shape[1]))
    if ny <= 0 or nx <= 0:
        raise ValueError("Image dimensions must be positive.")

    values = {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "plateau_radius": float(plateau_radius),
        "sky_margin": float(sky_margin),
    }

    for name, value in values.items():
        if not np.isfinite(value):
            raise ValueError(f"'{name}' must be finite.")

    if values["plateau_radius"] < 0.0:
        raise ValueError("'plateau_radius' must be non-negative.")

    if values["sky_margin"] < 0.0:
        raise ValueError("'sky_margin' must be non-negative.")

    distances = {
        "left": values["center_x"] + 0.5,
        "right": nx - 0.5 - values["center_x"],
        "bottom": values["center_y"] + 0.5,
        "top": ny - 0.5 - values["center_y"],
    }

    if min(distances.values()) < 0.0:
        raise ValueError("The fitted centre lies outside the final cutout.")

    required = values["plateau_radius"] + values["sky_margin"]
    failing = {
        edge: distance
        for edge, distance in distances.items()
        if distance + 1.0e-9 < required
    }

    if failing:
        available = ", ".join(
            f"{edge}={distance:.3f}" for edge, distance in failing.items()
        )
        raise RuntimeError(
            "Automatic cutout does not contain the final plateau radius "
            f"plus sky margin: required={required:.3f} pixels; "
            f"insufficient edge distances: {available}. Increase the "
            "automatic cutout limit or use a scientifically fixed cutout."
        )


def resolve_cutout(
    *, image_shape: tuple[int, int], center_x: float, center_y: float, config: dict
) -> CutoutDefinition:
    """Resolve a directly constructible cutout from configuration.

    This dispatcher handles fixed cutouts. Automatic cutouts require source
    segmentation and plateau detection, and are therefore resolved through
    :func:`resolve_preliminary_auto_cutout` and
    :func:`resolve_plateau_cutout` during image preparation.

    Parameters
    ----------
    image_shape : tuple of int
        Shape of the parent image as ``(ny, nx)``.
    center_x, center_y : float
        Target coordinates.
    config : dict
        Cutout configuration dictionary.

    Returns
    -------
    CutoutDefinition
        Resolved cutout definition.

    Raises
    ------
    ValueError
        If the mode is unsupported or requires pipeline-level information.
    """
    mode = config["mode"]

    if mode == "fixed":
        return resolve_fixed_cutout(
            image_shape=image_shape,
            center_x=center_x,
            center_y=center_y,
            size=int(config["size"]),
        )

    if mode == "auto":
        raise ValueError(
            "Automatic cutouts must be resolved by the image-preparation "
            "pipeline after target segmentation."
        )

    raise ValueError(f"Unsupported cutout mode: {mode!r}")
