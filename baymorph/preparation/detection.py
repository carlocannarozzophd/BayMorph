"""The detection of sources for the preparation.

The detection smooths the image with a Gaussian kernel, sets a threshold of
``n_sigma`` noise scales above the sky of the smoothed image, keeps the
connected areas of at least ``n_pixels`` pixels above it, and splits blended
sources by multi-threshold deblending (AD-003, Photutils). It ports
:func:`baymorph.core.detection.detect_sources_segmentation` with two changes:

- the sky of the smoothed image is a plane fitted with sigma clipping, not a
  constant, and the threshold is measured on the image less that plane, so a
  sky gradient no longer raises the threshold and hides faint sources;
- the map of segments before the deblending is kept beside the deblended
  map, so that a segment can be traced to the detection it was split from
  (:func:`baymorph.preparation.target.select_target_segment`).

Pixels of the excluded mask, such as the pixels without data, take no part in
the detection: the smoothing averages the other pixels only, the plane and
the threshold are measured without them, and no segment holds them.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from photutils.segmentation import deblend_sources, detect_sources
from photutils.utils.exceptions import NoDetectionsWarning
from scipy.ndimage import gaussian_filter

SMOOTHING_SIGMA = 2.0
"""Width, in pixels, of the Gaussian kernel that smooths the image."""

PLANE_CLIP_SIGMA = 3.0
"""Clipping, in scales of the residual, of the fit of the sky plane."""

PLANE_MAX_ITERATIONS = 10
"""Largest number of clipping passes of the fit of the sky plane."""


@dataclass(frozen=True, eq=False)
class Detection:
    """The sources detected on an image.

    Parameters
    ----------
    labels : ndarray of int32
        The map of segments after the deblending, 0 where there is no source.
    parents : ndarray of int32
        The map of segments before the deblending; every segment of
        ``labels`` lies inside one segment of it.
    smoothed : ndarray of float64
        The smoothed image less the sky plane, on which the sources were
        detected; the excluded pixels hold 0.
    plane : tuple of float
        The sky plane of the smoothed image, ``(level, gradient_x,
        gradient_y)``, with the level at the centre of the image and the
        gradients per pixel.
    noise : float
        The clipped scale of the smoothed image about the plane.
    threshold : float
        The detection threshold above the plane, ``n_sigma * noise``.
    excluded : ndarray of bool
        The pixels left out of the detection.
    settings : dict
        The settings of the detection.
    """

    labels: NDArray[np.int32] = field(repr=False)
    parents: NDArray[np.int32] = field(repr=False)
    smoothed: NDArray[np.float64] = field(repr=False)
    plane: tuple[float, float, float]
    noise: float
    threshold: float
    excluded: NDArray[np.bool_] = field(repr=False)
    settings: dict[str, Any]

    @property
    def count(self) -> int:
        """int : The number of segments after the deblending."""
        return int(np.unique(self.labels[self.labels > 0]).size)


def detect(
    image: ArrayLike,
    *,
    excluded: ArrayLike | None = None,
    n_sigma: float = 3.0,
    n_pixels: int = 20,
    n_levels: int = 32,
    contrast: float = 0.001,
) -> Detection:
    """Detect and deblend the sources of an image.

    Parameters
    ----------
    image : array-like
        Two-dimensional image, finite on every pixel that is not excluded.
    excluded : array-like of bool, optional
        Pixels left out of the detection, ``True`` = excluded.
    n_sigma : float, default=3.0
        Detection threshold, in noise scales of the smoothed image.
    n_pixels : int, default=20
        Smallest segment, in connected pixels, of the detection and of the
        deblending.
    n_levels : int, default=32
        Number of levels of the deblending.
    contrast : float, default=0.001
        Smallest fraction of the flux of a detection that a deblended segment
        holds.

    Returns
    -------
    Detection
        The maps of segments, before and after the deblending, and how they
        were obtained. Without any source both maps hold only zeros.

    Raises
    ------
    ValueError
        If the image is not two-dimensional, a pixel that is not excluded is
        not finite, every pixel is excluded, or a setting is not valid.
    """
    data = np.asarray(image, dtype=float)

    if data.ndim != 2 or data.size == 0:
        raise ValueError("The image of a detection must be two-dimensional.")

    out = _excluded(excluded, data)
    check_detection_settings(n_sigma, n_pixels, n_levels, contrast)

    weights = gaussian_filter((~out).astype(float), sigma=SMOOTHING_SIGMA)
    smoothed = gaussian_filter(np.where(out, 0.0, data), sigma=SMOOTHING_SIGMA)
    smoothed = np.divide(
        smoothed, weights, out=np.zeros_like(smoothed), where=weights > 0.0
    )
    plane, noise = fit_sky_plane(smoothed, out)
    residual = smoothed - plane_image(plane, data.shape)
    residual[out] = 0.0
    threshold = float(n_sigma) * noise
    settings = {
        "n_sigma": float(n_sigma),
        "n_pixels": int(n_pixels),
        "n_levels": int(n_levels),
        "contrast": float(contrast),
        "smoothing_sigma": SMOOTHING_SIGMA,
        "plane_clip_sigma": PLANE_CLIP_SIGMA,
    }
    found = None

    if noise > 0.0:
        # An image without sources gives empty maps, not a warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NoDetectionsWarning)
            found = detect_sources(
                residual, threshold, n_pixels=int(n_pixels), mask=out
            )

    if found is None:
        parents = np.zeros(data.shape, dtype=np.int32)
        labels = np.zeros(data.shape, dtype=np.int32)
    else:
        parents = np.asarray(found.data, dtype=np.int32)
        deblended = deblend_sources(
            residual,
            found,
            n_pixels=int(n_pixels),
            n_levels=int(n_levels),
            contrast=float(contrast),
            progress_bar=False,
        )
        labels = np.asarray(deblended.data, dtype=np.int32)

    for array in (labels, parents, residual, out):
        array.flags.writeable = False

    return Detection(
        labels=labels,
        parents=parents,
        smoothed=residual,
        plane=plane,
        noise=noise,
        threshold=threshold,
        excluded=out,
        settings=settings,
    )


def fit_sky_plane(
    image: ArrayLike,
    excluded: ArrayLike | None = None,
    *,
    clip: float = PLANE_CLIP_SIGMA,
    max_iterations: int = PLANE_MAX_ITERATIONS,
) -> tuple[tuple[float, float, float], float]:
    """Fit a sky plane to an image with iterative sigma clipping.

    Each pass fits the plane by least squares to the kept pixels and keeps
    those within ``clip`` standard deviations of it, until the kept pixels do
    not change. Sources lie above the sky, so they are clipped first.

    Parameters
    ----------
    image : array-like
        Two-dimensional image.
    excluded : array-like of bool, optional
        Pixels left out of the fit.
    clip : float, default=PLANE_CLIP_SIGMA
        Clipping, in standard deviations of the residual.
    max_iterations : int, default=PLANE_MAX_ITERATIONS
        Largest number of passes.

    Returns
    -------
    plane : tuple of float
        ``(level, gradient_x, gradient_y)``, with the level at the centre of
        the image, ``((nx - 1) / 2, (ny - 1) / 2)``.
    scale : float
        The standard deviation of the kept residuals.

    Raises
    ------
    ValueError
        If fewer than three pixels are left to fit.
    """
    data = np.asarray(image, dtype=float)
    out = (
        np.zeros(data.shape, dtype=bool)
        if excluded is None
        else np.asarray(excluded, dtype=bool)
    )
    rows, columns = np.nonzero(~out & np.isfinite(data))

    if rows.size < 3:
        raise ValueError("A sky plane needs at least three pixels.")

    x = columns - (data.shape[1] - 1) / 2.0
    y = rows - (data.shape[0] - 1) / 2.0
    values = data[rows, columns]
    design = np.column_stack([np.ones_like(x), x, y])
    keep = np.ones(values.size, dtype=bool)
    coefficients = np.zeros(3)
    scale = 0.0

    for _ in range(int(max_iterations)):
        coefficients, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
        residual = values - design @ coefficients
        scale = float(np.std(residual[keep]))

        if scale == 0.0:
            break

        updated = np.abs(residual) < float(clip) * scale

        if np.count_nonzero(updated) < 3 or np.array_equal(updated, keep):
            break

        keep = updated

    return (
        float(coefficients[0]),
        float(coefficients[1]),
        float(coefficients[2]),
    ), scale


def plane_image(
    plane: tuple[float, float, float], shape: tuple[int, int]
) -> NDArray[np.float64]:
    """Return the image of a sky plane given about the centre of the image."""
    level, gradient_x, gradient_y = plane
    rows, columns = np.indices(shape, dtype=float)

    return (
        level
        + gradient_x * (columns - (shape[1] - 1) / 2.0)
        + gradient_y * (rows - (shape[0] - 1) / 2.0)
    )


def _excluded(value: ArrayLike | None, data: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return the validated excluded pixels."""
    if value is None:
        out = np.zeros(data.shape, dtype=bool)
    else:
        out = np.asarray(value)

        if out.shape != data.shape:
            raise ValueError(
                f"The excluded pixels have the shape {out.shape}, the image "
                f"{data.shape}."
            )

        out = out.astype(bool)

    bad = ~out & ~np.isfinite(data)

    if np.any(bad):
        raise ValueError(
            f"The image holds {int(np.count_nonzero(bad))} non-finite pixels that "
            "are not excluded from the detection."
        )

    if np.all(out):
        raise ValueError("The detection excludes every pixel of the image.")

    return out


def check_detection_settings(
    n_sigma: Any, n_pixels: Any, n_levels: Any, contrast: Any
) -> None:
    """Refuse settings of the detection that are not valid.

    Raises
    ------
    ValueError
        If ``n_sigma`` is not a positive number, ``n_pixels`` or ``n_levels``
        is not a positive integer, or ``contrast`` does not lie in [0, 1].
    """
    if (
        isinstance(n_sigma, bool)
        or not isinstance(n_sigma, Real)
        or not math.isfinite(n_sigma)
        or n_sigma <= 0.0
    ):
        raise ValueError(f"n_sigma must be a positive number; got {n_sigma!r}.")

    for name, value in (("n_pixels", n_pixels), ("n_levels", n_levels)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer; got {value!r}.")

    if (
        isinstance(contrast, bool)
        or not isinstance(contrast, Real)
        or not 0.0 <= contrast <= 1.0
    ):
        raise ValueError(f"contrast must lie between 0 and 1; got {contrast!r}.")


__all__ = [
    "PLANE_CLIP_SIGMA",
    "PLANE_MAX_ITERATIONS",
    "SMOOTHING_SIGMA",
    "Detection",
    "check_detection_settings",
    "detect",
    "fit_sky_plane",
    "plane_image",
]
