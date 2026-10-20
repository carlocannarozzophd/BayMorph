"""Source detection and segmentation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from photutils.segmentation import (
    SourceCatalog,
    deblend_sources,
    detect_sources,
    detect_threshold,
)
from scipy.ndimage import find_objects, gaussian_filter, label


@dataclass
class DetectionResult:
    """Container for source detection results.

    Parameters
    ----------
    segmentation : photutils.segmentation.SegmentationImage or None
        Segmentation image returned by Photutils. This is ``None`` if no
        sources are detected.
    mask : ndarray of bool
        Boolean mask identifying detected source pixels.
    catalog : photutils.segmentation.SourceCatalog, optional
        Source catalog constructed from the segmentation image. This is
        ``None`` if no sources are detected or if catalog construction fails.
    detection_image : ndarray, optional
        The smoothed image on which the sources were detected.
    background_level : float, optional
        Sigma-clipped mean of ``detection_image``, the level above which the
        threshold is set.
    threshold_level : float, optional
        Detection threshold: ``background_level`` plus ``n_sigma`` times the
        sigma-clipped standard deviation of ``detection_image``.
    excluded : ndarray of bool, optional
        Pixels left out of the detection, such as pixels without data, when
        there were any; ``detection_image`` holds ``background_level`` there.
    """

    segmentation: object
    mask: np.ndarray
    catalog: object | None = None
    detection_image: np.ndarray | None = None
    background_level: float | None = None
    threshold_level: float | None = None
    excluded: np.ndarray | None = None


def detect_sources_segmentation(
    data,
    n_sigma: float = 3.0,
    n_pixels: int = 5,
    n_levels: int = 32,
    deblend: bool = True,
    deblend_contrast: float = 0.001,
    mask=None,
) -> DetectionResult:
    """Detect and optionally deblend sources using Photutils.

    The input image is Gaussian-smoothed to improve source detection. A
    detection threshold is estimated with
    :func:`photutils.segmentation.detect_threshold`, followed by connected
    source detection with :func:`photutils.segmentation.detect_sources`.

    When ``deblend`` is True, overlapping detections are subsequently
    separated using multi-threshold watershed deblending.

    Pixels of ``mask`` take no part in the detection: the smoothing averages
    the other pixels only, weighting each by the Gaussian kernel, the
    threshold is measured without them, and no segment contains them. Their
    values may be non-finite. Without excluded pixels the detection is the
    one of an image without a mask.

    Parameters
    ----------
    data : array-like
        Input image.
    n_sigma : float, default=3.0
        Detection threshold in units of the estimated background noise.
    n_pixels : int, default=5
        Minimum number of connected pixels required for detection and
        deblending.
    n_levels : int, default=32
        Number of multi-threshold levels used during deblending.
    deblend : bool, default=True
        Whether to deblend connected sources after initial detection.
    deblend_contrast : float, default=0.001
        Minimum fractional source flux required for a local peak to be
        retained as a separate deblended source.
    mask : array-like of bool, optional
        Pixels to leave out of the detection, ``True`` = excluded, such as
        the pixels without data of :func:`baymorph.core.mask.no_data_mask`;
        the shape of ``data``.

    Returns
    -------
    DetectionResult
        Container holding the segmentation image, detection mask, and source
        catalog when available.

    Raises
    ------
    ValueError
        If ``mask`` does not have the shape of ``data``, a pixel outside it
        is not finite, or it leaves no pixel to detect on.
    """
    data = np.asarray(data, dtype=float)
    excluded = _excluded_pixels(mask, data)

    if excluded is None:
        detection_image = gaussian_filter(data, sigma=2.0)
    else:
        detection_image = _smooth_included_pixels(data, excluded, sigma=2.0)

    threshold = detect_threshold(detection_image, n_sigma=n_sigma, mask=excluded)
    background_level = float(
        detect_threshold(detection_image, n_sigma=0.0, mask=excluded).flat[0]
    )
    threshold_level = float(threshold.flat[0])

    if excluded is not None:
        detection_image[excluded] = background_level

    segmentation = detect_sources(
        detection_image, threshold, n_pixels=n_pixels, mask=excluded
    )

    if segmentation is None:
        source_mask = np.zeros(data.shape, dtype=bool)

        return DetectionResult(
            segmentation=None,
            mask=source_mask,
            catalog=None,
            detection_image=detection_image,
            background_level=background_level,
            threshold_level=threshold_level,
            excluded=excluded,
        )

    if deblend:
        segmentation = deblend_sources(
            detection_image,
            segmentation,
            n_pixels=n_pixels,
            n_levels=n_levels,
            contrast=deblend_contrast,
            progress_bar=False,
        )

    catalog = SourceCatalog(data, segmentation, mask=excluded)

    return DetectionResult(
        segmentation=segmentation,
        mask=segmentation.data > 0,
        catalog=catalog,
        detection_image=detection_image,
        background_level=background_level,
        threshold_level=threshold_level,
        excluded=excluded,
    )


def _excluded_pixels(mask, data: np.ndarray) -> NDArray[np.bool_] | None:
    """Return the validated excluded pixels, or None when there are none.

    Without excluded pixels every pixel must be finite, and the detection
    runs exactly as it does without a mask.
    """
    excluded = None

    if mask is not None:
        excluded = np.asarray(mask)

        if excluded.shape != data.shape:
            raise ValueError(
                f"The detection mask of shape {excluded.shape} does not have "
                f"the shape {data.shape} of the image."
            )

        excluded = excluded.astype(bool)

        if not np.any(excluded):
            excluded = None

    included = np.isfinite(data) if excluded is None else np.isfinite(data[~excluded])

    if not np.all(included):
        raise ValueError(
            f"The image holds {int(np.count_nonzero(~included))} non-finite "
            "pixels that the detection mask does not exclude."
        )

    if excluded is not None and np.all(excluded):
        raise ValueError("The detection mask excludes every pixel of the image.")

    return excluded


def _smooth_included_pixels(
    data: np.ndarray, excluded: NDArray[np.bool_], *, sigma: float
) -> NDArray[np.floating]:
    """Smooth an image with a Gaussian kernel over its included pixels only.

    Each pixel takes the mean of the included pixels within reach of the
    kernel, weighted by it (normalized convolution), so the excluded pixels
    neither lower nor raise the pixels next to them. A pixel that the kernel
    reaches no included pixel from takes 0; the caller replaces the values
    of the excluded pixels.
    """
    weights = gaussian_filter((~excluded).astype(float), sigma=sigma)
    smoothed = gaussian_filter(np.where(excluded, 0.0, data), sigma=sigma)

    return np.divide(
        smoothed, weights, out=np.zeros_like(smoothed), where=weights > 0.0
    )


def segments_above_level(
    detection_image: NDArray[np.floating],
    segmentation: NDArray[np.integer],
    labels: NDArray[np.integer],
    *,
    level: NDArray[np.floating],
    excess: float,
    n_pixels: int,
) -> NDArray[np.bool_]:
    """Return which segments still hold a source above a local level.

    The detection compares the smoothed image with one threshold, a constant
    excess above the sigma-clipped background of the whole image. Where the
    smooth light of another source raises the image, peaks of the noise
    clear that threshold and become segments of their own. This function
    repeats the test of the detection inside each segment with the threshold
    raised by ``level``: a segment holds a source when at least ``n_pixels``
    of its pixels, connected along edges or corners as in the detection,
    exceed ``level`` by more than ``excess``.

    Parameters
    ----------
    detection_image : ndarray
        The smoothed image of the detection.
    segmentation : ndarray of int
        Segmentation map of the detection, 0 for no source.
    labels : ndarray of int
        Positive labels of the segments to test.
    level : ndarray
        The level each pixel is compared with, such as the background plus
        the smooth light of the target; the shape of ``detection_image``.
    excess : float
        Excess above ``level`` that a pixel must exceed: the detection
        threshold minus its background level.
    n_pixels : int
        Smallest number of connected pixels above the level, as in the
        detection.

    Returns
    -------
    ndarray of bool
        For each entry of ``labels``, whether the segment holds a source.

    Raises
    ------
    ValueError
        If the arrays do not share one two-dimensional shape, an array is
        not finite, ``excess`` is not finite and positive, ``n_pixels`` is
        not a positive integer, or a label is not a segment of
        ``segmentation``.
    """
    image = np.asarray(detection_image, dtype=float)
    segments = np.asarray(segmentation)
    local_level = np.asarray(level, dtype=float)
    tested = np.asarray(labels)

    if image.ndim != 2:
        raise ValueError("'detection_image' must be a two-dimensional array.")
    if segments.shape != image.shape or local_level.shape != image.shape:
        raise ValueError(
            "'detection_image', 'segmentation' and 'level' must have the " "same shape."
        )
    if not np.issubdtype(segments.dtype, np.integer):
        raise ValueError("'segmentation' must hold integer labels.")
    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(local_level)):
        raise ValueError(
            "'detection_image' and 'level' must contain only finite values."
        )
    if not np.isfinite(excess) or excess <= 0.0:
        raise ValueError("'excess' must be finite and positive.")
    if isinstance(n_pixels, bool) or int(n_pixels) != n_pixels or n_pixels < 1:
        raise ValueError("'n_pixels' must be a positive integer.")
    if tested.ndim != 1 or not np.issubdtype(tested.dtype, np.integer):
        raise ValueError("'labels' must be a one-dimensional integer array.")

    slices = find_objects(segments)
    above = (image - local_level) > float(excess)
    connectivity = np.ones((3, 3), dtype=bool)
    holds_source = np.zeros(tested.shape, dtype=bool)

    for index, segment_label in enumerate(tested):
        segment_label = int(segment_label)
        if (
            segment_label < 1
            or segment_label > len(slices)
            or slices[segment_label - 1] is None
        ):
            raise ValueError(
                f"Label {segment_label} is not a segment of 'segmentation'."
            )

        window = slices[segment_label - 1]
        candidates = above[window] & (segments[window] == segment_label)
        if np.count_nonzero(candidates) < n_pixels:
            continue

        components, count = label(candidates, structure=connectivity)
        sizes = np.bincount(components.ravel(), minlength=count + 1)[1:]
        holds_source[index] = bool(np.any(sizes >= n_pixels))

    return holds_source
