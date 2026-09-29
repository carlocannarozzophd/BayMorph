"""Source detection and segmentation utilities."""

from dataclasses import dataclass

import numpy as np
from photutils.segmentation import (
    SourceCatalog,
    deblend_sources,
    detect_sources,
    detect_threshold,
)
from scipy.ndimage import gaussian_filter


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
    """

    segmentation: object
    mask: np.ndarray
    catalog: object | None = None


def detect_sources_segmentation(
    data,
    n_sigma: float = 3.0,
    n_pixels: int = 5,
    n_levels: int = 32,
    deblend: bool = True,
    deblend_contrast: float = 0.001,
) -> DetectionResult:
    """Detect and optionally deblend sources using Photutils.

    The input image is Gaussian-smoothed to improve source detection. A
    detection threshold is estimated with
    :func:`photutils.segmentation.detect_threshold`, followed by connected
    source detection with :func:`photutils.segmentation.detect_sources`.

    When ``deblend`` is True, overlapping detections are subsequently
    separated using multi-threshold watershed deblending.

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

    Returns
    -------
    DetectionResult
        Container holding the segmentation image, detection mask, and source
        catalog when available.
    """
    data = np.asarray(
        data,
        dtype=float,
    )

    detection_image = gaussian_filter(
        data,
        sigma=2.0,
    )

    threshold = detect_threshold(
        detection_image,
        n_sigma=n_sigma,
    )

    segmentation = detect_sources(
        detection_image,
        threshold,
        n_pixels=n_pixels,
    )

    if segmentation is None:
        mask = np.zeros(
            data.shape,
            dtype=bool,
        )

        return DetectionResult(
            segmentation=None,
            mask=mask,
            catalog=None,
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

    mask = segmentation.data > 0

    catalog = SourceCatalog(
        data,
        segmentation,
    )

    return DetectionResult(
        segmentation=segmentation,
        mask=mask,
        catalog=catalog,
    )


def detect_sources_mask(
    data,
    n_sigma: float = 3.0,
    n_pixels: int = 5,
    n_levels: int = 32,
    deblend: bool = True,
    deblend_contrast: float = 0.001,
):
    """Detect sources and return a boolean source mask.

    Warning
    -------
    This mask uses ``True`` = detected source pixel, the *opposite*
    convention from likelihood exclusion masks (``True`` = excluded) used
    throughout :mod:`baymorph.core.mask` and :mod:`baymorph.likelihood`. Do
    not pass this mask directly to a likelihood; invert it first if it is
    to be used for exclusion.

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

    Returns
    -------
    ndarray of bool
        Boolean mask identifying detected source pixels (``True`` = source
        pixel detected, not ``True`` = excluded).
    """
    result = detect_sources_segmentation(
        data,
        n_sigma=n_sigma,
        n_pixels=n_pixels,
        n_levels=n_levels,
        deblend=deblend,
        deblend_contrast=deblend_contrast,
    )

    return result.mask