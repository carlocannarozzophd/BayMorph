"""Source detection and segmentation utilities."""

from dataclasses import dataclass

import numpy as np
from photutils.segmentation import (
    SourceCatalog,
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
) -> DetectionResult:
    """Detect sources in an image using Photutils.

    The input image is first Gaussian-smoothed to improve source detection.
    A detection threshold is then estimated with
    :func:`photutils.segmentation.detect_threshold`, followed by source
    segmentation using :func:`photutils.segmentation.detect_sources`.

    Parameters
    ----------
    data : array-like
        Input image.
    n_sigma : float, default=3.0
        Detection threshold in units of the estimated background noise.
    n_pixels : int, default=5
        Minimum number of connected pixels required for a detection.
    n_levels : int, default=32
        Reserved for future multi-threshold segmentation support. Currently
        unused.

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

    mask = segmentation.data > 0

    try:
        catalog = SourceCatalog(
            data,
            segmentation,
        )
    except Exception:
        catalog = None

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
):
    """Detect sources and return a boolean source mask.

    Parameters
    ----------
    data : array-like
        Input image.
    n_sigma : float, default=3.0
        Detection threshold in units of the estimated background noise.
    n_pixels : int, default=5
        Minimum number of connected pixels required for a detection.
    n_levels : int, default=32
        Reserved for future multi-threshold segmentation support. Currently
        unused.

    Returns
    -------
    ndarray of bool
        Boolean mask identifying detected source pixels.
    """
    result = detect_sources_segmentation(
        data,
        n_sigma=n_sigma,
        n_pixels=n_pixels,
        n_levels=n_levels,
    )

    return result.mask