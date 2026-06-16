"""Source detection and segmentation utilities."""

from dataclasses import dataclass

import numpy as np
from photutils.segmentation import SourceCatalog, detect_sources


@dataclass
class DetectionResult:
    """Container for source detection products."""

    segmentation: object
    mask: np.ndarray
    catalog: object | None = None


def detect_sources_segmentation(
    data,
    n_sigma: float = 3.0,
    n_pixels: int = 5,
    n_levels: int = 32,
) -> DetectionResult:
    """Detect sources and return segmentation, mask, and catalog."""
    data = np.asarray(
        data,
        dtype=float,
    )

    threshold = (
        np.nanmedian(data)
        + n_sigma * np.nanstd(data)
    )

    segmentation = detect_sources(
        data,
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
    """Detect sources and return a boolean mask."""
    result = detect_sources_segmentation(
        data,
        n_sigma=n_sigma,
        n_pixels=n_pixels,
        n_levels=n_levels,
    )

    return result.mask