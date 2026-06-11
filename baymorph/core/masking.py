"""Source masking utilities."""

import numpy as np

from photutils.segmentation import (
    detect_sources,
)
from photutils.segmentation import (
    detect_threshold,
)


def detect_sources_mask(
    data: np.ndarray,
    n_sigma: float = 2.0,
    n_pixels: int = 5,
) -> np.ndarray:
    """
    Detect sources and return a boolean mask.

    True = masked pixel
    False = usable pixel
    """

    threshold = detect_threshold(
        data,
        n_sigma=n_sigma,
    )

    segment_map = detect_sources(
        data,
        threshold,
        n_pixels=n_pixels,
    )

    if segment_map is None:
        return np.zeros(
            data.shape,
            dtype=bool,
        )

    return segment_map.data > 0