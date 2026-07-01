"""Mask-construction utilities for astronomical images."""

import numpy as np
from photutils.segmentation import (
    detect_sources,
    detect_threshold,
)
from scipy.ndimage import binary_dilation


def connected_source_mask(
    image,
    x,
    y,
    threshold_sigma=2.0,
    n_pixels=5,
    dilation_iterations=0,
    box_size=101,
):
    """Create a mask for the connected source containing a seed position.

    Parameters
    ----------
    image : array-like
        Input image.
    x : float
        X-coordinate of the seed position, in pixel coordinates.
    y : float
        Y-coordinate of the seed position, in pixel coordinates.
    threshold_sigma : float, default=2.0
        Detection threshold in units of the estimated local noise.
    n_pixels : int, default=5
        Minimum number of connected pixels required for a detection.
    dilation_iterations : int, default=0
        Number of binary-dilation iterations applied to the detected source
        mask. If zero, no dilation is applied.
    box_size : int, default=101
        Size of the local square window used for source detection.

    Returns
    -------
    ndarray of bool
        Boolean mask for the connected detected structure containing the seed
        position. If no valid connected source is found, the returned mask is
        empty.
    """
    ny, nx = image.shape

    half = box_size // 2

    x0 = max(0, int(round(x)) - half)
    x1 = min(nx, int(round(x)) + half + 1)

    y0 = max(0, int(round(y)) - half)
    y1 = min(ny, int(round(y)) + half + 1)

    local = image[y0:y1, x0:x1]

    threshold = detect_threshold(
        local,
        n_sigma=threshold_sigma,
    )

    segmentation = detect_sources(
        local,
        threshold,
        n_pixels=n_pixels,
    )

    if segmentation is None:
        return np.zeros_like(image, dtype=bool)

    x_local = int(round(x)) - x0
    y_local = int(round(y)) - y0

    label = segmentation.data[
        y_local,
        x_local,
    ]

    if label == 0:
        return np.zeros_like(image, dtype=bool)

    local_mask = segmentation.data == label

    if dilation_iterations > 0:
        local_mask = binary_dilation(
            local_mask,
            iterations=dilation_iterations,
        )

    mask = np.zeros_like(
        image,
        dtype=bool,
    )

    mask[
        y0:y1,
        x0:x1,
    ] = local_mask

    return mask


def star_halo_mask(
    image,
    x,
    y,
    radius=50,
    threshold_sigma=2.0,
    n_pixels=5,
    dilation_iterations=10,
):
    """Create an aggressive mask around a bright star.

    The final mask combines the connected source containing the seed position,
    a circular halo mask centered on the same position, and optional binary
    dilation.

    Parameters
    ----------
    image : array-like
        Input image.
    x : float
        X-coordinate of the star center, in pixel coordinates.
    y : float
        Y-coordinate of the star center, in pixel coordinates.
    radius : float, default=50
        Radius of the circular halo mask, in pixels.
    threshold_sigma : float, default=2.0
        Detection threshold in units of the estimated local noise.
    n_pixels : int, default=5
        Minimum number of connected pixels required for a detection.
    dilation_iterations : int, default=10
        Number of binary-dilation iterations applied to the combined mask. If
        zero, no final dilation is applied.

    Returns
    -------
    ndarray of bool
        Boolean mask covering the connected stellar source and its surrounding
        halo.
    """
    source_mask = connected_source_mask(
        image=image,
        x=x,
        y=y,
        threshold_sigma=threshold_sigma,
        n_pixels=n_pixels,
        dilation_iterations=0,
    )

    yy, xx = np.indices(
        image.shape,
    )

    circular_mask = ((xx - x) ** 2 + (yy - y) ** 2) <= radius**2

    mask = source_mask | circular_mask

    if dilation_iterations > 0:
        mask = binary_dilation(
            mask,
            iterations=dilation_iterations,
        )

    return mask