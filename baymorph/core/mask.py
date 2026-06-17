from scipy.ndimage import binary_dilation
from photutils.segmentation import detect_sources
from photutils.segmentation import detect_threshold

import numpy as np


def connected_source_mask(
    image,
    x,
    y,
    threshold_sigma=2.0,
    n_pixels=5,
    dilation_iterations=0,
    box_size=101,
):
    """
    Extract the connected source containing the clicked position.

    Parameters
    ----------
    image : ndarray
        Input image.

    x, y : float
        Seed position.

    threshold_sigma : float
        Detection threshold in units of local noise.

    n_pixels : int
        Minimum connected pixels.

    dilation_iterations : int
        Optional binary dilation.

    box_size : int
        Local analysis window size.

    Returns
    -------
    mask : ndarray of bool
        Boolean mask following the detected structure.
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
    """
    Create an aggressive mask around a bright star.

    The mask combines:

    - the connected source containing the click position
    - a circular halo mask
    - optional dilation

    Parameters
    ----------
    image : ndarray
        Input image.

    x, y : float
        Star position.

    radius : float
        Radius of the circular halo mask.

    threshold_sigma : float
        Detection threshold.

    n_pixels : int
        Minimum connected pixels.

    dilation_iterations : int
        Additional dilation.

    Returns
    -------
    mask : ndarray of bool
        Boolean mask.
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

    circular_mask = (
        (xx - x) ** 2 +
        (yy - y) ** 2
    ) <= radius**2

    mask = (
        source_mask |
        circular_mask
    )

    if dilation_iterations > 0:

        mask = binary_dilation(
            mask,
            iterations=dilation_iterations,
        )

    return mask