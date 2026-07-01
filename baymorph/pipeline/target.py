"""Target position estimation."""

import numpy as np


def estimate_target_position(
    data,
):
    """Estimate the target position from the brightest image pixel.

    Parameters
    ----------
    data : array-like
        Input image.

    Returns
    -------
    x : float
        Estimated x-coordinate of the target, in pixels.
    y : float
        Estimated y-coordinate of the target, in pixels.

    Notes
    -----
    The current implementation identifies the target as the brightest pixel in
    the image. This provides a simple and robust initialization for many
    datasets and may be replaced by a more sophisticated centroiding algorithm
    in future versions.
    """
    data = np.asarray(
        data,
        dtype=float,
    )

    y, x = np.unravel_index(
        np.nanargmax(data),
        data.shape,
    )

    return float(x), float(y)