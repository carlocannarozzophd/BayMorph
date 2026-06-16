"""Target position estimation."""

import numpy as np


def estimate_target_position(
    data,
):
    """
    Estimate target position from the brightest pixel.
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