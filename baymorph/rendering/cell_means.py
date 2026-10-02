"""Cell means from point samples, corrected to sixth order.

The fine cells of the effective-PSF rendering of AD-046 away from a
component's centre are small against the scale on which the profile changes,
so their means follow from point samples at their centres. The Sérsic
rendering and the profile components of AD-048 share this rule.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def sixth_order_cell_means(
    samples: FloatArray,
    shape: tuple[int, int],
) -> FloatArray:
    """Return the cell means of ``shape`` from point samples at cell centres.

    ``samples`` holds the profile at the centres of ``shape`` extended by two
    cells on every side. They are filtered along each axis with
    ``[-17, 308, 5178, 308, -17] / 5760``. The filter adds the second- and
    fourth-order terms of the cell mean, so the result is exact for quintic
    profiles and leaves an error of order ``h**6`` in the cell size ``h``:
    accurate where the profile is smooth on the scale of a cell.
    """
    weights = np.array([-17.0, 308.0, 5178.0, 308.0, -17.0]) / 5760.0
    rows = sum(
        weight * samples[offset:offset + shape[0]]
        for offset, weight in enumerate(weights)
    )

    return sum(
        weight * rows[:, offset:offset + shape[1]]
        for offset, weight in enumerate(weights)
    )
