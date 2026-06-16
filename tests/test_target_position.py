import numpy as np

from baymorph import (
    estimate_target_position,
)


def test_estimate_target_position():

    image = np.zeros(
        (
            20,
            20,
        )
    )

    image[
        8,
        12,
    ] = 100

    x0, y0 = (
        estimate_target_position(
            image,
        )
    )

    assert x0 == 12

    assert y0 == 8