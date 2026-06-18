import numpy as np
import pytest

from baymorph import make_target_cutout


def test_make_target_cutout_returns_expected_shape():
    data = np.zeros((100, 100))

    cutout, bounds = make_target_cutout(
        data,
        target_x=50,
        target_y=50,
        cutout_size=40,
    )

    assert cutout.shape == (40, 40)
    assert bounds == {
        "x_min": 30,
        "x_max": 70,
        "y_min": 30,
        "y_max": 70,
    }


def test_make_target_cutout_raises_if_outside_image():
    data = np.zeros((100, 100))

    with pytest.raises(ValueError, match="outside the image boundaries"):
        make_target_cutout(
            data,
            target_x=10,
            target_y=10,
            cutout_size=40,
        )