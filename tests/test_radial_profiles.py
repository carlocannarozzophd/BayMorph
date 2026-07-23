import numpy as np
import pytest

from baymorph.analysis import radial_profile_from_image



def test_radial_profile_from_image_returns_annular_medians():
    image = np.ones((5, 5))
    image[2, 2] = 10.0

    radii = np.array([0.0, 1.0, 2.0])

    profile = radial_profile_from_image(
        image,
        center_x=2.0,
        center_y=2.0,
        radii=radii,
    )

    assert profile.shape == (2,)
    assert profile[0] == pytest.approx(10.0)
    assert profile[1] == pytest.approx(1.0)


def test_radial_profile_from_image_applies_mask():
    image = np.ones((5, 5))
    image[2, 2] = 10.0

    mask = np.zeros_like(image, dtype=bool)
    mask[2, 2] = True

    radii = np.array([0.0, 1.0])

    profile = radial_profile_from_image(
        image,
        center_x=2.0,
        center_y=2.0,
        radii=radii,
        mask=mask,
    )

    assert np.isnan(profile[0])