import numpy as np

from baymorph import estimate_background_photutils


def test_photutils_background_flat_image():
    rng = np.random.default_rng(42)

    true_background = 5.0

    image = true_background + rng.normal(
        0,
        1,
        size=(100, 100),
    )

    estimate = estimate_background_photutils(
        image,
        box_size=20,
    )

    assert np.isclose(
        estimate.background,
        true_background,
        atol=0.2,
    )

    assert estimate.background_map is not None
    assert estimate.rms_map is not None


def test_photutils_background_accepts_mask():
    rng = np.random.default_rng(42)

    image = 5.0 + rng.normal(
        0,
        1,
        size=(100, 100),
    )

    mask = np.zeros(
        image.shape,
        dtype=bool,
    )

    mask[40:60, 40:60] = True

    estimate = estimate_background_photutils(
        image,
        box_size=20,
        mask=mask,
        exclude_percentile=50.0,
    )

    assert estimate.background_map is not None