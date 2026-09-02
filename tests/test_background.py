import numpy as np

from baymorph.core import (
    estimate_background_constant,
    subtract_background,
)



def test_estimate_background_constant_constant_image():
    data = np.ones((20, 20)) * 5.0

    estimate = estimate_background_constant(data)

    assert estimate.background == 5.0
    assert estimate.sigma == 0.0


def test_subtract_background():
    data = np.ones((20, 20)) * 5.0

    corrected = subtract_background(data, background=5.0)

    assert np.allclose(corrected, 0.0)


def test_estimate_background_constant_with_noise():
    rng = np.random.default_rng(42)

    data = rng.normal(
        loc=10.0,
        scale=2.0,
        size=(200, 200),
    )

    estimate = estimate_background_constant(data)

    assert np.isclose(estimate.background, 10.0, atol=0.1)
    assert np.isclose(estimate.sigma, 2.0, atol=0.1)
