import numpy as np
import pytest

from baymorph import estimate_noise_map, make_constant_noise_map


def test_make_constant_noise_map_shape():
    noise = make_constant_noise_map(
        shape=(20, 30),
        sigma=2.0,
    )

    assert noise.shape == (20, 30)
    assert np.allclose(noise, 2.0)


def test_make_constant_noise_map_rejects_non_positive_sigma():
    with pytest.raises(ValueError):
        make_constant_noise_map(
            shape=(20, 20),
            sigma=0.0,
        )


def test_estimate_noise_map_shape():
    data = np.ones((10, 15))

    noise = estimate_noise_map(
        data,
        sigma=1.5,
    )

    assert noise.shape == data.shape
    assert np.allclose(noise, 1.5)