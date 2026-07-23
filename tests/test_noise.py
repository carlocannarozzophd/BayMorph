import numpy as np

from baymorph.simulation import (
    add_gaussian_noise,
    add_poisson_noise,
)



def test_gaussian_noise_shape():
    image = np.ones((20, 20))

    noisy = add_gaussian_noise(
        image,
        sigma=1.0,
        seed=42,
    )

    assert noisy.shape == image.shape


def test_gaussian_noise_reproducible():
    image = np.ones((20, 20))

    noisy_a = add_gaussian_noise(image, sigma=1.0, seed=42)
    noisy_b = add_gaussian_noise(image, sigma=1.0, seed=42)

    assert np.allclose(noisy_a, noisy_b)


def test_poisson_noise_shape():
    image = np.ones((20, 20)) * 10.0

    noisy = add_poisson_noise(
        image,
        seed=42,
    )

    assert noisy.shape == image.shape