import numpy as np
import pytest

from baymorph.core import (
    estimate_noise_map,
    make_constant_noise_map,
)



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
        clip_sigma=1.5,
    )

    assert noise.shape == data.shape
    assert np.all(np.isfinite(noise))
    assert np.allclose(noise, noise[0, 0])


def _estimated_scale(
    values,
):
    """Return the scalar scale estimated from one flat sample."""
    side = int(np.sqrt(values.size))

    return float(
        estimate_noise_map(
            values[: side * side].reshape(
                side,
                side,
            )
        )[0, 0]
    )


def test_estimate_noise_map_recovers_a_known_gaussian_scale():
    """The estimator must recover a known sigma on clean Gaussian pixels.

    A sigma-clipped plain standard deviation underestimates the parent scale
    by about 1.5 percent, because clipping truncates the Gaussian tails. The
    tolerance here is tight enough to fail that estimator.
    """
    rng = np.random.default_rng(
        1234
    )
    sigma = 2.0

    estimates = [
        _estimated_scale(
            rng.normal(
                0.0,
                sigma,
                40000,
            )
        )
        for _ in range(12)
    ]

    relative_bias = float(
        np.mean(estimates)
    ) / sigma - 1.0

    assert abs(relative_bias) < 0.008


def test_estimate_noise_map_resists_unmasked_bright_pixels():
    """Unmasked source pixels must not inflate the estimated scale.

    This is why iterative clipping is retained: a scaled MAD alone is less
    biased on clean pixels but overestimates the scale here.
    """
    rng = np.random.default_rng(
        4321
    )
    sigma = 2.0

    estimates = []

    for _ in range(12):
        values = rng.normal(
            0.0,
            sigma,
            40000,
        )
        contaminated = int(
            0.02 * values.size
        )
        values[:contaminated] += rng.uniform(
            5.0,
            200.0,
            contaminated,
        )

        estimates.append(
            _estimated_scale(
                values
            )
        )

    relative_bias = float(
        np.mean(estimates)
    ) / sigma - 1.0

    assert abs(relative_bias) < 0.010


def test_estimate_noise_map_excludes_masked_pixels():
    """Masked pixels must not contribute to the estimated scale."""
    rng = np.random.default_rng(
        99
    )
    sigma = 2.0

    values = rng.normal(
        0.0,
        sigma,
        (200, 200),
    )
    mask = np.zeros(
        values.shape,
        dtype=bool,
    )

    values[:20] = 5000.0
    mask[:20] = True

    estimate = float(
        estimate_noise_map(
            values,
            mask=mask,
        )[0, 0]
    )

    assert abs(estimate / sigma - 1.0) < 0.02