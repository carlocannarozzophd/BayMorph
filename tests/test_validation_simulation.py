import numpy as np
import pytest

from baymorph.validation import (
    GroundTruth,
    build_synthetic_observation,
)


def make_truth(**kwargs):
    values = {
        "model_name": "single_sersic",
        "parameters": {
            "flux": 1000.0,
            "re": 5.0,
            "n": 2.0,
            "q": 0.8,
            "theta": 0.3,
            "x0": 15.5,
            "y0": 15.5,
        },
        "image_shape": (32, 32),
        "sky_level": 10.0,
        "random_seed": 42,
        "psf_metadata": {"fwhm": 2.0},
        "noise_metadata": {
            "poisson_noise": False,
            "read_noise_sigma": None,
        },
    }
    values.update(kwargs)
    return GroundTruth(**values)


def test_build_synthetic_observation_from_ground_truth():
    truth = make_truth()

    observation = build_synthetic_observation(truth)

    assert observation.shape == (32, 32)
    assert observation.has_psf
    assert observation.sky_level == pytest.approx(10.0)

    assert observation.truth == {
        "flux": 1000.0,
        "re": 5.0,
        "n": 2.0,
        "q": 0.8,
        "theta": 0.3,
        "x0": 15.5,
        "y0": 15.5,
        "sky": 10.0,
    }

    assert np.allclose(
        observation.image,
        observation.sky_image,
    )
    assert np.allclose(
        observation.noise_realization,
        0.0,
    )


def test_build_synthetic_observation_is_reproducible():
    truth = make_truth(
        psf_metadata={},
        noise_metadata={
            "poisson_noise": True,
            "read_noise_sigma": None,
        },
    )

    first = build_synthetic_observation(truth)
    second = build_synthetic_observation(truth)

    assert np.array_equal(first.image, second.image)


def test_build_synthetic_observation_allows_no_psf():
    truth = make_truth(psf_metadata={})

    observation = build_synthetic_observation(truth)

    assert not observation.has_psf
    assert observation.psf_kernel is None
    assert np.array_equal(
        observation.intrinsic_image,
        observation.convolved_image,
    )


def test_build_synthetic_observation_rejects_unsupported_model():
    truth = make_truth(model_name="double_sersic")

    with pytest.raises(ValueError, match="currently supports only"):
        build_synthetic_observation(truth)


def test_build_synthetic_observation_requires_all_parameters():
    parameters = dict(make_truth().parameters)
    del parameters["q"]

    truth = make_truth(parameters=parameters)

    with pytest.raises(
        ValueError,
        match="Missing ground-truth Sérsic parameters: q",
    ):
        build_synthetic_observation(truth)