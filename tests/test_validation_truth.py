from dataclasses import FrozenInstanceError

import pytest

from baymorph.validation import GroundTruth


def test_ground_truth_normalizes_and_exposes_simulation_truth():
    truth = GroundTruth(
        model_name=" single_sersic ",
        parameters={"flux": 1000, "re": 8.5, "n": 2},
        image_shape=[64, 80],
        pixel_scale=0.2,
        sky_level=12,
        random_seed=42,
        psf_metadata={"kind": "gaussian", "fwhm": 2.1},
        noise_metadata={"kind": "gaussian", "sigma": 3.0},
        metadata={"campaign": "smoke-test"},
    )

    assert truth.model_name == "single_sersic"
    assert truth.parameter_names == ("flux", "re", "n")
    assert truth.parameter("re") == 8.5
    assert truth.image_shape == (64, 80)
    assert truth.pixel_scale == 0.2
    assert truth.sky_level == 12.0
    assert truth.random_seed == 42
    assert truth.psf_metadata["kind"] == "gaussian"
    assert truth.noise_metadata["sigma"] == 3.0
    assert truth.metadata["campaign"] == "smoke-test"


def test_ground_truth_is_immutable():
    parameters = {"re": 8.5}

    truth = GroundTruth(
        model_name="single_sersic",
        parameters=parameters,
        image_shape=(64, 64),
    )

    parameters["re"] = 10.0

    assert truth.parameter("re") == 8.5

    with pytest.raises(TypeError):
        truth.parameters["re"] = 10.0

    with pytest.raises(FrozenInstanceError):
        truth.model_name = "double_sersic"


def test_ground_truth_rejects_invalid_scientific_inputs():
    with pytest.raises(ValueError, match="Model name"):
        GroundTruth(
            model_name="",
            parameters={"re": 8.5},
            image_shape=(64, 64),
        )

    with pytest.raises(ValueError, match="at least one parameter"):
        GroundTruth(
            model_name="sersic",
            parameters={},
            image_shape=(64, 64),
        )

    with pytest.raises(ValueError, match="finite"):
        GroundTruth(
            model_name="sersic",
            parameters={"re": float("nan")},
            image_shape=(64, 64),
        )

    with pytest.raises(ValueError, match="positive"):
        GroundTruth(
            model_name="sersic",
            parameters={"re": 8.5},
            image_shape=(0, 64),
        )

    with pytest.raises(ValueError, match="Pixel scale"):
        GroundTruth(
            model_name="sersic",
            parameters={"re": 8.5},
            image_shape=(64, 64),
            pixel_scale=0.0,
        )

    with pytest.raises(ValueError, match="Random seed"):
        GroundTruth(
            model_name="sersic",
            parameters={"re": 8.5},
            image_shape=(64, 64),
            random_seed=-1,
        )