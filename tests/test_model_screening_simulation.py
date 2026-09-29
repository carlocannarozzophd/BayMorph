"""Tests for model-screening simulations."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.models import registered_models
from baymorph.validation.model_screening_simulation import (
    screening_parameter_names,
    simulate_model_screening_case,
)


MODEL_PARAMETERS = {
    "single_sersic": {
        "flux": 1000.0,
        "re": 4.0,
        "n": 2.0,
        "q": 0.8,
        "theta": 0.2,
        "x0": 15.5,
        "y0": 15.5,
        "background_level": 10.0,
    },
    "double_sersic": {
        "flux_bulge": 600.0,
        "re_bulge": 2.5,
        "n_bulge": 3.0,
        "q_bulge": 0.9,
        "theta_bulge": 0.1,
        "flux_disk": 800.0,
        "re_disk": 7.0,
        "n_disk": 1.4,
        "q_disk": 0.7,
        "theta_disk": 0.3,
        "x0": 15.5,
        "y0": 15.5,
        "background_level": 10.0,
    },
    "sersic_point_source": {
        "flux_sersic": 1000.0,
        "re": 4.0,
        "n": 2.0,
        "q": 0.8,
        "theta": 0.2,
        "flux_point_source": 200.0,
        "x0": 15.5,
        "y0": 15.5,
        "background_level": 10.0,
    },
    "bulge_disk": {
        "flux_bulge": 600.0,
        "re_bulge": 2.5,
        "n_bulge": 3.0,
        "q_bulge": 0.9,
        "theta_bulge": 0.1,
        "flux_disk": 800.0,
        "re_disk": 7.0,
        "q_disk": 0.7,
        "theta_disk": 0.3,
        "x0": 15.5,
        "y0": 15.5,
        "background_level": 10.0,
    },
}


@pytest.mark.parametrize(
    "model_name",
    registered_models(),
)
def test_simulator_supports_every_registered_model(
    model_name: str,
) -> None:
    """Every production morphology must generate a labelled image."""
    case = simulate_model_screening_case(
        model_name=model_name,
        parameters=MODEL_PARAMETERS[
            model_name
        ],
        image_shape=(
            32,
            32,
        ),
        psf_fwhm=2.0,
        noise_sigma=0.0,
        random_seed=17,
    )

    assert case.model_name == model_name
    assert case.shape == (
        32,
        32,
    )
    assert set(
        case.parameters
    ) == set(
        screening_parameter_names(
            model_name
        )
    )

    assert np.all(
        np.isfinite(
            case.image
        )
    )
    assert np.array_equal(
        case.image,
        case.noiseless_image,
    )

    assert not case.image.flags.writeable
    assert not case.noiseless_image.flags.writeable


def test_simulator_is_reproducible() -> None:
    """The random seed must reproduce the complete observed image."""
    options = {
        "model_name": "single_sersic",
        "parameters": MODEL_PARAMETERS[
            "single_sersic"
        ],
        "image_shape": (
            32,
            32,
        ),
        "psf_fwhm": 2.0,
        "noise_sigma": 3.0,
        "random_seed": 91,
    }

    first = simulate_model_screening_case(
        **options
    )
    second = simulate_model_screening_case(
        **options
    )

    assert np.array_equal(
        first.image,
        second.image,
    )
    assert np.array_equal(
        first.noiseless_image,
        second.noiseless_image,
    )


def test_simulator_adds_declared_gaussian_noise() -> None:
    """The observed image must differ from its noiseless expectation."""
    case = simulate_model_screening_case(
        model_name="single_sersic",
        parameters=MODEL_PARAMETERS[
            "single_sersic"
        ],
        image_shape=(
            32,
            32,
        ),
        psf_fwhm=2.0,
        noise_sigma=3.0,
        random_seed=12,
    )

    noise = (
        case.image
        - case.noiseless_image
    )

    assert not np.array_equal(
        case.image,
        case.noiseless_image,
    )
    assert np.std(
        noise
    ) == pytest.approx(
        3.0,
        rel=0.1,
    )


def test_simulator_requires_complete_parameters() -> None:
    """A model cannot be simulated from an incomplete truth."""
    parameters = dict(
        MODEL_PARAMETERS[
            "single_sersic"
        ]
    )
    del parameters[
        "n"
    ]

    with pytest.raises(
        ValueError,
        match="complete model contract",
    ):
        simulate_model_screening_case(
            model_name="single_sersic",
            parameters=parameters,
            image_shape=(
                32,
                32,
            ),
            psf_fwhm=2.0,
            noise_sigma=1.0,
            random_seed=0,
        )
