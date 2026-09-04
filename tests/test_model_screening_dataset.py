"""Tests for balanced model-screening simulation datasets."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from baymorph.models import registered_models
from baymorph.validation.model_screening_dataset import (
    generate_model_screening_cases,
)


def fixed_prior(
    value: float,
) -> dict[str, float | str]:
    """Return a fixed BayMorph prior configuration."""
    return {
        "type": "fixed",
        "value": value,
    }


def fixed_model_priors() -> dict:
    """Return complete non-degenerate priors for every model."""
    parameter_values = {
        "single_sersic": {
            "flux": 1000.0,
            "re": 4.0,
            "n": 2.0,
            "q": 0.8,
            "theta": 0.2,
            "x0": 15.5,
            "y0": 15.5,
            "sky": 10.0,
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
            "sky": 10.0,
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
            "sky": 10.0,
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
            "sky": 10.0,
        },
    }

    return {
        model_name: {
            parameter_name: fixed_prior(
                value
            )
            for parameter_name, value in values.items()
        }
        for model_name, values in parameter_values.items()
    }


def generation_options() -> dict:
    """Return a reproducible screening-dataset configuration."""
    return {
        "model_priors": fixed_model_priors(),
        "n_per_model": 2,
        "image_shape": (
            32,
            32,
        ),
        "psf_fwhm_prior": {
            "type": "uniform",
            "lower": 1.5,
            "upper": 3.0,
        },
        "noise_sigma_prior": {
            "type": "uniform",
            "lower": 1.0,
            "upper": 3.0,
        },
        "random_state": 19,
    }


def test_generator_is_balanced_across_registered_models() -> None:
    """Every model must contribute the same number of cases."""
    cases = list(
        generate_model_screening_cases(
            **generation_options()
        )
    )

    counts = Counter(
        case.model_name
        for case in cases
    )

    assert len(
        cases
    ) == 2 * len(
        registered_models()
    )

    assert counts == {
        model_name: 2
        for model_name in registered_models()
    }

    assert [
        case.index
        for case in cases
    ] == list(
        range(
            len(
                cases
            )
        )
    )


def test_generator_is_fully_reproducible() -> None:
    """One root seed must reproduce parameters, nuisances, and images."""
    first = list(
        generate_model_screening_cases(
            **generation_options()
        )
    )
    second = list(
        generate_model_screening_cases(
            **generation_options()
        )
    )

    for left, right in zip(
        first,
        second,
        strict=True,
    ):
        assert left.model_name == right.model_name
        assert left.parameters == right.parameters
        assert left.psf_fwhm == right.psf_fwhm
        assert left.noise_sigma == right.noise_sigma
        assert left.random_seed == right.random_seed

        assert np.array_equal(
            left.image,
            right.image,
        )


def test_generator_uses_shared_nuisance_prior_support() -> None:
    """All classes must draw PSF and noise from common supports."""
    cases = list(
        generate_model_screening_cases(
            **generation_options()
        )
    )

    for case in cases:
        assert 1.5 <= case.psf_fwhm <= 3.0
        assert 1.0 <= case.noise_sigma <= 3.0


def test_generator_requires_every_registered_model() -> None:
    """A training dataset cannot silently omit one candidate model."""
    model_priors = fixed_model_priors()

    del model_priors[
        "double_sersic"
    ]

    options = generation_options()
    options[
        "model_priors"
    ] = model_priors

    with pytest.raises(
        ValueError,
        match="cover every registered model",
    ):
        list(
            generate_model_screening_cases(
                **options
            )
        )


def test_generator_requires_complete_parameter_priors() -> None:
    """Each model requires its complete parameter contract."""
    model_priors = fixed_model_priors()

    del model_priors[
        "single_sersic"
    ][
        "n"
    ]

    options = generation_options()
    options[
        "model_priors"
    ] = model_priors

    with pytest.raises(
        ValueError,
        match="complete parameter contract",
    ):
        list(
            generate_model_screening_cases(
                **options
            )
        )


def test_generator_supports_relative_centroid_priors() -> None:
    """Centroid draws may be relative to the common target seed."""
    model_priors = fixed_model_priors()

    for priors in model_priors.values():
        priors[
            "x0"
        ] = {
            "type": "uniform",
            "lower": -0.5,
            "upper": 0.5,
            "relative_to": "target_x",
        }
        priors[
            "y0"
        ] = {
            "type": "uniform",
            "lower": -0.5,
            "upper": 0.5,
            "relative_to": "target_y",
        }

    options = generation_options()
    options[
        "model_priors"
    ] = model_priors

    cases = list(
        generate_model_screening_cases(
            **options
        )
    )

    for case in cases:
        assert 15.0 <= case.parameters["x0"] <= 16.0
        assert 15.0 <= case.parameters["y0"] <= 16.0
