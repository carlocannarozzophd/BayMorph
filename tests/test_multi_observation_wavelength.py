"""Tests for physical-prior-preserving wavelength coupling."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

from baymorph.fitting.multi_observation import (
    MultiObservationParameterLayout,
    MultiObservationParameterSpec,
)
from baymorph.fitting.priors import (
    BetaPrior,
    FixedPrior,
    UniformPrior,
)


def _coordinates(
    observations,
):
    wavelengths = np.geomspace(
        4000.0,
        8000.0,
        len(observations),
    )

    logarithmic = np.log(
        wavelengths
    )

    centered = (
        logarithmic
        - np.mean(
            logarithmic
        )
    )

    return {
        observation: float(value)
        for observation, value in zip(
            observations,
            centered,
            strict=True,
        )
    }


def _specification(
    name: str = "future_parameter",
):
    return MultiObservationParameterSpec.wavelength(
        name,
        prior=UniformPrior(
            lower=2.0,
            upper=12.0,
        ),
    )


@pytest.mark.parametrize(
    "observations",
    [
        ["g", "r"],
        ["u", "g", "r", "i", "z"],
        [
            f"obs_{index}"
            for index in range(10)
        ],
    ],
)
def test_wavelength_layout_is_n_observation_native(
    observations,
):
    layout = MultiObservationParameterLayout(
        observations=observations,
        specifications=[
            _specification(),
        ],
        wavelength_coordinates=_coordinates(
            observations
        ),
    )

    assert layout.ndim == (
        1 + len(observations)
    )

    assert layout.parameter_set.names[0] == (
        "future_parameter__rstar"
    )

    assert layout.parameter_set.names[1:] == [
        f"future_parameter__z__{observation}"
        for observation in observations
    ]


def test_wavelength_reference_prior_is_beta_2_2():
    specification = _specification()

    assert specification.correlation_prior == BetaPrior(
        alpha=2.0,
        beta=2.0,
    )


def test_zero_reference_correlation_routes_independent_quantiles():
    observations = [
        "blue",
        "red",
    ]

    prior = UniformPrior(
        lower=2.0,
        upper=12.0,
    )

    layout = MultiObservationParameterLayout(
        observations=observations,
        specifications=[
            MultiObservationParameterSpec.wavelength(
                "scale",
                prior=prior,
            ),
        ],
        wavelength_coordinates={
            "blue": -0.5 * np.log(2.0),
            "red": 0.5 * np.log(2.0),
        },
    )

    routed = layout.route(
        {
            "scale__rstar": 0.0,
            "scale__z__blue": -1.0,
            "scale__z__red": 1.0,
        }
    )

    assert routed["blue"]["scale"] == pytest.approx(
        prior.transform(
            float(
                ndtr(-1.0)
            )
        )
    )

    assert routed["red"]["scale"] == pytest.approx(
        prior.transform(
            float(
                ndtr(1.0)
            )
        )
    )


def test_unit_reference_correlation_produces_exact_sharing():
    observations = [
        "a",
        "b",
        "c",
        "d",
        "e",
    ]

    layout = MultiObservationParameterLayout(
        observations=observations,
        specifications=[
            _specification(),
        ],
        wavelength_coordinates=_coordinates(
            observations
        ),
    )

    values = {
        "future_parameter__rstar": 1.0,
    }

    for index, observation in enumerate(
        observations
    ):
        values[
            f"future_parameter__z__{observation}"
        ] = float(
            index - 2
        )

    routed = layout.route(
        values
    )

    physical = np.asarray(
        [
            routed[
                observation
            ][
                "future_parameter"
            ]
            for observation in observations
        ]
    )

    assert physical == pytest.approx(
        np.full(
            len(observations),
            physical[0],
        )
    )


def test_wavelength_preserves_physical_prior_bounds():
    observations = [
        "a",
        "b",
        "c",
        "d",
        "e",
    ]

    layout = MultiObservationParameterLayout(
        observations=observations,
        specifications=[
            _specification(
                name="bounded_parameter"
            ),
        ],
        wavelength_coordinates=_coordinates(
            observations
        ),
    )

    values = {
        "bounded_parameter__rstar": 0.7,
    }

    latents = [
        -10.0,
        -2.0,
        0.0,
        2.0,
        10.0,
    ]

    for observation, latent in zip(
        observations,
        latents,
        strict=True,
    ):
        values[
            f"bounded_parameter__z__{observation}"
        ] = latent

    routed = layout.route(
        values
    )

    for observation in observations:
        assert (
            2.0
            <= routed[
                observation
            ][
                "bounded_parameter"
            ]
            <= 12.0
        )


def test_wavelength_is_parameter_name_agnostic():
    observations = [
        "first",
        "second",
        "third",
    ]

    layout = MultiObservationParameterLayout(
        observations=observations,
        specifications=[
            _specification(
                name="future_component_shape"
            ),
        ],
        wavelength_coordinates=_coordinates(
            observations
        ),
    )

    routed = layout.route(
        {
            "future_component_shape__rstar": 0.5,
            "future_component_shape__z__first": -0.4,
            "future_component_shape__z__second": 0.2,
            "future_component_shape__z__third": 1.1,
        }
    )

    for observation in observations:
        assert (
            2.0
            <= routed[
                observation
            ][
                "future_component_shape"
            ]
            <= 12.0
        )


def test_wavelength_requires_coordinates():
    with pytest.raises(
        ValueError,
        match="requires wavelength coordinates",
    ):
        MultiObservationParameterLayout(
            observations=[
                "a",
                "b",
            ],
            specifications=[
                _specification(),
            ],
        )


def test_wavelength_requires_exact_observation_coordinates():
    with pytest.raises(
        ValueError,
        match="exactly the configured observations",
    ):
        MultiObservationParameterLayout(
            observations=[
                "a",
                "b",
            ],
            specifications=[
                _specification(),
            ],
            wavelength_coordinates={
                "a": 0.0,
            },
        )


def test_wavelength_rejects_fixed_physical_prior():
    with pytest.raises(
        ValueError,
        match="fixed physical prior",
    ):
        MultiObservationParameterLayout(
            observations=[
                "a",
                "b",
            ],
            specifications=[
                MultiObservationParameterSpec.wavelength(
                    "shape",
                    prior=FixedPrior(
                        value=3.0
                    ),
                ),
            ],
            wavelength_coordinates={
                "a": -0.1,
                "b": 0.1,
            },
        )
