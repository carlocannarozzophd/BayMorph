"""Tests for physical-prior-preserving hierarchical coupling."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.multi_observation import (
    MultiObservationParameterLayout,
    MultiObservationParameterSpec,
)
from baymorph.fitting.priors import (
    FixedPrior,
    UniformPrior,
)


def _hierarchical_spec(
    name: str = "n",
) -> MultiObservationParameterSpec:
    return MultiObservationParameterSpec.hierarchical(
        name,
        prior=UniformPrior(
            lower=0.1,
            upper=6.0,
        ),
    )


@pytest.mark.parametrize(
    "observations",
    [
        ["g", "r"],
        ["u", "g", "r", "i", "z"],
        [
            "obs_0",
            "obs_1",
            "obs_2",
            "obs_3",
            "obs_4",
            "obs_5",
            "obs_6",
            "obs_7",
            "obs_8",
            "obs_9",
        ],
    ],
)
def test_hierarchical_layout_is_n_observation_native(
    observations,
):
    layout = MultiObservationParameterLayout(
        observations=observations,
        specifications=[
            _hierarchical_spec(),
        ],
    )

    assert layout.ndim == (
        1 + len(observations)
    )

    assert layout.parameter_set.names[0] == (
        "n__rho"
    )

    assert layout.parameter_set.names[1:] == [
        f"n__z__{observation}"
        for observation in observations
    ]


def test_zero_correlation_routes_independent_quantiles():
    layout = MultiObservationParameterLayout(
        observations=["g", "r"],
        specifications=[
            _hierarchical_spec(),
        ],
    )

    routed = layout.route(
        {
            "n__rho": 0.0,
            "n__z__g": -1.0,
            "n__z__r": 1.0,
        }
    )

    assert 0.1 < routed["g"]["n"] < 6.0
    assert 0.1 < routed["r"]["n"] < 6.0
    assert routed["g"]["n"] != pytest.approx(
        routed["r"]["n"]
    )


def test_unit_correlation_produces_common_physical_value_for_n_observations():
    observations = [
        "u",
        "g",
        "r",
        "i",
        "z",
    ]

    layout = MultiObservationParameterLayout(
        observations=observations,
        specifications=[
            _hierarchical_spec(),
        ],
    )

    values = {
        "n__rho": 1.0,
        "n__z__u": -2.0,
        "n__z__g": -1.0,
        "n__z__r": 0.0,
        "n__z__i": 1.0,
        "n__z__z": 2.0,
    }

    routed = layout.route(
        values
    )

    physical = np.asarray(
        [
            routed[observation]["n"]
            for observation in observations
        ]
    )

    assert physical == pytest.approx(
        np.full(
            len(observations),
            physical[0],
        )
    )


def test_physical_prior_bounds_are_preserved():
    layout = MultiObservationParameterLayout(
        observations=[
            "u",
            "g",
            "r",
            "i",
            "z",
        ],
        specifications=[
            _hierarchical_spec(),
        ],
    )

    routed = layout.route(
        {
            "n__rho": 0.75,
            "n__z__u": -10.0,
            "n__z__g": -2.0,
            "n__z__r": 0.0,
            "n__z__i": 2.0,
            "n__z__z": 10.0,
        }
    )

    for observation in layout.observations:
        assert (
            0.1
            <= routed[observation]["n"]
            <= 6.0
        )


def test_hierarchical_can_use_bounded_effective_radius_prior():
    layout = MultiObservationParameterLayout(
        observations=["g", "r", "i"],
        specifications=[
            MultiObservationParameterSpec.hierarchical(
                "re",
                prior=UniformPrior(
                    lower=0.5,
                    upper=30.0,
                ),
            ),
        ],
    )

    routed = layout.route(
        {
            "re__rho": 0.6,
            "re__z__g": -1.0,
            "re__z__r": 0.0,
            "re__z__i": 1.0,
        }
    )

    for observation in layout.observations:
        assert (
            0.5
            <= routed[observation]["re"]
            <= 30.0
        )


def test_hierarchical_is_parameter_name_agnostic():
    layout = MultiObservationParameterLayout(
        observations=["a", "b"],
        specifications=[
            MultiObservationParameterSpec.hierarchical(
                "future_component_scale",
                prior=UniformPrior(
                    lower=2.0,
                    upper=12.0,
                ),
            ),
        ],
    )

    routed = layout.route(
        {
            "future_component_scale__rho": 0.5,
            "future_component_scale__z__a": -0.3,
            "future_component_scale__z__b": 0.8,
        }
    )

    assert (
        2.0
        <= routed["a"]["future_component_scale"]
        <= 12.0
    )
    assert (
        2.0
        <= routed["b"]["future_component_scale"]
        <= 12.0
    )


def test_hierarchical_rejects_fixed_physical_prior():
    with pytest.raises(
        ValueError,
        match="fixed physical prior",
    ):
        MultiObservationParameterLayout(
            observations=["g", "r"],
            specifications=[
                MultiObservationParameterSpec.hierarchical(
                    "n",
                    prior=FixedPrior(
                        value=2.0
                    ),
                ),
            ],
        )


def test_hierarchical_requires_at_least_two_observations():
    with pytest.raises(
        ValueError,
        match="at least two observations",
    ):
        MultiObservationParameterLayout(
            observations=["g"],
            specifications=[
                _hierarchical_spec(),
            ],
        )


def test_hierarchical_reference_dependence_prior_is_beta_2_2():
    from baymorph.fitting.priors import BetaPrior

    specification = (
        MultiObservationParameterSpec.hierarchical(
            "future_parameter",
            prior=UniformPrior(
                lower=1.0,
                upper=5.0,
            ),
        )
    )

    assert specification.correlation_prior == BetaPrior(
        alpha=2.0,
        beta=2.0,
    )
