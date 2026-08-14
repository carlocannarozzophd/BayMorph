"""Tests for hierarchical multi-observation parameter coupling."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.multi_observation import (
    MultiObservationParameterLayout,
    MultiObservationParameterSpec,
)
from baymorph.fitting.parameters import Parameter


def _hierarchical_re_spec() -> MultiObservationParameterSpec:
    return MultiObservationParameterSpec.hierarchical_log_normal(
        "re",
        location=Parameter(
            "re__mu_log",
            lower=np.log(2.0),
            upper=np.log(8.0),
        ),
        scale=Parameter(
            "re__tau_log",
            lower=0.0,
            upper=1.0,
        ),
    )


def test_hierarchical_log_normal_dimension_and_names():
    layout = MultiObservationParameterLayout(
        observations=["g", "r"],
        specifications=[
            _hierarchical_re_spec(),
        ],
    )

    assert layout.ndim == 4
    assert layout.parameter_set.names == [
        "re__mu_log",
        "re__tau_log",
        "re__z__g",
        "re__z__r",
    ]


def test_hierarchical_log_normal_routes_derived_physical_values():
    layout = MultiObservationParameterLayout(
        observations=["g", "r"],
        specifications=[
            _hierarchical_re_spec(),
        ],
    )

    sampled = {
        "re__mu_log": np.log(5.0),
        "re__tau_log": 0.2,
        "re__z__g": -1.0,
        "re__z__r": 1.0,
    }

    routed = layout.route(sampled)

    assert routed["g"]["re"] == pytest.approx(
        5.0 * np.exp(-0.2)
    )
    assert routed["r"]["re"] == pytest.approx(
        5.0 * np.exp(0.2)
    )

    assert routed["g"]["re"] > 0.0
    assert routed["r"]["re"] > 0.0


def test_hierarchical_log_normal_zero_scale_matches_exact_sharing():
    layout = MultiObservationParameterLayout(
        observations=["g", "r", "i"],
        specifications=[
            _hierarchical_re_spec(),
        ],
    )

    sampled = {
        "re__mu_log": np.log(4.0),
        "re__tau_log": 0.0,
        "re__z__g": -2.0,
        "re__z__r": 0.5,
        "re__z__i": 3.0,
    }

    routed = layout.route(sampled)

    for observation in layout.observations:
        assert routed[observation]["re"] == pytest.approx(
            4.0
        )


def test_hierarchical_log_normal_rejects_negative_scale():
    layout = MultiObservationParameterLayout(
        observations=["g", "r"],
        specifications=[
            _hierarchical_re_spec(),
        ],
    )

    sampled = {
        "re__mu_log": np.log(5.0),
        "re__tau_log": -0.1,
        "re__z__g": 0.0,
        "re__z__r": 0.0,
    }

    with pytest.raises(
        ValueError,
        match="scale.*non-negative",
    ):
        layout.route(sampled)


def test_hierarchical_log_normal_coexists_with_existing_couplings():
    layout = MultiObservationParameterLayout(
        observations=["g", "r"],
        specifications=[
            MultiObservationParameterSpec.shared_parameter(
                Parameter(
                    "n",
                    lower=0.5,
                    upper=6.0,
                )
            ),
            MultiObservationParameterSpec.independent(
                "flux",
                {
                    "g": Parameter(
                        "flux",
                        lower=100.0,
                        upper=1000.0,
                    ),
                    "r": Parameter(
                        "flux",
                        lower=100.0,
                        upper=1000.0,
                    ),
                },
            ),
            _hierarchical_re_spec(),
        ],
    )

    sampled = {
        "n": 2.5,
        "flux__g": 400.0,
        "flux__r": 600.0,
        "re__mu_log": np.log(3.0),
        "re__tau_log": 0.1,
        "re__z__g": -1.0,
        "re__z__r": 1.0,
    }

    routed = layout.route(sampled)

    assert routed["g"]["n"] == pytest.approx(2.5)
    assert routed["r"]["n"] == pytest.approx(2.5)

    assert routed["g"]["flux"] == pytest.approx(400.0)
    assert routed["r"]["flux"] == pytest.approx(600.0)

    assert routed["g"]["re"] == pytest.approx(
        3.0 * np.exp(-0.1)
    )
    assert routed["r"]["re"] == pytest.approx(
        3.0 * np.exp(0.1)
    )


def test_hierarchical_latent_prior_transform_is_standard_normal():
    layout = MultiObservationParameterLayout(
        observations=["g", "r"],
        specifications=[
            _hierarchical_re_spec(),
        ],
    )

    transformed = layout.parameter_set.transform(
        np.array(
            [
                0.5,
                0.25,
                0.5,
                0.5,
            ]
        )
    )

    assert transformed[2] == pytest.approx(0.0)
    assert transformed[3] == pytest.approx(0.0)
