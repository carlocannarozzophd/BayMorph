"""Tests for hierarchical multi-observation parameter coupling."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.hierarchical import (
    ExpHierarchicalTransform,
    IdentityHierarchicalTransform,
)
from baymorph.fitting.multi_observation import (
    MultiObservationParameterLayout,
    MultiObservationParameterSpec,
)
from baymorph.fitting.parameters import Parameter


def _positive_spec(
    name: str = "re",
) -> MultiObservationParameterSpec:
    return MultiObservationParameterSpec.hierarchical(
        name,
        location=Parameter(
            f"{name}__eta_mu",
            lower=np.log(2.0),
            upper=np.log(8.0),
        ),
        scale=Parameter(
            f"{name}__eta_tau",
            lower=0.0,
            upper=1.0,
        ),
        transform=ExpHierarchicalTransform(),
    )


def test_hierarchical_dimension_and_names():
    layout = MultiObservationParameterLayout(
        observations=["g", "r"],
        specifications=[
            _positive_spec(),
        ],
    )

    assert layout.ndim == 4
    assert layout.parameter_set.names == [
        "re__eta_mu",
        "re__eta_tau",
        "re__z__g",
        "re__z__r",
    ]


def test_hierarchical_exp_routes_positive_physical_values():
    layout = MultiObservationParameterLayout(
        observations=["g", "r"],
        specifications=[
            _positive_spec(),
        ],
    )

    routed = layout.route(
        {
            "re__eta_mu": np.log(5.0),
            "re__eta_tau": 0.2,
            "re__z__g": -1.0,
            "re__z__r": 1.0,
        }
    )

    assert routed["g"]["re"] == pytest.approx(
        5.0 * np.exp(-0.2)
    )
    assert routed["r"]["re"] == pytest.approx(
        5.0 * np.exp(0.2)
    )
    assert routed["g"]["re"] > 0.0
    assert routed["r"]["re"] > 0.0


def test_hierarchical_identity_routes_real_values():
    spec = MultiObservationParameterSpec.hierarchical(
        "background",
        location=Parameter(
            "background__eta_mu",
            lower=-10.0,
            upper=10.0,
        ),
        scale=Parameter(
            "background__eta_tau",
            lower=0.0,
            upper=5.0,
        ),
        transform=IdentityHierarchicalTransform(),
    )

    layout = MultiObservationParameterLayout(
        observations=["g", "r"],
        specifications=[
            spec,
        ],
    )

    routed = layout.route(
        {
            "background__eta_mu": 2.0,
            "background__eta_tau": 0.5,
            "background__z__g": -1.0,
            "background__z__r": 1.0,
        }
    )

    assert routed["g"]["background"] == pytest.approx(
        1.5
    )
    assert routed["r"]["background"] == pytest.approx(
        2.5
    )


def test_zero_scale_collapses_observation_values():
    layout = MultiObservationParameterLayout(
        observations=["g", "r", "i"],
        specifications=[
            _positive_spec(),
        ],
    )

    routed = layout.route(
        {
            "re__eta_mu": np.log(4.0),
            "re__eta_tau": 0.0,
            "re__z__g": -2.0,
            "re__z__r": 0.5,
            "re__z__i": 3.0,
        }
    )

    for observation in layout.observations:
        assert routed[observation]["re"] == pytest.approx(
            4.0
        )


def test_hierarchical_rejects_negative_scale():
    layout = MultiObservationParameterLayout(
        observations=["g", "r"],
        specifications=[
            _positive_spec(),
        ],
    )

    with pytest.raises(
        ValueError,
        match="scale.*non-negative",
    ):
        layout.route(
            {
                "re__eta_mu": np.log(5.0),
                "re__eta_tau": -0.1,
                "re__z__g": 0.0,
                "re__z__r": 0.0,
            }
        )


def test_hierarchical_coexists_with_existing_couplings():
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
            _positive_spec(),
        ],
    )

    routed = layout.route(
        {
            "n": 2.5,
            "flux__g": 400.0,
            "flux__r": 600.0,
            "re__eta_mu": np.log(3.0),
            "re__eta_tau": 0.1,
            "re__z__g": -1.0,
            "re__z__r": 1.0,
        }
    )

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


def test_hierarchical_latent_prior_is_standard_normal():
    layout = MultiObservationParameterLayout(
        observations=["g", "r"],
        specifications=[
            _positive_spec(),
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
