"""Tests for typed priors and explicit multi-band coupling."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting import (
    FixedPrior,
    NormalPrior,
    Parameter,
    UniformPrior,
)
from baymorph.fitting.multi_observation import (
    MultiObservationParameterLayout,
    MultiObservationParameterSpec,
)


def test_layout_supports_shared_and_independent_parameters():
    layout = MultiObservationParameterLayout(
        observations=["g", "r"],
        specifications=[
            MultiObservationParameterSpec.shared_parameter(
                Parameter(
                    "re",
                    prior=UniformPrior(
                        lower=1.0,
                        upper=10.0,
                    ),
                )
            ),
            MultiObservationParameterSpec.independent(
                "flux",
                {
                    "g": Parameter(
                        "flux",
                        prior=UniformPrior(
                            lower=100.0,
                            upper=1000.0,
                        ),
                    ),
                    "r": Parameter(
                        "flux",
                        prior=NormalPrior(
                            mean=900.0,
                            sigma=100.0,
                        ),
                    ),
                },
            ),
        ],
    )

    assert layout.parameter_set.names == [
        "re",
        "flux__g",
        "flux__r",
    ]

    assert layout.ndim == 3


def test_layout_routes_fixed_parameters_without_sampling_them():
    layout = MultiObservationParameterLayout(
        observations=["g", "r"],
        specifications=[
            MultiObservationParameterSpec.shared_parameter(
                Parameter(
                    "x0",
                    prior=FixedPrior(
                        value=15.5,
                    ),
                )
            ),
            MultiObservationParameterSpec.independent(
                "background_level",
                {
                    "g": Parameter(
                        "background_level",
                        prior=FixedPrior(
                            value=1.0,
                        ),
                    ),
                    "r": Parameter(
                        "background_level",
                        prior=UniformPrior(
                            lower=-2.0,
                            upper=2.0,
                        ),
                    ),
                },
            ),
        ],
    )

    assert layout.ndim == 1
    assert layout.parameter_set.names == [
        "background_level__r"
    ]

    routed = layout.route(
        {
            "background_level__r": 0.5,
        }
    )

    assert routed == {
        "g": {
            "x0": pytest.approx(15.5),
            "background_level": pytest.approx(1.0),
        },
        "r": {
            "x0": pytest.approx(15.5),
            "background_level": pytest.approx(0.5),
        },
    }


def test_independent_prior_types_are_preserved():
    layout = MultiObservationParameterLayout(
        observations=["g", "r"],
        specifications=[
            MultiObservationParameterSpec.independent(
                "flux",
                {
                    "g": Parameter(
                        "flux",
                        prior=UniformPrior(
                            lower=1.0,
                            upper=3.0,
                        ),
                    ),
                    "r": Parameter(
                        "flux",
                        prior=NormalPrior(
                            mean=5.0,
                            sigma=2.0,
                        ),
                    ),
                },
            )
        ],
    )

    transformed = layout.parameter_set.transform(
        np.array(
            [
                0.5,
                0.5,
            ]
        )
    )

    assert transformed == pytest.approx(
        np.array(
            [
                2.0,
                5.0,
            ]
        )
    )
