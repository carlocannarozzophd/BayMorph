import pytest

from baymorph.fitting.multi_observation import (
    MultiObservationParameterLayout,
    MultiObservationParameterSpec,
)
from baymorph.fitting.parameters import Parameter


def make_layout() -> MultiObservationParameterLayout:
    return MultiObservationParameterLayout(
        observations=["g", "r"],
        specifications=[
            MultiObservationParameterSpec.shared_parameter(
                Parameter(
                    name="re",
                    lower=1.0,
                    upper=20.0,
                )
            ),
            MultiObservationParameterSpec.independent(
                name="flux",
                parameters={
                    "g": Parameter(
                        name="flux",
                        lower=10.0,
                        upper=100.0,
                    ),
                    "r": Parameter(
                        name="flux",
                        lower=20.0,
                        upper=200.0,
                    ),
                },
            ),
        ],
    )


def test_layout_builds_global_parameter_set():
    layout = make_layout()

    assert layout.parameter_set.names == [
        "re",
        "flux__g",
        "flux__r",
    ]
    assert layout.ndim == 3
    assert layout.parameter_set.bounds == {
        "re": (1.0, 20.0),
        "flux__g": (10.0, 100.0),
        "flux__r": (20.0, 200.0),
    }


def test_layout_routes_shared_and_observation_specific_values():
    layout = make_layout()

    routed = layout.route(
        {
            "re": 4.5,
            "flux__g": 50.0,
            "flux__r": 120.0,
        }
    )

    assert routed == {
        "g": {
            "re": 4.5,
            "flux": 50.0,
        },
        "r": {
            "re": 4.5,
            "flux": 120.0,
        },
    }


def test_layout_rejects_missing_global_values():
    layout = make_layout()

    with pytest.raises(
        KeyError,
        match="flux__r",
    ):
        layout.route(
            {
                "re": 4.5,
                "flux__g": 50.0,
            }
        )


def test_layout_rejects_duplicate_bands():
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        MultiObservationParameterLayout(
            observations=["g", "g"],
            specifications=[],
        )


def test_layout_rejects_incomplete_observation_specific_definition():
    with pytest.raises(
        ValueError,
        match="exactly the configured observations",
    ):
        MultiObservationParameterLayout(
            observations=["g", "r"],
            specifications=[
                MultiObservationParameterSpec.independent(
                    name="flux",
                    parameters={
                        "g": Parameter(
                            name="flux",
                            lower=10.0,
                            upper=100.0,
                        ),
                    },
                )
            ],
        )


def test_layout_rejects_parameter_name_mismatch():
    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        MultiObservationParameterLayout(
            observations=["g"],
            specifications=[
                MultiObservationParameterSpec.shared_parameter(
                    Parameter(
                        name="radius",
                        lower=1.0,
                        upper=10.0,
                    )
                ),
                MultiObservationParameterSpec.independent(
                    name="flux",
                    parameters={
                        "g": Parameter(
                            name="wrong_name",
                            lower=10.0,
                            upper=100.0,
                        ),
                    },
                ),
            ],
        )
