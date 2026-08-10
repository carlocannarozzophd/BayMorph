"""Tests for deterministic observation parameter routing."""

import pytest

from baymorph.fitting.routing import (
    ProjectionParameterRouter,
)


def test_projection_router_supports_shared_global_coordinates():
    router = ProjectionParameterRouter(
        observations=[
            "g",
            "r",
        ],
        routes={
            "g": {
                "re": "re",
            },
            "r": {
                "re": "re",
            },
        },
    )

    assert router.route(
        {
            "re": 4.5,
        }
    ) == {
        "g": {
            "re": 4.5,
        },
        "r": {
            "re": 4.5,
        },
    }


def test_projection_router_supports_observation_specific_coordinates():
    router = ProjectionParameterRouter(
        observations=[
            "g",
            "r",
        ],
        routes={
            "g": {
                "flux": "flux__g",
            },
            "r": {
                "flux": "flux__r",
            },
        },
    )

    assert router.route(
        {
            "flux__g": 50.0,
            "flux__r": 120.0,
        }
    ) == {
        "g": {
            "flux": 50.0,
        },
        "r": {
            "flux": 120.0,
        },
    }


def test_projection_router_rejects_missing_global_values():
    router = ProjectionParameterRouter(
        observations=[
            "g",
        ],
        routes={
            "g": {
                "flux": "flux__g",
            },
        },
    )

    with pytest.raises(
        KeyError,
        match="flux__g",
    ):
        router.route({})
