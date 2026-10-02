"""Tests for canonical multi-observation configuration parsing."""

from __future__ import annotations

import pytest

from baymorph.fitting.multi_observation import (
    MultiObservationParameterLayout,
)
from baymorph.pipeline.multi_observation_config import (
    build_multi_observation_parameter_layout,
    multi_observation_parameter_frame,
)


def _prior(
    value: float,
) -> dict[str, object]:
    return {
        "type": "fixed",
        "value": value,
    }


def _config(
    *,
    section_name: str = "multi_observation",
) -> dict:
    parameters = {
        "flux": {
            "coupling": "independent",
            "priors": {
                "g": _prior(
                    100.0
                ),
                "r": _prior(
                    120.0
                ),
            },
        },
        "re": {
            "coupling": "shared",
            "prior": _prior(
                5.0
            ),
        },
        "n": {
            "coupling": "shared",
            "prior": _prior(
                2.0
            ),
        },
        "q": {
            "coupling": "shared",
            "prior": _prior(
                0.8
            ),
        },
        "theta": {
            "coupling": "shared",
            "prior": _prior(
                0.2
            ),
        },
        "x0": {
            "coupling": "shared",
            "prior": _prior(
                20.0
            ),
        },
        "y0": {
            "coupling": "shared",
            "prior": _prior(
                20.0
            ),
        },
        "background_level": {
            "coupling": "independent",
            "priors": {
                "g": _prior(
                    0.0
                ),
                "r": _prior(
                    0.0
                ),
            },
        },
    }

    return {
        "observations": {
            "g": {
                "config_path": "g.yaml",
            },
            "r": {
                "config_path": "r.yaml",
            },
        },
        "fit": {
            "models": [
                "single_sersic",
            ],
            section_name: {
                "parameter_frame": "detector",
                "parameters": parameters,
            },
        },
    }


def test_canonical_multi_observation_section_builds_layout():
    layout = (
        build_multi_observation_parameter_layout(
            _config()
        )
    )

    assert isinstance(
        layout,
        MultiObservationParameterLayout,
    )

    assert layout.observations == (
        "g",
        "r",
    )


def test_parameter_frame_reads_canonical_section():
    assert (
        multi_observation_parameter_frame(
            _config()
        )
        == "detector"
    )


def test_parameter_frame_reports_canonical_path():
    config = _config()

    config[
        "fit"
    ][
        "multi_observation"
    ][
        "parameter_frame"
    ] = "invalid"

    with pytest.raises(
        ValueError,
        match=(
            r"fit\.multi_observation"
            r"\.parameter_frame"
        ),
    ):
        multi_observation_parameter_frame(
            config
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "boxy_sersic",
        "core_sersic",
        "bulge_broken_disk",
        "bulge_disk_bar",
        "bulge_disk_ring",
        "bulge_disk_spiral",
    ],
)
def test_models_without_geometry_are_rejected_in_every_frame(model_name):
    """The families of AD-048 are single-observation models for now."""
    for frame in ("detector", "sky"):
        config = _config()
        config["fit"]["models"] = [model_name]
        config["fit"]["multi_observation"]["parameter_frame"] = frame

        with pytest.raises(ValueError, match="no geometry contract"):
            build_multi_observation_parameter_layout(config)
