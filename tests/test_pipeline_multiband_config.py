"""Tests for production multi-band configuration parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from baymorph.fitting import (
    FixedPrior,
    NormalPrior,
    UniformPrior,
)
from baymorph.pipeline.multiband_config import (
    build_multiband_parameter_layout,
    parse_prior_config,
    resolve_multiband_observation_configs,
)


def _parameter_config() -> dict:
    return {
        "flux": {
            "coupling": "independent",
            "priors": {
                "g": {
                    "type": "uniform",
                    "lower": 100.0,
                    "upper": 1000.0,
                },
                "r": {
                    "type": "normal",
                    "mean": 900.0,
                    "sigma": 100.0,
                },
            },
        },
        "re": {
            "coupling": "shared",
            "prior": {
                "type": "uniform",
                "lower": 1.0,
                "upper": 20.0,
            },
        },
        "n": {
            "coupling": "shared",
            "prior": {
                "type": "fixed",
                "value": 2.5,
            },
        },
        "q": {
            "coupling": "shared",
            "prior": {
                "type": "uniform",
                "lower": 0.3,
                "upper": 1.0,
            },
        },
        "theta": {
            "coupling": "shared",
            "prior": {
                "type": "uniform",
                "lower": -1.57,
                "upper": 1.57,
            },
        },
        "x0": {
            "coupling": "shared",
            "prior": {
                "type": "uniform",
                "lower": 10.0,
                "upper": 20.0,
            },
        },
        "y0": {
            "coupling": "shared",
            "prior": {
                "type": "uniform",
                "lower": 10.0,
                "upper": 20.0,
            },
        },
        "sky": {
            "coupling": "independent",
            "priors": {
                "g": {
                    "type": "fixed",
                    "value": 0.0,
                },
                "r": {
                    "type": "uniform",
                    "lower": -5.0,
                    "upper": 5.0,
                },
            },
        },
    }


def _multiband_config() -> dict:
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
            "multiband": {
                "parameters": (
                    _parameter_config()
                ),
            },
        },
    }


def test_parse_prior_config_builds_all_supported_types():
    fixed = parse_prior_config(
        {
            "type": "fixed",
            "value": 3.0,
        }
    )

    uniform = parse_prior_config(
        {
            "type": "uniform",
            "lower": 1.0,
            "upper": 4.0,
        }
    )

    normal = parse_prior_config(
        {
            "type": "normal",
            "mean": 2.0,
            "sigma": 0.5,
        }
    )

    assert fixed == FixedPrior(
        value=3.0
    )
    assert uniform == UniformPrior(
        lower=1.0,
        upper=4.0,
    )
    assert normal == NormalPrior(
        mean=2.0,
        sigma=0.5,
    )


def test_parse_prior_config_rejects_implicit_type():
    with pytest.raises(
        ValueError,
        match="Prior type",
    ):
        parse_prior_config(
            {
                "lower": 1.0,
                "upper": 4.0,
            }
        )


def test_parse_prior_config_rejects_extra_fields():
    with pytest.raises(
        ValueError,
        match="Extra",
    ):
        parse_prior_config(
            {
                "type": "fixed",
                "value": 3.0,
                "sigma": 1.0,
            }
        )


def test_build_layout_preserves_order_and_typed_priors():
    layout = (
        build_multiband_parameter_layout(
            _multiband_config()
        )
    )

    assert layout.bands == (
        "g",
        "r",
    )

    assert layout.parameter_set.names == [
        "flux__g",
        "flux__r",
        "re",
        "q",
        "theta",
        "x0",
        "y0",
        "sky__r",
    ]

    assert (
        layout.parameter_set.fixed_values
        == {
            "n": pytest.approx(
                2.5
            ),
            "sky__g": pytest.approx(
                0.0
            ),
        }
    )

    all_parameters = (
        layout.parameter_set
        .all_parameters
    )

    assert isinstance(
        all_parameters[0].prior,
        UniformPrior,
    )
    assert isinstance(
        all_parameters[1].prior,
        NormalPrior,
    )


def test_build_layout_requires_every_model_parameter():
    config = (
        _multiband_config()
    )

    del (
        config["fit"]
        ["multiband"]
        ["parameters"]
        ["sky"]
    )

    with pytest.raises(
        ValueError,
        match=r"Missing: \['sky'\]",
    ):
        build_multiband_parameter_layout(
            config
        )


def test_independent_parameter_requires_every_observation():
    config = (
        _multiband_config()
    )

    del (
        config["fit"]
        ["multiband"]
        ["parameters"]
        ["flux"]
        ["priors"]
        ["r"]
    )

    with pytest.raises(
        ValueError,
        match=(
            "exactly the configured "
            "observations"
        ),
    ):
        build_multiband_parameter_layout(
            config
        )


def test_layout_rejects_unsupported_model_set():
    config = (
        _multiband_config()
    )

    config["fit"]["models"] = [
        "double_sersic",
    ]

    with pytest.raises(
        ValueError,
        match="single_sersic",
    ):
        build_multiband_parameter_layout(
            config
        )


def test_resolve_observation_configs_loads_pipeline_configs(
    tmp_path: Path,
):
    for band in (
        "g",
        "r",
    ):
        path = (
            tmp_path
            / f"{band}.yaml"
        )

        path.write_text(
            yaml.safe_dump(
                {
                    "object": {
                        "name": (
                            f"target_{band}"
                        ),
                    },
                    "data": {
                        "image_path": (
                            f"data/{band}.fits"
                        ),
                    },
                    "outputs": {
                        "root": "runs",
                    },
                }
            )
        )

    resolved = (
        resolve_multiband_observation_configs(
            _multiband_config(),
            base_dir=tmp_path,
        )
    )

    assert tuple(
        resolved
    ) == (
        "g",
        "r",
    )

    assert (
        resolved["g"]
        ["object"]
        ["name"]
        == "target_g"
    )

    assert (
        resolved["r"]
        ["data"]
        ["image_path"]
        == "data/r.fits"
    )

    assert (
        resolved["g"]
        ["fit"]
        ["sampler"]
        ["name"]
        == "pocomc"
    )


def test_effective_radius_can_be_observation_specific():
    config = _multiband_config()

    config["fit"]["multiband"]["parameters"]["re"] = {
        "coupling": "independent",
        "priors": {
            "g": {
                "type": "uniform",
                "lower": 1.0,
                "upper": 15.0,
            },
            "r": {
                "type": "uniform",
                "lower": 1.0,
                "upper": 25.0,
            },
        },
    }

    layout = build_multiband_parameter_layout(
        config
    )

    assert "re" not in layout.parameter_set.names
    assert "re__g" in layout.parameter_set.names
    assert "re__r" in layout.parameter_set.names

    routed = layout.route(
        {
            "flux__g": 500.0,
            "flux__r": 900.0,
            "re__g": 4.0,
            "re__r": 7.0,
            "q": 0.7,
            "theta": 0.2,
            "x0": 15.0,
            "y0": 15.0,
            "sky__r": 1.0,
        }
    )

    assert routed["g"]["re"] == pytest.approx(4.0)
    assert routed["r"]["re"] == pytest.approx(7.0)
