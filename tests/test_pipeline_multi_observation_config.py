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
from baymorph.pipeline.multi_observation_config import (
    build_multi_observation_parameter_layout,
    parse_prior_config,
    resolve_multi_observation_observation_configs,
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


def _multi_observation_config() -> dict:
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
            "multi_observation": {
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
        build_multi_observation_parameter_layout(
            _multi_observation_config()
        )
    )

    assert layout.observations == (
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
        _multi_observation_config()
    )

    del (
        config["fit"]
        ["multi_observation"]
        ["parameters"]
        ["sky"]
    )

    with pytest.raises(
        ValueError,
        match=r"Missing: \['sky'\]",
    ):
        build_multi_observation_parameter_layout(
            config
        )


def test_independent_parameter_requires_every_observation():
    config = (
        _multi_observation_config()
    )

    del (
        config["fit"]
        ["multi_observation"]
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
        build_multi_observation_parameter_layout(
            config
        )


def test_layout_rejects_multiple_model_set():
    config = (
        _multi_observation_config()
    )

    config["fit"]["models"] = [
        "single_sersic",
        "double_sersic",
    ]

    with pytest.raises(
        ValueError,
        match="exactly one registered model",
    ):
        build_multi_observation_parameter_layout(
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
        resolve_multi_observation_observation_configs(
            _multi_observation_config(),
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
    config = _multi_observation_config()

    config["fit"]["multi_observation"]["parameters"]["re"] = {
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

    layout = build_multi_observation_parameter_layout(
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


def _sky_parameter_config() -> dict:
    parameters = _parameter_config()

    parameters.pop("re")
    parameters.pop("theta")
    parameters.pop("x0")
    parameters.pop("y0")

    parameters["re_arcsec"] = {
        "coupling": "independent",
        "priors": {
            "g": {
                "type": "uniform",
                "lower": 0.1,
                "upper": 5.0,
            },
            "r": {
                "type": "uniform",
                "lower": 0.1,
                "upper": 8.0,
            },
        },
    }

    parameters["position_angle"] = {
        "coupling": "shared",
        "prior": {
            "type": "uniform",
            "lower": -1.57,
            "upper": 1.57,
        },
    }

    parameters["ra"] = {
        "coupling": "shared",
        "prior": {
            "type": "uniform",
            "lower": 149.99,
            "upper": 150.01,
        },
    }

    parameters["dec"] = {
        "coupling": "shared",
        "prior": {
            "type": "uniform",
            "lower": 29.99,
            "upper": 30.01,
        },
    }

    return parameters


def test_sky_parameter_frame_builds_distinct_scientific_state():
    config = _multi_observation_config()

    config["fit"]["multi_observation"][
        "parameter_frame"
    ] = "sky"
    config["fit"]["multi_observation"][
        "parameters"
    ] = _sky_parameter_config()

    layout = build_multi_observation_parameter_layout(
        config
    )

    assert layout.parameter_set.names == [
        "flux__g",
        "flux__r",
        "re_arcsec__g",
        "re_arcsec__r",
        "q",
        "position_angle",
        "ra",
        "dec",
        "sky__r",
    ]

    assert "re" not in layout.parameter_set.all_names
    assert "theta" not in layout.parameter_set.all_names
    assert "x0" not in layout.parameter_set.all_names
    assert "y0" not in layout.parameter_set.all_names


def test_detector_parameter_frame_remains_default():
    config = _multi_observation_config()

    layout = build_multi_observation_parameter_layout(
        config
    )

    assert "re" in layout.parameter_set.all_names
    assert "theta" in layout.parameter_set.all_names
    assert "x0" in layout.parameter_set.all_names
    assert "y0" in layout.parameter_set.all_names

    assert "re_arcsec" not in layout.parameter_set.all_names
    assert "ra" not in layout.parameter_set.all_names
    assert "dec" not in layout.parameter_set.all_names


def test_multi_observation_parameter_frame_rejects_unknown_frame():
    config = _multi_observation_config()

    config["fit"]["multi_observation"][
        "parameter_frame"
    ] = "physical"

    with pytest.raises(
        ValueError,
        match="parameter_frame",
    ):
        build_multi_observation_parameter_layout(
            config
        )


def test_detector_frame_uses_registered_model_parameter_contract():
    from baymorph.models import get_model

    config = _multi_observation_config()

    config["fit"]["models"] = [
        "bulge_disk",
    ]

    model_parameters = list(
        get_model(
            "bulge_disk"
        ).parameter_names
    )

    config["fit"]["multi_observation"][
        "parameters"
    ] = {
        name: {
            "coupling": "shared",
            "prior": {
                "type": "uniform",
                "lower": 0.1,
                "upper": 10.0,
            },
        }
        for name in model_parameters
    }

    layout = build_multi_observation_parameter_layout(
        config
    )

    assert layout.parameter_set.all_names == (
        model_parameters
    )


def test_sky_frame_uses_registered_model_geometry_contract():
    from baymorph.models.geometry import (
        sky_parameter_names,
    )

    config = _multi_observation_config()

    config["fit"]["models"] = [
        "bulge_disk",
    ]
    config["fit"]["multi_observation"][
        "parameter_frame"
    ] = "sky"

    names = sky_parameter_names(
        "bulge_disk"
    )

    config["fit"]["multi_observation"][
        "parameters"
    ] = {
        name: {
            "coupling": "shared",
            "prior": {
                "type": "uniform",
                "lower": 0.1,
                "upper": 10.0,
            },
        }
        for name in names
    }

    layout = build_multi_observation_parameter_layout(
        config
    )

    assert layout.parameter_set.all_names == list(
        names
    )


def test_effective_radius_can_be_hierarchical():
    import math

    config = _multi_observation_config()

    config["fit"]["multi_observation"]["parameters"]["re"] = {
        "coupling": "hierarchical",
        "transform": "exp",
        "location_prior": {
            "type": "uniform",
            "lower": math.log(1.0),
            "upper": math.log(20.0),
        },
        "scale_prior": {
            "type": "uniform",
            "lower": 0.0,
            "upper": 0.5,
        },
    }

    layout = build_multi_observation_parameter_layout(
        config
    )

    assert "re" not in layout.parameter_set.all_names
    assert "re__g" not in layout.parameter_set.all_names
    assert "re__r" not in layout.parameter_set.all_names

    assert "re__eta_mu" in layout.parameter_set.names
    assert "re__eta_tau" in layout.parameter_set.names
    assert "re__z__g" in layout.parameter_set.names
    assert "re__z__r" in layout.parameter_set.names

    routed = layout.route(
        {
            "flux__g": 500.0,
            "flux__r": 900.0,
            "re__eta_mu": math.log(5.0),
            "re__eta_tau": 0.2,
            "re__z__g": -1.0,
            "re__z__r": 1.0,
            "q": 0.7,
            "theta": 0.2,
            "x0": 15.0,
            "y0": 15.0,
            "sky__r": 1.0,
        }
    )

    assert routed["g"]["re"] == pytest.approx(
        5.0 * math.exp(-0.2)
    )
    assert routed["r"]["re"] == pytest.approx(
        5.0 * math.exp(0.2)
    )

    assert routed["g"]["re"] > 0.0
    assert routed["r"]["re"] > 0.0


def test_hierarchical_rejects_negative_scale_support():
    import math

    config = _multi_observation_config()

    config["fit"]["multi_observation"]["parameters"]["re"] = {
        "coupling": "hierarchical",
        "transform": "exp",
        "location_prior": {
            "type": "uniform",
            "lower": math.log(1.0),
            "upper": math.log(20.0),
        },
        "scale_prior": {
            "type": "uniform",
            "lower": -0.1,
            "upper": 0.5,
        },
    }

    with pytest.raises(
        ValueError,
        match="non-negative support",
    ):
        build_multi_observation_parameter_layout(
            config
        )


def test_hierarchical_rejects_extra_fields():
    import math

    config = _multi_observation_config()

    config["fit"]["multi_observation"]["parameters"]["re"] = {
        "coupling": "hierarchical",
        "transform": "exp",
        "location_prior": {
            "type": "uniform",
            "lower": math.log(1.0),
            "upper": math.log(20.0),
        },
        "scale_prior": {
            "type": "uniform",
            "lower": 0.0,
            "upper": 0.5,
        },
        "latent_priors": {},
    }

    with pytest.raises(
        ValueError,
        match="Extra",
    ):
        build_multi_observation_parameter_layout(
            config
        )



def test_positive_radius_and_axis_ratio_can_be_hierarchical_together():
    import math

    config = _multi_observation_config()

    central_q = 0.7
    central_q_eta = math.log(
        central_q
        / (1.0 - central_q)
    )

    parameters = (
        config["fit"]
        ["multi_observation"]
        ["parameters"]
    )

    parameters["re"] = {
        "coupling": "hierarchical",
        "transform": "exp",
        "location_prior": {
            "type": "uniform",
            "lower": math.log(1.0),
            "upper": math.log(20.0),
        },
        "scale_prior": {
            "type": "uniform",
            "lower": 0.0,
            "upper": 0.5,
        },
    }

    parameters["q"] = {
        "coupling": "hierarchical",
        "transform": "logistic",
        "location_prior": {
            "type": "uniform",
            "lower": central_q_eta - 1.0,
            "upper": central_q_eta + 1.0,
        },
        "scale_prior": {
            "type": "uniform",
            "lower": 0.0,
            "upper": 0.5,
        },
    }

    layout = build_multi_observation_parameter_layout(
        config
    )

    for name in (
        "re__eta_mu",
        "re__eta_tau",
        "re__z__g",
        "re__z__r",
        "q__eta_mu",
        "q__eta_tau",
        "q__z__g",
        "q__z__r",
    ):
        assert name in layout.parameter_set.names

    routed = layout.route(
        {
            "flux__g": 500.0,
            "flux__r": 900.0,
            "re__eta_mu": math.log(5.0),
            "re__eta_tau": 0.2,
            "re__z__g": -1.0,
            "re__z__r": 1.0,
            "q__eta_mu": central_q_eta,
            "q__eta_tau": 0.3,
            "q__z__g": -1.0,
            "q__z__r": 1.0,
            "theta": 0.2,
            "x0": 15.0,
            "y0": 15.0,
            "sky__r": 1.0,
        }
    )

    assert routed["g"]["re"] == pytest.approx(
        5.0 * math.exp(-0.2)
    )
    assert routed["r"]["re"] == pytest.approx(
        5.0 * math.exp(0.2)
    )

    assert 0.0 < routed["g"]["q"] < 1.0
    assert 0.0 < routed["r"]["q"] < 1.0

    assert routed["g"]["q"] < central_q
    assert routed["r"]["q"] > central_q


def test_morphology_geometry_can_be_hierarchical_together():
    import math

    config = _multi_observation_config()

    parameters = (
        config["fit"]
        ["multi_observation"]
        ["parameters"]
    )

    central_q = 0.7
    central_q_eta = math.log(
        central_q
        / (1.0 - central_q)
    )

    parameters["re"] = {
        "coupling": "hierarchical",
        "transform": "exp",
        "location_prior": {
            "type": "uniform",
            "lower": math.log(1.0),
            "upper": math.log(20.0),
        },
        "scale_prior": {
            "type": "uniform",
            "lower": 0.0,
            "upper": 0.5,
        },
    }

    parameters["q"] = {
        "coupling": "hierarchical",
        "transform": "logistic",
        "location_prior": {
            "type": "uniform",
            "lower": central_q_eta - 1.0,
            "upper": central_q_eta + 1.0,
        },
        "scale_prior": {
            "type": "uniform",
            "lower": 0.0,
            "upper": 0.5,
        },
    }

    parameters["theta"] = {
        "coupling": "hierarchical",
        "transform": "axial_angle",
        "location_prior": {
            "type": "uniform",
            "lower": -math.pi / 2.0,
            "upper": math.pi / 2.0,
        },
        "scale_prior": {
            "type": "uniform",
            "lower": 0.0,
            "upper": 0.5,
        },
    }

    layout = build_multi_observation_parameter_layout(
        config
    )

    expected_hierarchical_names = {
        "re__eta_mu",
        "re__eta_tau",
        "re__z__g",
        "re__z__r",
        "q__eta_mu",
        "q__eta_tau",
        "q__z__g",
        "q__z__r",
        "theta__eta_mu",
        "theta__eta_tau",
        "theta__z__g",
        "theta__z__r",
    }

    assert expected_hierarchical_names.issubset(
        layout.parameter_set.names
    )

    theta_location = 1.5
    theta_scale = 0.2

    routed = layout.route(
        {
            "flux__g": 500.0,
            "flux__r": 900.0,
            "re__eta_mu": math.log(5.0),
            "re__eta_tau": 0.2,
            "re__z__g": -1.0,
            "re__z__r": 1.0,
            "q__eta_mu": central_q_eta,
            "q__eta_tau": 0.3,
            "q__z__g": -1.0,
            "q__z__r": 1.0,
            "theta__eta_mu": theta_location,
            "theta__eta_tau": theta_scale,
            "theta__z__g": -1.0,
            "theta__z__r": 1.0,
            "x0": 15.0,
            "y0": 15.0,
            "sky__r": 1.0,
        }
    )

    assert routed["g"]["re"] == pytest.approx(
        5.0 * math.exp(-0.2)
    )
    assert routed["r"]["re"] == pytest.approx(
        5.0 * math.exp(0.2)
    )

    assert 0.0 < routed["g"]["q"] < 1.0
    assert 0.0 < routed["r"]["q"] < 1.0

    expected_theta_g = (
        (
            theta_location
            - theta_scale
            + math.pi / 2.0
        )
        % math.pi
        - math.pi / 2.0
    )
    expected_theta_r = (
        (
            theta_location
            + theta_scale
            + math.pi / 2.0
        )
        % math.pi
        - math.pi / 2.0
    )

    assert routed["g"]["theta"] == pytest.approx(
        expected_theta_g
    )
    assert routed["r"]["theta"] == pytest.approx(
        expected_theta_r
    )

    assert (
        -math.pi / 2.0
        <= routed["g"]["theta"]
        < math.pi / 2.0
    )
    assert (
        -math.pi / 2.0
        <= routed["r"]["theta"]
        < math.pi / 2.0
    )


def test_full_morphology_can_use_domain_aware_hierarchical_coupling():
    import math

    config = _multi_observation_config()

    parameters = (
        config["fit"]
        ["multi_observation"]
        ["parameters"]
    )

    central_q = 0.7
    central_q_eta = math.log(
        central_q
        / (1.0 - central_q)
    )

    parameters["re"] = {
        "coupling": "hierarchical",
        "transform": "exp",
        "location_prior": {
            "type": "uniform",
            "lower": math.log(1.0),
            "upper": math.log(20.0),
        },
        "scale_prior": {
            "type": "uniform",
            "lower": 0.0,
            "upper": 0.5,
        },
    }

    parameters["n"] = {
        "coupling": "hierarchical",
        "transform": "exp",
        "location_prior": {
            "type": "uniform",
            "lower": math.log(0.5),
            "upper": math.log(8.0),
        },
        "scale_prior": {
            "type": "uniform",
            "lower": 0.0,
            "upper": 0.5,
        },
    }

    parameters["q"] = {
        "coupling": "hierarchical",
        "transform": "logistic",
        "location_prior": {
            "type": "uniform",
            "lower": central_q_eta - 1.0,
            "upper": central_q_eta + 1.0,
        },
        "scale_prior": {
            "type": "uniform",
            "lower": 0.0,
            "upper": 0.5,
        },
    }

    parameters["theta"] = {
        "coupling": "hierarchical",
        "transform": "axial_angle",
        "location_prior": {
            "type": "uniform",
            "lower": -math.pi / 2.0,
            "upper": math.pi / 2.0,
        },
        "scale_prior": {
            "type": "uniform",
            "lower": 0.0,
            "upper": 0.5,
        },
    }

    layout = build_multi_observation_parameter_layout(
        config
    )

    routed = layout.route(
        {
            "flux__g": 500.0,
            "flux__r": 900.0,
            "re__eta_mu": math.log(5.0),
            "re__eta_tau": 0.2,
            "re__z__g": -1.0,
            "re__z__r": 1.0,
            "n__eta_mu": math.log(2.5),
            "n__eta_tau": 0.1,
            "n__z__g": -1.0,
            "n__z__r": 1.0,
            "q__eta_mu": central_q_eta,
            "q__eta_tau": 0.3,
            "q__z__g": -1.0,
            "q__z__r": 1.0,
            "theta__eta_mu": 1.5,
            "theta__eta_tau": 0.2,
            "theta__z__g": -1.0,
            "theta__z__r": 1.0,
            "x0": 15.0,
            "y0": 15.0,
            "sky__r": 1.0,
        }
    )

    assert routed["g"]["re"] == pytest.approx(
        5.0 * math.exp(-0.2)
    )
    assert routed["r"]["re"] == pytest.approx(
        5.0 * math.exp(0.2)
    )

    assert routed["g"]["n"] == pytest.approx(
        2.5 * math.exp(-0.1)
    )
    assert routed["r"]["n"] == pytest.approx(
        2.5 * math.exp(0.1)
    )
    assert routed["g"]["n"] > 0.0
    assert routed["r"]["n"] > 0.0

    assert 0.0 < routed["g"]["q"] < 1.0
    assert 0.0 < routed["r"]["q"] < 1.0

    for observation in ("g", "r"):
        assert (
            -math.pi / 2.0
            <= routed[observation]["theta"]
            < math.pi / 2.0
        )
