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
        "background_level": {
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
        "background_level__r",
    ]

    assert (
        layout.parameter_set.fixed_values
        == {
            "n": pytest.approx(
                2.5
            ),
            "background_level__g": pytest.approx(
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
        ["background_level"]
    )

    with pytest.raises(
        ValueError,
        match=r"Missing: \['background_level'\]",
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
        == "nautilus"
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
            "background_level__r": 1.0,
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
        "background_level__r",
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


def _independent_background() -> dict:
    return {
        "coupling": "independent",
        "priors": {
            observation: {
                "type": "uniform",
                "lower": -5.0,
                "upper": 5.0,
            }
            for observation in (
                "g",
                "r",
            )
        },
    }


def test_background_parameters_must_be_observation_specific():
    config = _multi_observation_config()

    config["fit"]["multi_observation"]["parameters"][
        "background_level"
    ] = {
        "coupling": "shared",
        "prior": {
            "type": "uniform",
            "lower": -5.0,
            "upper": 5.0,
        },
    }

    with pytest.raises(
        ValueError,
        match="independent",
    ):
        build_multi_observation_parameter_layout(
            config
        )


def test_legacy_sky_parameter_is_rejected_with_migration_hint():
    config = _multi_observation_config()

    parameters = config["fit"]["multi_observation"]["parameters"]
    parameters["sky"] = parameters.pop(
        "background_level"
    )

    with pytest.raises(
        ValueError,
        match="legacy 'sky' parameter",
    ):
        build_multi_observation_parameter_layout(
            config
        )


def test_plane_background_extends_the_joint_parameter_contract():
    config = _multi_observation_config()

    config["fit"]["background"] = {
        "model": "plane",
    }

    parameters = config["fit"]["multi_observation"]["parameters"]

    for name in (
        "background_gradient_x",
        "background_gradient_y",
    ):
        parameters[name] = _independent_background()

    layout = build_multi_observation_parameter_layout(
        config
    )

    assert layout.parameter_set.all_names[-6:] == [
        "background_level__g",
        "background_level__r",
        "background_gradient_x__g",
        "background_gradient_x__r",
        "background_gradient_y__g",
        "background_gradient_y__r",
    ]


def test_joint_background_section_declares_only_the_model_family():
    config = _multi_observation_config()

    config["fit"]["background"] = {
        "model": "constant",
        "priors": {},
    }

    with pytest.raises(
        ValueError,
        match="accepts only 'model'",
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
    config["fit"]["multi_observation"]["parameters"][
        "background_level"
    ] = _independent_background()

    layout = build_multi_observation_parameter_layout(
        config
    )

    assert layout.parameter_set.all_names == (
        model_parameters
        + [
            "background_level__g",
            "background_level__r",
        ]
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
    config["fit"]["multi_observation"]["parameters"][
        "background_level"
    ] = _independent_background()

    layout = build_multi_observation_parameter_layout(
        config
    )

    assert layout.parameter_set.all_names == list(
        names
    ) + [
        "background_level__g",
        "background_level__r",
    ]


def test_effective_radius_can_use_physical_prior_hierarchically():
    from scipy.special import ndtr

    config = _multi_observation_config()

    parameters = (
        config["fit"]
        ["multi_observation"]
        ["parameters"]
    )

    parameters["re"] = {
        "coupling": "hierarchical",
        "prior": {
            "type": "uniform",
            "lower": 0.5,
            "upper": 30.0,
        },
    }

    layout = build_multi_observation_parameter_layout(
        config
    )

    assert "re" not in layout.parameter_set.all_names
    assert "re__g" not in layout.parameter_set.all_names
    assert "re__r" not in layout.parameter_set.all_names

    assert "re__rho" in layout.parameter_set.names
    assert "re__z__g" in layout.parameter_set.names
    assert "re__z__r" in layout.parameter_set.names

    routed = layout.route(
        {
            "flux__g": 500.0,
            "flux__r": 900.0,
            "re__rho": 0.0,
            "re__z__g": -1.0,
            "re__z__r": 1.0,
            "n": 2.5,
            "q": 0.7,
            "theta": 0.2,
            "x0": 15.0,
            "y0": 15.0,
            "background_level__r": 1.0,
        }
    )

    expected_g = (
        0.5
        + (30.0 - 0.5) * float(ndtr(-1.0))
    )
    expected_r = (
        0.5
        + (30.0 - 0.5) * float(ndtr(1.0))
    )

    assert routed["g"]["re"] == pytest.approx(
        expected_g
    )
    assert routed["r"]["re"] == pytest.approx(
        expected_r
    )

    assert 0.5 <= routed["g"]["re"] <= 30.0
    assert 0.5 <= routed["r"]["re"] <= 30.0


def test_hierarchical_rejects_extra_configuration_fields():
    config = _multi_observation_config()

    parameters = (
        config["fit"]
        ["multi_observation"]
        ["parameters"]
    )

    parameters["re"] = {
        "coupling": "hierarchical",
        "prior": {
            "type": "uniform",
            "lower": 0.5,
            "upper": 30.0,
        },
        "correlation_prior": {
            "type": "uniform",
            "lower": 0.0,
            "upper": 1.0,
        },
    }

    with pytest.raises(
        ValueError,
        match="Extra",
    ):
        build_multi_observation_parameter_layout(
            config
        )


def test_radius_and_axis_ratio_preserve_physical_priors_hierarchically():
    config = _multi_observation_config()

    parameters = (
        config["fit"]
        ["multi_observation"]
        ["parameters"]
    )

    parameters["re"] = {
        "coupling": "hierarchical",
        "prior": {
            "type": "uniform",
            "lower": 0.5,
            "upper": 30.0,
        },
    }

    parameters["q"] = {
        "coupling": "hierarchical",
        "prior": {
            "type": "uniform",
            "lower": 0.2,
            "upper": 1.0,
        },
    }

    layout = build_multi_observation_parameter_layout(
        config
    )

    for name in (
        "re__rho",
        "re__z__g",
        "re__z__r",
        "q__rho",
        "q__z__g",
        "q__z__r",
    ):
        assert name in layout.parameter_set.names

    routed = layout.route(
        {
            "flux__g": 500.0,
            "flux__r": 900.0,
            "re__rho": 0.7,
            "re__z__g": -1.0,
            "re__z__r": 1.0,
            "n": 2.5,
            "q__rho": 0.7,
            "q__z__g": -1.0,
            "q__z__r": 1.0,
            "theta": 0.2,
            "x0": 15.0,
            "y0": 15.0,
            "background_level__r": 1.0,
        }
    )

    for observation in ("g", "r"):
        assert (
            0.5
            <= routed[observation]["re"]
            <= 30.0
        )
        assert (
            0.2
            <= routed[observation]["q"]
            <= 1.0
        )


def test_re_n_q_can_be_hierarchical_with_shared_axial_angle():
    config = _multi_observation_config()

    parameters = (
        config["fit"]
        ["multi_observation"]
        ["parameters"]
    )

    parameters["re"] = {
        "coupling": "hierarchical",
        "prior": {
            "type": "uniform",
            "lower": 0.5,
            "upper": 30.0,
        },
    }

    parameters["n"] = {
        "coupling": "hierarchical",
        "prior": {
            "type": "uniform",
            "lower": 0.1,
            "upper": 6.0,
        },
    }

    parameters["q"] = {
        "coupling": "hierarchical",
        "prior": {
            "type": "uniform",
            "lower": 0.2,
            "upper": 1.0,
        },
    }

    parameters["theta"] = {
        "coupling": "shared",
        "prior": {
            "type": "uniform",
            "lower": -1.5707963267948966,
            "upper": 1.5707963267948966,
        },
    }

    layout = build_multi_observation_parameter_layout(
        config
    )

    expected_names = {
        "re__rho",
        "re__z__g",
        "re__z__r",
        "n__rho",
        "n__z__g",
        "n__z__r",
        "q__rho",
        "q__z__g",
        "q__z__r",
        "theta",
    }

    assert expected_names.issubset(
        layout.parameter_set.all_names
    )

    routed = layout.route(
        {
            "flux__g": 500.0,
            "flux__r": 900.0,
            "re__rho": 1.0,
            "re__z__g": -1.0,
            "re__z__r": 1.0,
            "n__rho": 1.0,
            "n__z__g": -0.5,
            "n__z__r": 0.5,
            "q__rho": 1.0,
            "q__z__g": -2.0,
            "q__z__r": 2.0,
            "theta": 0.3,
            "x0": 15.0,
            "y0": 15.0,
            "background_level__r": 1.0,
        }
    )

    assert routed["g"]["re"] == pytest.approx(
        routed["r"]["re"]
    )
    assert routed["g"]["n"] == pytest.approx(
        routed["r"]["n"]
    )
    assert routed["g"]["q"] == pytest.approx(
        routed["r"]["q"]
    )

    assert routed["g"]["theta"] == pytest.approx(
        0.3
    )
    assert routed["r"]["theta"] == pytest.approx(
        0.3
    )

    for observation in ("g", "r"):
        assert (
            0.5
            <= routed[observation]["re"]
            <= 30.0
        )
        assert (
            0.1
            <= routed[observation]["n"]
            <= 6.0
        )
        assert (
            0.2
            <= routed[observation]["q"]
            <= 1.0
        )


def test_parse_beta_prior_config():
    from baymorph.fitting.priors import BetaPrior
    from baymorph.pipeline.multi_observation_config import (
        parse_prior_config,
    )

    standard = parse_prior_config(
        {
            "type": "beta",
            "alpha": 2.0,
            "beta": 2.0,
        }
    )

    scaled = parse_prior_config(
        {
            "type": "beta",
            "alpha": 2.0,
            "beta": 5.0,
            "lower": 0.2,
            "upper": 1.0,
        }
    )

    assert standard == BetaPrior(
        alpha=2.0,
        beta=2.0,
    )

    assert scaled == BetaPrior(
        alpha=2.0,
        beta=5.0,
        lower=0.2,
        upper=1.0,
    )


def test_beta_prior_config_requires_both_scaled_bounds():
    from baymorph.pipeline.multi_observation_config import (
        parse_prior_config,
    )

    with pytest.raises(
        ValueError,
        match="Missing",
    ):
        parse_prior_config(
            {
                "type": "beta",
                "alpha": 2.0,
                "beta": 2.0,
                "lower": 0.0,
            }
        )


def test_wavelength_coupling_builds_from_observation_covariates():
    config = _multi_observation_config()

    config["observations"]["g"]["covariates"] = {
        "wavelength": {
            "value": 4770.0,
            "unit": "Angstrom",
        }
    }

    config["observations"]["r"]["covariates"] = {
        "wavelength": {
            "value": 6231.0,
            "unit": "Angstrom",
        }
    }

    parameters = (
        config["fit"]
        ["multi_observation"]
        ["parameters"]
    )

    parameters["re"] = {
        "coupling": "wavelength",
        "prior": {
            "type": "uniform",
            "lower": 0.5,
            "upper": 30.0,
        },
    }

    layout = build_multi_observation_parameter_layout(
        config
    )

    assert "re__rstar" in layout.parameter_set.names
    assert "re__z__g" in layout.parameter_set.names
    assert "re__z__r" in layout.parameter_set.names

    routed = layout.route(
        {
            name: 0.0
            for name in layout.parameter_set.names
        }
        | {
            "flux__g": 500.0,
            "flux__r": 900.0,
            "re__rstar": 0.5,
            "re__z__g": -0.5,
            "re__z__r": 0.5,
            "q": 0.7,
            "theta": 0.0,
            "x0": 15.0,
            "y0": 15.0,
            "background_level__r": 0.0,
        }
    )

    assert 0.5 <= routed["g"]["re"] <= 30.0
    assert 0.5 <= routed["r"]["re"] <= 30.0


def test_wavelength_coupling_requires_explicit_wavelength_metadata():
    config = _multi_observation_config()

    parameters = (
        config["fit"]
        ["multi_observation"]
        ["parameters"]
    )

    parameters["re"] = {
        "coupling": "wavelength",
        "prior": {
            "type": "uniform",
            "lower": 0.5,
            "upper": 30.0,
        },
    }

    with pytest.raises(
        ValueError,
        match="covariates",
    ):
        build_multi_observation_parameter_layout(
            config
        )


def test_observation_config_resolution_accepts_covariates(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_load_config(path):
        path = Path(path)
        calls.append(path)
        band = path.stem

        return {
            "object": {
                "name": f"target_{band}",
            },
            "data": {
                "image_path": f"data/{band}.fits",
            },
            "outputs": {
                "root": "runs",
            },
        }

    monkeypatch.setattr(
        "baymorph.pipeline.multi_observation_config.load_config",
        fake_load_config,
    )

    config = {
        "observations": {
            "g": {
                "config_path": "g.yaml",
                "covariates": {
                    "wavelength": {
                        "value": 4770.0,
                        "unit": "Angstrom",
                    },
                },
            },
            "r": {
                "config_path": "r.yaml",
                "covariates": {
                    "wavelength": {
                        "value": 6231.0,
                        "unit": "Angstrom",
                    },
                },
            },
        },
    }

    resolved = resolve_multi_observation_observation_configs(
        config,
        base_dir=tmp_path,
    )

    assert set(resolved) == {
        "g",
        "r",
    }
    assert resolved["g"]["object"]["name"] == "target_g"
    assert resolved["r"]["object"]["name"] == "target_r"
    assert calls == [
        tmp_path / "g.yaml",
        tmp_path / "r.yaml",
    ]


def test_joint_likelihood_rejects_estimator_settings():
    """AD-047 fixes the sky-correlation estimator of every observation."""
    config = _multi_observation_config()
    config["fit"]["likelihood"] = {
        "type": "correlated_gaussian",
        "calibration": {
            "max_lag": 20,
        },
    }

    with pytest.raises(
        ValueError,
        match="AD-047",
    ):
        resolve_multi_observation_observation_configs(
            config,
        )


def test_joint_likelihood_cannot_declare_a_common_gain():
    """AD-047 requires each observation to declare its own gain."""
    config = _multi_observation_config()
    config["fit"]["likelihood"] = {
        "type": "correlated_gaussian",
        "effective_gain": {
            "source": "measured_sky",
        },
    }

    with pytest.raises(
        ValueError,
        match="each observation",
    ):
        resolve_multi_observation_observation_configs(
            config,
        )


def test_joint_inference_rejects_a_declared_hdu(
    tmp_path: Path,
):
    """Joint inference reads images by path alone, so data.hdu is refused."""
    for band, data in (
        ("g", {"image_path": "data/g.fits"}),
        ("r", {"image_path": "data/r.fits", "hdu": "SCI"}),
    ):
        (tmp_path / f"{band}.yaml").write_text(
            yaml.safe_dump(
                {
                    "object": {"name": f"target_{band}"},
                    "data": data,
                    "outputs": {"root": "runs"},
                }
            )
        )

    with pytest.raises(ValueError, match="declares data.hdu"):
        resolve_multi_observation_observation_configs(
            _multi_observation_config(),
            base_dir=tmp_path,
        )
