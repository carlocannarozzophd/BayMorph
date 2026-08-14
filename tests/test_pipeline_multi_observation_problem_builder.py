"""Tests for deterministic multi-observation problem construction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from baymorph.fitting.multi_observation import (
    MultiObservationParameterLayout,
    MultiObservationParameterSpec,
)
from baymorph.fitting.multi_observation_problem import (
    MultiObservationInferenceProblem,
)
from baymorph.fitting.parameters import Parameter
from baymorph.pipeline import multi_observation_fit


def _layout():
    return MultiObservationParameterLayout(
        observations=[
            "g",
            "r",
        ],
        specifications=[
            MultiObservationParameterSpec.shared_parameter(
                Parameter(
                    "re",
                    lower=1.0,
                    upper=10.0,
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
        ],
    )


def _products():
    return SimpleNamespace(
        prepared_data=np.zeros(
            (15, 15),
            dtype=float,
        ),
        noise=np.ones(
            (15, 15),
            dtype=float,
        ),
        fitting_mask=np.zeros(
            (15, 15),
            dtype=bool,
        ),
        cutout_bounds={
            "xmin": 0,
            "xmax": 15,
            "ymin": 0,
            "ymax": 15,
        },
    )


def test_build_multi_observation_problem_does_not_run_sampler(
    monkeypatch,
):
    config = {
        "fit": {
            "models": [
                "single_sersic",
            ],
            "multi_observation": {
                "parameter_frame": "detector",
            },
        },
    }

    observation_configs = {
        "g": {},
        "r": {},
    }

    monkeypatch.setattr(
        multi_observation_fit,
        "resolve_multi_observation_config",
        lambda raw_config, base_dir: (
            observation_configs,
            _layout(),
        ),
    )

    monkeypatch.setattr(
        multi_observation_fit,
        "load_image_preparation_products",
        lambda observation_config: _products(),
    )

    monkeypatch.setattr(
        multi_observation_fit,
        "_load_psf_from_config",
        lambda observation_config: object(),
    )

    monkeypatch.setattr(
        multi_observation_fit,
        "_resolve_model_factory",
        lambda model_name: (
            lambda **parameters: parameters
        ),
    )

    def fail_if_sampler_is_built(*args, **kwargs):
        raise AssertionError(
            "Problem construction must not instantiate a sampler."
        )

    monkeypatch.setattr(
        multi_observation_fit,
        "make_sampler",
        fail_if_sampler_is_built,
    )

    problem = (
        multi_observation_fit
        .build_multi_observation_problem_from_config(
            config
        )
    )

    assert isinstance(
        problem,
        MultiObservationInferenceProblem,
    )

    assert problem.parameter_layout.observations == (
        "g",
        "r",
    )

    assert problem.likelihood.observations == (
        "g",
        "r",
    )

    assert problem.representation_factory.observations == (
        "g",
        "r",
    )


def test_production_problem_supports_hierarchical_effective_radius(
    monkeypatch,
):
    """Production construction must preserve hierarchical radius coupling."""
    from math import exp, log

    import numpy as np
    import pytest

    from baymorph.fitting.multi_observation_problem import (
        MultiObservationInferenceProblem,
    )
    from baymorph.pipeline.multi_observation_config import (
        build_multi_observation_parameter_layout,
    )

    config = {
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
                "parameter_frame": "detector",
                "parameters": {
                    "flux": {
                        "coupling": "independent",
                        "priors": {
                            "g": {
                                "type": "fixed",
                                "value": 500.0,
                            },
                            "r": {
                                "type": "fixed",
                                "value": 900.0,
                            },
                        },
                    },
                    "re": {
                        "coupling": "hierarchical_log_normal",
                        "location_prior": {
                            "type": "uniform",
                            "lower": log(5.0) - 0.1,
                            "upper": log(5.0) + 0.1,
                        },
                        "scale_prior": {
                            "type": "uniform",
                            "lower": 0.0,
                            "upper": 0.4,
                        },
                    },
                    "n": {
                        "coupling": "shared",
                        "prior": {
                            "type": "fixed",
                            "value": 2.0,
                        },
                    },
                    "q": {
                        "coupling": "shared",
                        "prior": {
                            "type": "fixed",
                            "value": 0.7,
                        },
                    },
                    "theta": {
                        "coupling": "shared",
                        "prior": {
                            "type": "fixed",
                            "value": 0.1,
                        },
                    },
                    "x0": {
                        "coupling": "shared",
                        "prior": {
                            "type": "fixed",
                            "value": 5.0,
                        },
                    },
                    "y0": {
                        "coupling": "shared",
                        "prior": {
                            "type": "fixed",
                            "value": 5.0,
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
                                "type": "fixed",
                                "value": 0.0,
                            },
                        },
                    },
                },
            },
        },
    }

    observation_configs = {
        "g": {},
        "r": {},
    }

    def resolve_config(
        raw_config,
        base_dir,
    ):
        del base_dir
        return (
            observation_configs,
            build_multi_observation_parameter_layout(
                raw_config
            ),
        )

    monkeypatch.setattr(
        multi_observation_fit,
        "resolve_multi_observation_config",
        resolve_config,
    )

    monkeypatch.setattr(
        multi_observation_fit,
        "load_image_preparation_products",
        lambda observation_config: _products(),
    )

    monkeypatch.setattr(
        multi_observation_fit,
        "_load_psf_from_config",
        lambda observation_config: object(),
    )

    monkeypatch.setattr(
        multi_observation_fit,
        "_resolve_model_factory",
        lambda model_name: (
            lambda **parameters: parameters
        ),
    )

    problem = (
        multi_observation_fit
        .build_multi_observation_problem_from_config(
            config
        )
    )

    assert isinstance(
        problem,
        MultiObservationInferenceProblem,
    )

    assert problem.parameter_names == [
        "re__mu_log",
        "re__tau_log",
        "re__z__g",
        "re__z__r",
    ]

    cube = np.asarray(
        [
            0.5,
            0.5,
            0.15865525393145707,
            0.8413447460685429,
        ],
        dtype=float,
    )

    theta = problem.prior_transform(
        cube
    )

    assert theta[0] == pytest.approx(
        log(5.0)
    )
    assert theta[1] == pytest.approx(
        0.2
    )
    assert theta[2] == pytest.approx(
        -1.0
    )
    assert theta[3] == pytest.approx(
        1.0
    )

    routed = problem.routed_parameters(
        theta
    )

    assert routed["g"]["re"] == pytest.approx(
        5.0 * exp(-0.2)
    )
    assert routed["r"]["re"] == pytest.approx(
        5.0 * exp(0.2)
    )

    assert routed["g"]["re"] > 0.0
    assert routed["r"]["re"] > 0.0
    assert routed["g"]["re"] != pytest.approx(
        routed["r"]["re"]
    )

    assert routed["g"]["flux"] == pytest.approx(
        500.0
    )
    assert routed["r"]["flux"] == pytest.approx(
        900.0
    )
