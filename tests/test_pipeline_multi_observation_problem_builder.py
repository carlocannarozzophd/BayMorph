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
