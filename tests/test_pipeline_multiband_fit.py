"""Tests for the production multi-band fitting workflow."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from baymorph.fitting.multiband import (
    MultiBandParameterLayout,
    MultiBandParameterSpec,
)
from baymorph.fitting.parameters import Parameter
from baymorph.pipeline import multiband_fit


class DummyBackend:
    """Minimal backend recording the received fit request."""

    def __init__(
        self,
        result,
    ):
        self.result = result
        self.fit_kwargs = None

    def fit(
        self,
        **kwargs,
    ):
        self.fit_kwargs = kwargs
        return self.result


def _layout():
    return MultiBandParameterLayout(
        bands=(
            "g",
            "r",
        ),
        specifications=[
            MultiBandParameterSpec.shared_parameter(
                Parameter(
                    "re",
                    lower=1.0,
                    upper=10.0,
                )
            ),
            MultiBandParameterSpec.independent(
                "flux",
                {
                    "g": Parameter(
                        "flux",
                        lower=10.0,
                        upper=100.0,
                    ),
                    "r": Parameter(
                        "flux",
                        lower=20.0,
                        upper=200.0,
                    ),
                },
            ),
            MultiBandParameterSpec.shared_parameter(
                Parameter(
                    "n",
                    lower=0.5,
                    upper=6.0,
                )
            ),
            MultiBandParameterSpec.shared_parameter(
                Parameter(
                    "q",
                    lower=0.2,
                    upper=1.0,
                )
            ),
            MultiBandParameterSpec.shared_parameter(
                Parameter(
                    "theta",
                    lower=-1.57,
                    upper=1.57,
                )
            ),
            MultiBandParameterSpec.shared_parameter(
                Parameter(
                    "x0",
                    lower=2.0,
                    upper=8.0,
                )
            ),
            MultiBandParameterSpec.shared_parameter(
                Parameter(
                    "y0",
                    lower=2.0,
                    upper=8.0,
                )
            ),
            MultiBandParameterSpec.independent(
                "sky",
                {
                    "g": Parameter(
                        "sky",
                        lower=-5.0,
                        upper=5.0,
                    ),
                    "r": Parameter(
                        "sky",
                        lower=-5.0,
                        upper=5.0,
                    ),
                },
            ),
        ],
    )


def _products(
    shape=(11, 11),
):
    return SimpleNamespace(
        prepared_data=np.zeros(
            shape,
            dtype=float,
        ),
        noise=np.ones(
            shape,
            dtype=float,
        ),
        fitting_mask=np.zeros(
            shape,
            dtype=bool,
        ),
    )


def _config():
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
            "likelihood": {
                "type": "gaussian",
            },
            "sampler": {
                "name": "dynesty",
                "nlive": 50,
                "sample": "auto",
                "seed": 4,
                "n_processes": 1,
            },
            "multiband": {
                "parameters": {},
            },
        },
    }


def test_run_multiband_fit_builds_problem_and_returns_result(
    monkeypatch,
):
    expected_result = object()
    backend = DummyBackend(
        expected_result
    )

    observation_configs = {
        "g": {
            "fit": {},
        },
        "r": {
            "fit": {},
        },
    }

    monkeypatch.setattr(
        multiband_fit,
        "resolve_multiband_config",
        lambda config, base_dir: (
            observation_configs,
            _layout(),
        ),
    )

    monkeypatch.setattr(
        multiband_fit,
        "load_image_preparation_products",
        lambda config: _products(),
    )

    monkeypatch.setattr(
        multiband_fit,
        "_load_psf_from_config",
        lambda config: np.ones(
            (3, 3),
            dtype=float,
        )
        / 9.0,
    )

    captured = {}

    def fake_make_sampler(
        config,
        *,
        problem,
    ):
        captured["config"] = config
        captured["problem"] = problem
        return backend

    monkeypatch.setattr(
        multiband_fit,
        "make_sampler",
        fake_make_sampler,
    )

    result = (
        multiband_fit
        .run_multiband_fit_from_config(
            _config()
        )
    )

    assert result is expected_result
    assert captured["problem"].parameter_layout.bands == (
        "g",
        "r",
    )
    assert captured["problem"].likelihood.bands == (
        "g",
        "r",
    )
    assert captured["problem"].representation_factory.observations == (
        "g",
        "r",
    )

    assert backend.fit_kwargs == {
        "progress_label": "multi_band_single_sersic",
    }


def test_run_multiband_fit_supports_different_observation_shapes(
    monkeypatch,
):
    observation_configs = {
        "g": {},
        "r": {},
    }

    monkeypatch.setattr(
        multiband_fit,
        "resolve_multiband_config",
        lambda config, base_dir: (
            observation_configs,
            _layout(),
        ),
    )

    products = {
        id(
            observation_configs["g"]
        ): _products(
            (11, 11)
        ),
        id(
            observation_configs["r"]
        ): _products(
            (13, 17)
        ),
    }

    monkeypatch.setattr(
        multiband_fit,
        "load_image_preparation_products",
        lambda config: products[
            id(
                config
            )
        ],
    )

    monkeypatch.setattr(
        multiband_fit,
        "_load_psf_from_config",
        lambda config: np.ones(
            (3, 3),
            dtype=float,
        )
        / 9.0,
    )

    expected_result = object()
    backend = DummyBackend(
        expected_result
    )
    captured = {}

    def fake_make_sampler(
        config,
        *,
        problem,
    ):
        captured["problem"] = problem
        return backend

    monkeypatch.setattr(
        multiband_fit,
        "make_sampler",
        fake_make_sampler,
    )

    result = (
        multiband_fit
        .run_multiband_fit_from_config(
            _config()
        )
    )

    assert result is expected_result

    problem = captured["problem"]

    assert problem.likelihood.image_shapes == {
        "g": (11, 11),
        "r": (13, 17),
    }
