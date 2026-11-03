"""A declared prior replaces the default prior of its parameter whole."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

from baymorph.cli import _model_priors_from_args, main
from baymorph.fitting.priors import BetaPrior, NormalPrior, UniformPrior
from baymorph.pipeline.config import DEFAULT_CONFIG, resolve_config, validate_fit_priors
from baymorph.pipeline.prior_config import parse_prior_config
from baymorph.pipeline.real_fit import _build_parameter_set, run_real_fit_from_config

FLUX_PRIOR = {"type": "uniform", "lower": 1.0, "upper": 1.0e5}

REFERENCES = {"target_x": 4.0, "target_y": 4.0}


def _config(tmp_path: Path, model_name: str = "single_sersic", **fit: dict) -> dict:
    """Return a minimal configuration of one model with the given fit entries."""
    return {
        "object": {"name": "declared_prior_test"},
        "data": {
            "image_path": str(tmp_path / "image.fits"),
            "target_position": {"x": 4.0, "y": 4.0},
            "cutout": {"mode": "fixed", "size": 9},
        },
        "outputs": {"root": str(tmp_path / "runs")},
        "fit": {"models": [model_name], **copy.deepcopy(fit)},
    }


def _single_sersic(tmp_path: Path, background: dict | None = None, **priors) -> dict:
    """Return a single-Sérsic configuration with a flux prior and ``priors``."""
    fit = {"priors": {"single_sersic": {"flux": dict(FLUX_PRIOR), **priors}}}

    if background is not None:
        fit["background"] = background

    return _config(tmp_path, **fit)


def test_declared_beta_prior_keeps_its_own_support(tmp_path):
    beta = {"type": "beta", "alpha": 2, "beta": 2}

    resolved = resolve_config(_single_sersic(tmp_path, q=beta))
    validate_fit_priors(resolved)

    prior = resolved["fit"]["priors"]["single_sersic"]["q"]

    assert prior == beta
    assert parse_prior_config(prior) == BetaPrior(
        alpha=2.0, beta=2.0, lower=0.0, upper=1.0
    )


def test_declared_shared_beta_prior_of_double_sersic_keeps_its_support(tmp_path):
    beta = {"type": "beta", "alpha": 2, "beta": 5}
    priors = {"flux_bulge": dict(FLUX_PRIOR), "flux_disk": dict(FLUX_PRIOR), "q": beta}

    resolved = resolve_config(
        _config(tmp_path, "double_sersic", priors={"double_sersic": priors})
    )
    validate_fit_priors(resolved)

    assert resolved["fit"]["priors"]["double_sersic"]["q"] == beta


def test_declared_normal_priors_reach_the_parameters_of_the_fit(tmp_path):
    re_prior = {"type": "normal", "mean": 10.0, "sigma": 3.0}
    level_prior = {"type": "normal", "mean": 0.0, "sigma": 1.0}

    resolved = resolve_config(
        _single_sersic(
            tmp_path,
            background={"priors": {"background_level": level_prior}},
            re=re_prior,
        )
    )
    validate_fit_priors(resolved)

    assert resolved["fit"]["priors"]["single_sersic"]["re"] == re_prior
    assert resolved["fit"]["background"]["priors"]["background_level"] == level_prior

    names, parameters = _build_parameter_set(
        "single_sersic",
        prior_cfg=resolved["fit"]["priors"],
        background_cfg=resolved["fit"]["background"],
        references=REFERENCES,
    )
    priors = dict(zip(names, (p.prior for p in parameters.all_parameters)))

    assert priors["re"] == NormalPrior(mean=10.0, sigma=3.0)
    assert priors["background_level"] == NormalPrior(mean=0.0, sigma=1.0)


def test_centre_prior_without_relative_to_is_absolute(tmp_path):
    x0 = {"type": "uniform", "lower": 500.0, "upper": 520.0}

    resolved = resolve_config(_single_sersic(tmp_path, x0=x0))

    prior = resolved["fit"]["priors"]["single_sersic"]["x0"]

    assert prior == x0
    assert parse_prior_config(prior, references=REFERENCES) == UniformPrior(
        lower=500.0, upper=520.0
    )


def test_undeclared_parameters_keep_their_defaults(tmp_path):
    resolved = resolve_config(
        _single_sersic(tmp_path, q={"type": "beta", "alpha": 2, "beta": 2})
    )

    default_model = DEFAULT_CONFIG["fit"]["priors"]["single_sersic"]
    resolved_model = resolved["fit"]["priors"]["single_sersic"]

    for name, prior in default_model.items():
        if name != "q":
            assert resolved_model[name] == prior

    assert (
        resolved["fit"]["background"]["priors"]
        == DEFAULT_CONFIG["fit"]["background"]["priors"]
    )
    assert (
        resolved["fit"]["priors"]["double_sersic"]
        == DEFAULT_CONFIG["fit"]["priors"]["double_sersic"]
    )


def test_resolution_does_not_change_the_defaults_or_the_configuration(tmp_path):
    defaults = copy.deepcopy(DEFAULT_CONFIG)
    config = _single_sersic(tmp_path, q={"type": "beta", "alpha": 2, "beta": 2})
    declared = copy.deepcopy(config)

    resolved = resolve_config(config)
    resolved["fit"]["priors"]["single_sersic"]["q"]["alpha"] = 9

    assert DEFAULT_CONFIG == defaults
    assert config == declared


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("model", "single_sersic.n"),
        ("background", "background.background_level"),
    ],
)
def test_fixed_prior_is_refused_before_any_product(tmp_path, location, expected):
    fixed = {"type": "fixed", "value": 4.0}

    if location == "model":
        config = _single_sersic(tmp_path, n=fixed)
    else:
        config = _single_sersic(
            tmp_path, background={"priors": {"background_level": fixed}}
        )

    with pytest.raises(ValueError, match="does not yet support fixed priors") as error:
        run_real_fit_from_config(config)

    assert expected in str(error.value)
    assert list(tmp_path.iterdir()) == []


def test_command_line_centre_priors_are_about_the_target():
    args = argparse.Namespace(
        uniform_prior=[["x0", "-3", "3"], ["y0", "-2", "2"], ["re", "1", "50"]]
    )

    priors = _model_priors_from_args(args, ["single_sersic"])["single_sersic"]

    assert priors["x0"] == {
        "type": "uniform",
        "lower": -3.0,
        "upper": 3.0,
        "relative_to": "target_x",
    }
    assert priors["y0"]["relative_to"] == "target_y"
    assert "relative_to" not in priors["re"]


def test_config_command_writes_a_centre_prior_about_the_target(tmp_path, monkeypatch):
    fits_path = tmp_path / "galaxy.fits"
    output_path = tmp_path / "galaxy.yaml"

    fits.writeto(fits_path, np.zeros((21, 21), dtype=float))

    monkeypatch.setattr(
        "sys.argv",
        [
            "baymorph",
            "config",
            str(fits_path),
            "--target-center",
            "--uniform-prior",
            "flux",
            "1",
            "1000",
            "--uniform-prior",
            "x0",
            "-3",
            "3",
            "--output",
            str(output_path),
        ],
    )

    assert main() == 0

    config = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert config["fit"]["priors"]["single_sersic"]["x0"] == {
        "type": "uniform",
        "lower": -3.0,
        "upper": 3.0,
        "relative_to": "target_x",
    }
