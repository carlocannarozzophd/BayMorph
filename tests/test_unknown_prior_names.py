"""Configuration refuses priors under names that the model does not read."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from astropy.io import fits

from baymorph.cli import main
from baymorph.models import get_model
from baymorph.pipeline.config import (
    DEFAULT_CONFIG,
    validate_fit_priors,
    validate_prior_names,
)
from baymorph.pipeline.config_maker import make_config_from_fits
from baymorph.pipeline.prior_config import shared_prior_names, unknown_prior_names

FLUX_PRIOR = {"type": "uniform", "lower": 1.0, "upper": 1.0e5}

NARROW_PRIOR = {"type": "uniform", "lower": 1.0, "upper": 2.0}


def _config(model_name: str, **declared: dict) -> dict:
    """Return a configuration with the default priors, fluxes and declared priors."""
    priors = copy.deepcopy(DEFAULT_CONFIG["fit"]["priors"][model_name])

    for parameter_name in get_model(model_name).parameter_names:
        if parameter_name.startswith("flux"):
            priors[parameter_name] = dict(FLUX_PRIOR)

    priors.update(copy.deepcopy(declared))

    return {"fit": {"models": [model_name], "priors": {model_name: priors}}}


def _refusal(config: dict) -> str:
    """Return the message of the error that refuses the configuration."""
    with pytest.raises(ValueError) as error:
        validate_fit_priors(config)

    return str(error.value)


@pytest.mark.parametrize(
    ("model_name", "name"),
    [
        ("single_sersic", "sersic_index"),
        ("double_sersic", "n"),
        ("double_sersic", "re"),
        ("bulge_disk", "re_bulg"),
        ("bulge_disk", "n_disk"),
    ],
)
def test_unknown_prior_name_is_refused_with_the_parameters(model_name, name):
    message = _refusal(_config(model_name, **{name: dict(NARROW_PRIOR)}))

    assert f"fit.priors.{model_name}.{name} would have no effect" in message
    assert ", ".join(get_model(model_name).parameter_names) in message


def test_several_unknown_names_are_refused_in_one_error():
    message = _refusal(
        _config(
            "single_sersic",
            sersic_index=dict(NARROW_PRIOR),
            re_e=dict(NARROW_PRIOR),
        )
    )

    assert (
        "fit.priors.single_sersic.re_e and fit.priors.single_sersic.sersic_index"
        in message
    )
    assert "no parameters re_e and sersic_index" in message


def test_message_offers_a_shared_prior_only_where_it_applies():
    double = _refusal(_config("double_sersic", n=dict(NARROW_PRIOR)))
    bulge = _refusal(_config("bulge_disk", re=dict(NARROW_PRIOR)))

    assert "a prior under q or theta applies to both of its components" in double
    assert "applies to both" not in bulge


def test_shared_priors_of_double_sersic_are_still_accepted():
    validate_fit_priors(
        _config(
            "double_sersic",
            q={"type": "uniform", "lower": 0.5, "upper": 1.0},
            theta={"type": "uniform", "lower": -0.5, "upper": 0.5},
        )
    )


@pytest.mark.parametrize("model_name", sorted(DEFAULT_CONFIG["fit"]["priors"]))
def test_default_priors_use_only_names_that_the_model_reads(model_name):
    parameter_names = get_model(model_name).parameter_names

    assert unknown_prior_names(
        parameter_names, DEFAULT_CONFIG["fit"]["priors"][model_name]
    ) == ()

    validate_prior_names(model_name, DEFAULT_CONFIG["fit"]["priors"][model_name])


def test_shared_names_serve_only_component_models():
    assert shared_prior_names(get_model("single_sersic").parameter_names) == ()
    assert shared_prior_names(get_model("double_sersic").parameter_names) == (
        "q",
        "theta",
    )


def test_config_maker_refuses_an_unknown_prior_name(tmp_path):
    path = tmp_path / "galaxy.fits"
    output_path = tmp_path / "galaxy.yaml"

    fits.writeto(path, np.zeros((21, 21), dtype=float))

    with pytest.raises(ValueError, match="single_sersic.sersic_index"):
        make_config_from_fits(
            path,
            target_position=(10.0, 10.0),
            model_priors={
                "flux": dict(FLUX_PRIOR),
                "sersic_index": dict(NARROW_PRIOR),
            },
            output_path=output_path,
        )

    assert not output_path.exists()


def test_config_command_refuses_an_unknown_prior_name(tmp_path, monkeypatch, capsys):
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
            "sersic_index",
            "1",
            "2",
            "--output",
            str(output_path),
        ],
    )

    assert main() == 1
    assert "fit.priors.single_sersic.sersic_index" in capsys.readouterr().err
    assert not output_path.exists()
