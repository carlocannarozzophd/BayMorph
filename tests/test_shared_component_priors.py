"""Configuration refuses shared q and theta priors that would have no effect."""

from __future__ import annotations

import copy

import pytest

from baymorph.models import get_model
from baymorph.pipeline.config import DEFAULT_CONFIG, resolve_config, validate_fit_priors
from baymorph.pipeline.prior_config import (
    ineffective_shared_priors,
    prior_config_for_parameter,
)

BULGE_MODELS = (
    "bulge_disk",
    "bulge_broken_disk",
    "bulge_disk_bar",
    "bulge_disk_ring",
    "bulge_disk_spiral",
)

FLUX_PRIOR = {"type": "uniform", "lower": 1.0, "upper": 1.0e5}

SHARED_PRIORS = {
    "q": {"type": "uniform", "lower": 0.9, "upper": 1.0},
    "theta": {"type": "uniform", "lower": -0.5, "upper": 0.5},
}

COMPONENT_Q_PRIOR = {"type": "uniform", "lower": 0.5, "upper": 1.0}

SERVED = {
    "q": ("q_bulge", "q_disk"),
    "theta": ("theta_bulge", "theta_disk"),
}

CASES = [(model_name, name) for model_name in BULGE_MODELS for name in SHARED_PRIORS]


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


@pytest.mark.parametrize(("model_name", "name"), CASES)
def test_shared_prior_is_refused_where_component_defaults_win(model_name, name):
    message = _refusal(_config(model_name, **{name: SHARED_PRIORS[name]}))

    assert f"fit.priors.{model_name}.{name}" in message

    for served in SERVED[name]:
        assert served in message


def test_both_shared_priors_are_refused_in_one_error():
    message = _refusal(_config("bulge_disk", **SHARED_PRIORS))

    q_part, theta_part = message.split("fit.priors.bulge_disk.theta")

    assert "fit.priors.bulge_disk.q" in q_part
    assert all(served in q_part for served in SERVED["q"])
    assert all(served in theta_part for served in SERVED["theta"])


def test_component_priors_without_shared_prior_are_accepted():
    validate_fit_priors(
        _config("bulge_disk", q_bulge=COMPONENT_Q_PRIOR, q_disk=COMPONENT_Q_PRIOR)
    )


def test_double_sersic_shared_prior_applies_to_both_components():
    config = _config("double_sersic", q=SHARED_PRIORS["q"])
    priors = config["fit"]["priors"]["double_sersic"]

    validate_fit_priors(config)

    assert prior_config_for_parameter("q_bulge", priors) == SHARED_PRIORS["q"]
    assert prior_config_for_parameter("q_disk", priors) == SHARED_PRIORS["q"]


def test_double_sersic_component_priors_take_precedence_over_shared_prior():
    config = _config(
        "double_sersic",
        q=SHARED_PRIORS["q"],
        q_bulge=COMPONENT_Q_PRIOR,
        q_disk=COMPONENT_Q_PRIOR,
    )
    priors = config["fit"]["priors"]["double_sersic"]

    validate_fit_priors(config)

    assert prior_config_for_parameter("q_bulge", priors) == COMPONENT_Q_PRIOR
    assert prior_config_for_parameter("q_disk", priors) == COMPONENT_Q_PRIOR


def test_resolved_configuration_with_shared_prior_is_refused(tmp_path):
    config = {
        "object": {"name": "shared_prior_test"},
        "data": {
            "image_path": str(tmp_path / "image.fits"),
            "target_position": {"x": 4.0, "y": 4.0},
            "cutout": {"mode": "fixed", "size": 9},
        },
        "outputs": {"root": str(tmp_path / "runs")},
        "fit": {
            "models": ["bulge_disk"],
            "priors": {
                "bulge_disk": {
                    "flux_bulge": dict(FLUX_PRIOR),
                    "flux_disk": dict(FLUX_PRIOR),
                    "q": dict(SHARED_PRIORS["q"]),
                }
            },
        },
    }

    message = _refusal(resolve_config(config))

    assert "fit.priors.bulge_disk.q" in message
    assert all(served in message for served in SERVED["q"])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", list(SHARED_PRIORS))
def test_only_the_bulge_models_ignore_a_shared_prior(name):
    flagged = set()

    for model_name, defaults in DEFAULT_CONFIG["fit"]["priors"].items():
        priors = {**defaults, name: SHARED_PRIORS[name]}
        parameter_names = get_model(model_name).parameter_names

        if name in ineffective_shared_priors(parameter_names, priors, defaults):
            flagged.add(model_name)

    assert flagged == set(BULGE_MODELS)
