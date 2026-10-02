"""Configuration keeps the families of AD-048 within their validated domains."""

from __future__ import annotations

import copy

import pytest

from baymorph.models import get_model
from baymorph.models.validated_domains import VALIDATED_DOMAINS
from baymorph.pipeline.config import DEFAULT_CONFIG, validate_fit_priors

BOUNDS = [
    (model_name, parameter_name, side, bound)
    for model_name, domain in sorted(VALIDATED_DOMAINS.items())
    for parameter_name, bounds in sorted(domain.items())
    for side, bound in zip(("lower", "upper"), bounds, strict=True)
    if bound is not None
]


def _config(model_name: str) -> dict:
    """Return a configuration with the default priors and declared fluxes."""
    priors = copy.deepcopy(DEFAULT_CONFIG["fit"]["priors"][model_name])

    for parameter_name in get_model(model_name).parameter_names:
        if parameter_name.startswith("flux"):
            priors[parameter_name] = {
                "type": "uniform",
                "lower": 1.0,
                "upper": 1.0e5,
            }

    return {"fit": {"models": [model_name], "priors": {model_name: priors}}}


@pytest.mark.parametrize("model_name", sorted(VALIDATED_DOMAINS))
def test_default_priors_lie_in_the_validated_domain(model_name):
    validate_fit_priors(_config(model_name))


@pytest.mark.parametrize(
    ("model_name", "parameter_name", "side", "bound"),
    BOUNDS,
)
def test_priors_beyond_the_validated_domain_are_rejected(
    model_name,
    parameter_name,
    side,
    bound,
):
    config = _config(model_name)
    prior = config["fit"]["priors"][model_name][parameter_name]
    step = 0.05 * max(abs(bound), 1.0)
    prior[side] = bound - step if side == "lower" else bound + step

    with pytest.raises(ValueError, match="is validated"):
        validate_fit_priors(config)
