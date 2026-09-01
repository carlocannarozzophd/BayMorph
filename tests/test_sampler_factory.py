"""Tests for the sampling-backend factory."""

import numpy as np
import pytest

from baymorph.fitting import (
    DynestyFitter,
    PocoMCFitter,
    make_sampler,
    Parameter,
    ParameterSet,
)
from baymorph.likelihood import GaussianImageLikelihood



def _make_inputs():
    data = np.ones((3, 3), dtype=float)
    likelihood = GaussianImageLikelihood(
        data=data,
        noise=np.ones_like(data),
    )
    parameters = ParameterSet(
        [
            Parameter(
                "flux",
                1.0,
                2.0,
            )
        ]
    )

    def model_factory(*, flux):
        del flux
        return None

    return model_factory, parameters, likelihood


def test_make_sampler_builds_dynesty_fitter():
    model_factory, parameters, likelihood = _make_inputs()

    sampler = make_sampler(
        {
            "name": "dynesty",
            "mode": "static",
            "init": {
                "nlive": 100,
            },
            "run": {},
        },
        model_factory=model_factory,
        parameters=parameters,
        likelihood=likelihood,
    )

    assert isinstance(sampler, DynestyFitter)
    assert sampler.model_factory is model_factory
    assert sampler.parameters is parameters
    assert sampler.likelihood is likelihood


def test_make_sampler_defaults_to_pocomc():
    model_factory, parameters, likelihood = _make_inputs()

    sampler = make_sampler(
        None,
        model_factory=model_factory,
        parameters=parameters,
        likelihood=likelihood,
    )

    assert isinstance(sampler, PocoMCFitter)


def test_make_sampler_rejects_unknown_backend():
    model_factory, parameters, likelihood = _make_inputs()

    with pytest.raises(
        ValueError,
        match="Unknown sampler",
    ):
        make_sampler(
            "nautilus",
            model_factory=model_factory,
            parameters=parameters,
            likelihood=likelihood,
        )
