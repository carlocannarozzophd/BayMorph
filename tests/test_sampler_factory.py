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


def test_make_sampler_builds_native_fitter():
    from baymorph.fitting import NativeFitter

    model_factory, parameters, likelihood = _make_inputs()

    sampler = make_sampler(
        {
            "name": "native",
            "nlive": 50,
            "dlogz": 0.1,
        },
        model_factory=model_factory,
        parameters=parameters,
        likelihood=likelihood,
    )

    assert isinstance(
        sampler,
        NativeFitter,
    )
    assert sampler.model_factory is model_factory
    assert sampler.parameters is parameters
    assert sampler.likelihood is likelihood
    assert sampler.config["nlive"] == 50


def test_native_geometry_uses_morphology_structure():
    from baymorph.fitting.native import make_morphology_geometry

    parameters = ParameterSet(
        [
            Parameter(
                "flux_bulge",
                1.0,
                2.0,
            ),
            Parameter(
                "re_bulge",
                0.5,
                3.0,
            ),
            Parameter(
                "theta_bulge",
                0.0,
                float(np.pi),
            ),
            Parameter(
                "x0",
                0.0,
                2.0,
            ),
            Parameter(
                "sky",
                -1.0,
                1.0,
            ),
        ]
    )

    geometry = make_morphology_geometry(
        parameters
    )

    assert geometry.names == parameters.names
    assert (
        geometry.geometry("flux_bulge").block
        == "photometry_bulge"
    )
    assert (
        geometry.geometry("re_bulge").block
        == "structure_bulge"
    )
    assert (
        geometry.geometry("theta_bulge").block
        == "shape_bulge"
    )
    assert geometry.geometry(
        "theta_bulge"
    ).period == pytest.approx(
        np.pi
    )
    assert geometry.geometry(
        "x0"
    ).block == "position"
    assert geometry.geometry(
        "sky"
    ).block == "background"


def test_native_geometry_rejects_unknown_parameters():
    from baymorph.fitting.native import make_morphology_geometry

    parameters = ParameterSet(
        [
            Parameter(
                "unsupported_parameter",
                0.0,
                1.0,
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="no morphology geometry",
    ):
        make_morphology_geometry(
            parameters
        )
