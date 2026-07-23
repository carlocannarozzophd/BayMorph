"""Tests for ParameterSet integration with Dynesty."""

import numpy as np

from baymorph.core import make_constant_noise_map
from baymorph.fitting import (
    DynestyFitter,
    Parameter,
    ParameterSet,
)
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.models import SersicModel



def test_dynesty_accepts_parameter_set():
    """DynestyFitter should accept a ParameterSet."""
    data = SersicModel(
        flux=1000.0,
        re=8.0,
        n=2.0,
        q=0.8,
        theta=0.0,
        x0=30.0,
        y0=30.0,
    ).render(
        shape=(60, 60),
    )

    noise = make_constant_noise_map(
        data.shape,
        sigma=1.0,
    )

    likelihood = GaussianImageLikelihood(
        data=data,
        noise=noise,
    )

    parameters = ParameterSet(
        [
            Parameter(
                "flux",
                100.0,
                2000.0,
            ),
        ]
    )

    def model_factory(
        *,
        flux,
    ):
        return SersicModel(
            flux=flux,
            re=8.0,
            n=2.0,
            q=0.8,
            theta=0.0,
            x0=30.0,
            y0=30.0,
        )

    fitter = DynestyFitter(
        model_factory=model_factory,
        parameters=parameters,
        likelihood=likelihood,
    )

    assert fitter.ndim == 1
    assert fitter.parameter_names == ["flux"]

    transformed = fitter.prior_transform(
        np.array([0.5])
    )

    assert np.isclose(
        transformed[0],
        1050.0,
    )