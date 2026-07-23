"""Integration tests for a noiseless single-Sérsic Dynesty fitter."""

import numpy as np

from baymorph.fitting import (
    DynestyFitter,
    Parameter,
    ParameterSet,
)
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.models import SersicModel



def test_single_sersic_fitter_no_noise_evaluates_truth():
    """The fitter should evaluate the true noiseless model successfully."""
    shape = (31, 31)

    truth = {
        "flux": 1.0e4,
        "re": 4.0,
        "n": 2.0,
        "q": 0.7,
        "theta": np.deg2rad(25.0),
        "x0": 15.0,
        "y0": 15.0,
    }

    data = SersicModel(
        **truth,
    ).render(shape)

    noise = np.ones(
        shape,
        dtype=float,
    )

    mask = np.zeros(
        shape,
        dtype=bool,
    )

    likelihood = GaussianImageLikelihood(
        data=data,
        noise=noise,
        mask=mask,
    )

    parameters = ParameterSet(
        [
            Parameter("flux", 0.8e4, 1.2e4),
            Parameter("re", 3.0, 5.0),
            Parameter("n", 1.0, 3.0),
            Parameter("q", 0.5, 0.9),
            Parameter(
                "theta",
                -0.5 * np.pi,
                0.5 * np.pi,
            ),
            Parameter("x0", 14.0, 16.0),
            Parameter("y0", 14.0, 16.0),
        ]
    )

    def make_model(
        *,
        flux,
        re,
        n,
        q,
        theta,
        x0,
        y0,
    ):
        return SersicModel(
            flux=flux,
            re=re,
            n=n,
            q=q,
            theta=theta,
            x0=x0,
            y0=y0,
        )

    fitter = DynestyFitter(
        model_factory=make_model,
        parameters=parameters,
        likelihood=likelihood,
    )

    theta_truth = np.array(
        [
            truth[name]
            for name in fitter.parameter_names
        ],
        dtype=float,
    )

    logl_truth = fitter.loglikelihood(theta_truth)

    offset_parameters = dict(truth)
    offset_parameters["x0"] += 1.0

    theta_offset = np.array(
        [
            offset_parameters[name]
            for name in fitter.parameter_names
        ],
        dtype=float,
    )

    logl_offset = fitter.loglikelihood(theta_offset)

    assert np.isfinite(logl_truth)
    assert np.isfinite(logl_offset)
    assert logl_truth > logl_offset