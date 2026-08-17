"""Smoke test for joint multi-band inference with Dynesty."""

import numpy as np

from baymorph.fitting import make_sampler
from baymorph.fitting.multi_observation import (
    MultiObservationParameterLayout,
    MultiObservationParameterSpec,
)
from baymorph.fitting.multi_observation_problem import MultiObservationInferenceProblem
from baymorph.fitting.parameters import Parameter
from baymorph.fitting.representations import RepresentationFactory
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.likelihood.multi_observation import MultiObservationLikelihood
from baymorph.models import SersicModel


def make_model(
    *,
    flux,
    re,
):
    """Build one Sérsic representation."""
    return SersicModel(
        flux=flux,
        re=re,
        n=2.0,
        q=0.8,
        theta=0.0,
        x0=7.0,
        y0=7.0,
    )


def test_dynesty_runs_multi_observation_problem():
    """Dynesty should sample one joint multi-band posterior."""
    shape = (15, 15)

    data_g = make_model(
        flux=500.0,
        re=4.0,
    ).render(shape=shape)

    data_r = make_model(
        flux=900.0,
        re=4.0,
    ).render(shape=shape)

    layout = MultiObservationParameterLayout(
        observations=["g", "r"],
        specifications=[
            MultiObservationParameterSpec.shared_parameter(
                Parameter(
                    name="re",
                    lower=2.0,
                    upper=8.0,
                )
            ),
            MultiObservationParameterSpec.independent(
                name="flux",
                parameters={
                    "g": Parameter(
                        name="flux",
                        lower=100.0,
                        upper=1000.0,
                    ),
                    "r": Parameter(
                        name="flux",
                        lower=100.0,
                        upper=1500.0,
                    ),
                },
            ),
        ],
    )

    likelihood = MultiObservationLikelihood(
        {
            "g": GaussianImageLikelihood(
                data=data_g,
                noise=np.ones(shape),
            ),
            "r": GaussianImageLikelihood(
                data=data_r,
                noise=np.ones(shape),
            ),
        }
    )

    representation_factory = RepresentationFactory(
        factory=make_model,
        fixed_kwargs={
            "g": {},
            "r": {},
        },
    )

    problem = MultiObservationInferenceProblem(
        representation_factory=representation_factory,
        parameter_layout=layout,
        likelihood=likelihood,
    )

    backend = make_sampler(
        {
            "name": "dynesty",
            "mode": "static",
            "init": {
                "nlive": 20,
                "rstate": np.random.default_rng(
                    123
                ),
            },
            "execution": {
                "processes": 1,
                "progress": False,
            },
            "run": {
                "dlogz": 5.0,
            },
        },
        problem=problem,
    )

    result = backend.fit()

    assert result.samples.ndim == 2
    assert result.samples.shape[1] == 3
    assert result.parameter_names == [
        "re",
        "flux__g",
        "flux__r",
    ]
    assert np.isfinite(result.logz)
    assert np.isfinite(result.logz_error)
