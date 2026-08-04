"""Smoke test for joint multi-band inference with Dynesty."""

import numpy as np

from baymorph.fitting import make_sampler
from baymorph.fitting.multiband import (
    MultiBandParameterLayout,
    MultiBandParameterSpec,
)
from baymorph.fitting.multiband_problem import MultiBandInferenceProblem
from baymorph.fitting.parameters import Parameter
from baymorph.fitting.representations import RepresentationFactory
from baymorph.likelihood import GaussianImageLikelihood
from baymorph.likelihood.multiband import MultiBandLikelihood
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


def test_dynesty_runs_multiband_problem():
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

    layout = MultiBandParameterLayout(
        bands=["g", "r"],
        specifications=[
            MultiBandParameterSpec.shared_parameter(
                Parameter(
                    name="re",
                    lower=2.0,
                    upper=8.0,
                )
            ),
            MultiBandParameterSpec.band_specific(
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

    likelihood = MultiBandLikelihood(
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

    problem = MultiBandInferenceProblem(
        representation_factory=representation_factory,
        parameter_layout=layout,
        likelihood=likelihood,
    )

    sampler = make_sampler(
        "dynesty",
        problem=problem,
    )

    result = sampler.run(
        nlive=20,
        dlogz=5.0,
        print_progress=False,
        seed=123,
    )

    assert result.samples.ndim == 2
    assert result.samples.shape[1] == 3
    assert result.parameter_names == [
        "re",
        "flux__g",
        "flux__r",
    ]
    assert np.isfinite(result.logz[-1])
    assert np.isfinite(result.logzerr[-1])
