"""Synthetic recovery test for restricted multi-band inference."""

from __future__ import annotations

import numpy as np
import pytest

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
    flux: float,
    re: float,
) -> SersicModel:
    """Build one Sérsic representation."""
    return SersicModel(
        flux=flux,
        re=re,
        n=2.0,
        q=0.8,
        theta=0.0,
        x0=10.0,
        y0=10.0,
    )


@pytest.mark.slow
def test_multiband_posterior_recovers_shared_radius_and_local_fluxes():
    """Joint inference should recover a controlled two-band Sérsic model."""
    rng = np.random.default_rng(20260804)

    shape = (21, 21)
    noise_sigma = 2.0

    truth = {
        "re": 4.5,
        "flux__g": 500.0,
        "flux__r": 900.0,
    }

    noiseless_g = make_model(
        flux=truth["flux__g"],
        re=truth["re"],
    ).render(shape=shape)

    noiseless_r = make_model(
        flux=truth["flux__r"],
        re=truth["re"],
    ).render(shape=shape)

    data_g = noiseless_g + rng.normal(
        loc=0.0,
        scale=noise_sigma,
        size=shape,
    )

    data_r = noiseless_r + rng.normal(
        loc=0.0,
        scale=noise_sigma,
        size=shape,
    )

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
                        lower=200.0,
                        upper=800.0,
                    ),
                    "r": Parameter(
                        name="flux",
                        lower=500.0,
                        upper=1300.0,
                    ),
                },
            ),
        ],
    )

    likelihood = MultiBandLikelihood(
        {
            "g": GaussianImageLikelihood(
                data=data_g,
                noise=np.full(
                    shape,
                    noise_sigma,
                ),
            ),
            "r": GaussianImageLikelihood(
                data=data_r,
                noise=np.full(
                    shape,
                    noise_sigma,
                ),
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
        sample="rwalk",
        dlogz=0.5,
        print_progress=False,
        seed=12345,
    )

    assert result.parameter_names == [
        "re",
        "flux__g",
        "flux__r",
    ]

    summaries = {
        name: result.summary(name)
        for name in result.parameter_names
    }

    for name, expected in truth.items():
        assert summaries[name]["median"] == pytest.approx(
            expected,
            rel=0.15,
        )

        lower_two_sigma = (
            summaries[name]["median"]
            - 2.0
            * (
                summaries[name]["median"]
                - summaries[name]["q16"]
            )
        )

        upper_two_sigma = (
            summaries[name]["median"]
            + 2.0
            * (
                summaries[name]["q84"]
                - summaries[name]["median"]
            )
        )

        assert lower_two_sigma <= expected <= upper_two_sigma

    assert np.isfinite(result.final_logz)
    assert result.logzerr is not None
    assert np.isfinite(result.logzerr[-1])
