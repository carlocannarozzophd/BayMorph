"""Minimal Dynesty fitter for flux and effective radius."""

import dynesty
import numpy as np

from baymorph.likelihood import gaussian_log_likelihood
from baymorph.results import PosteriorResult


class DynestyFluxReResult(
    PosteriorResult
):
    """Flux + Re posterior result."""


class DynestyFluxReFitter:
    """Fit flux and effective radius with Dynesty."""

    def __init__(
        self,
        model_factory,
        data,
        noise,
        flux_min,
        flux_max,
        re_min,
        re_max,
    ):
        self.model_factory = model_factory
        self.data = data
        self.noise = noise
        self.flux_min = flux_min
        self.flux_max = flux_max
        self.re_min = re_min
        self.re_max = re_max

    def prior_transform(
        self,
        u,
    ):
        flux = (
            self.flux_min
            + u[0]
            * (
                self.flux_max
                - self.flux_min
            )
        )

        re = (
            self.re_min
            + u[1]
            * (
                self.re_max
                - self.re_min
            )
        )

        return np.array(
            [
                flux,
                re,
            ]
        )

    def loglikelihood(
        self,
        theta,
    ):
        flux, re = theta

        model = self.model_factory(
            flux,
            re,
        )

        image = model.render(
            shape=self.data.shape
        )

        return gaussian_log_likelihood(
            data=self.data,
            model=image,
            noise=self.noise,
        )

    def run(
        self,
        nlive=100,
    ):
        sampler = dynesty.NestedSampler(
            self.loglikelihood,
            self.prior_transform,
            ndim=2,
            nlive=nlive,
        )

        sampler.run_nested()

        results = sampler.results

        return DynestyFluxReResult(
            samples=results.samples,
            logwt=results.logwt,
            logz=results.logz,
            logl=results.logl,
            parameter_names=[
                "flux",
                "re",
            ],
        )