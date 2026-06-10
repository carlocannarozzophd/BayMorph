"""Minimal Dynesty flux fitter."""

from dataclasses import dataclass

import dynesty
import numpy as np

from baymorph.likelihood import gaussian_log_likelihood


@dataclass
class DynestyFluxResult:
    """Result of a flux-only Dynesty fit."""

    samples: np.ndarray
    logwt: np.ndarray
    logz: np.ndarray

    @property
    def flux_median(self) -> float:
        return float(np.median(self.samples[:, 0]))


class DynestyFluxFitter:
    """
    Fit only the flux parameter.

    All structural parameters are fixed.
    """

    def __init__(
        self,
        model_factory,
        data,
        noise,
        flux_min,
        flux_max,
    ):
        self.model_factory = model_factory
        self.data = data
        self.noise = noise
        self.flux_min = flux_min
        self.flux_max = flux_max

    def prior_transform(self, u):
        flux = (
            self.flux_min
            + u[0] * (self.flux_max - self.flux_min)
        )
        return np.array([flux])

    def loglikelihood(self, theta):
        flux = theta[0]

        model = self.model_factory(flux)

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
            ndim=1,
            nlive=nlive,
        )

        sampler.run_nested()

        results = sampler.results

        return DynestyFluxResult(
            samples=results.samples,
            logwt=results.logwt,
            logz=results.logz,
        )