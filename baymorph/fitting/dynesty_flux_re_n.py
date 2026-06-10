"""Minimal Dynesty fitter for flux, effective radius and Sersic index."""

from dataclasses import dataclass

import dynesty
import numpy as np

from baymorph.likelihood import gaussian_log_likelihood


@dataclass
class DynestyFluxReNResult:
    """Result of a flux + Re + n Dynesty fit."""

    samples: np.ndarray
    logwt: np.ndarray
    logz: np.ndarray

    @property
    def flux_median(self) -> float:
        return float(np.median(self.samples[:, 0]))

    @property
    def re_median(self) -> float:
        return float(np.median(self.samples[:, 1]))

    @property
    def n_median(self) -> float:
        return float(np.median(self.samples[:, 2]))


class DynestyFluxReNFitter:
    """Fit flux, effective radius and Sersic index with Dynesty."""

    def __init__(
        self,
        model_factory,
        data,
        noise,
        flux_min,
        flux_max,
        re_min,
        re_max,
        n_min,
        n_max,
    ):
        self.model_factory = model_factory
        self.data = data
        self.noise = noise
        self.flux_min = flux_min
        self.flux_max = flux_max
        self.re_min = re_min
        self.re_max = re_max
        self.n_min = n_min
        self.n_max = n_max

    def prior_transform(self, u):
        flux = self.flux_min + u[0] * (self.flux_max - self.flux_min)
        re = self.re_min + u[1] * (self.re_max - self.re_min)
        n = self.n_min + u[2] * (self.n_max - self.n_min)

        return np.array([flux, re, n])

    def loglikelihood(self, theta):
        flux, re, n = theta

        model = self.model_factory(
            flux,
            re,
            n,
        )

        image = model.render(
            shape=self.data.shape,
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
            ndim=3,
            nlive=nlive,
        )

        sampler.run_nested()

        results = sampler.results

        return DynestyFluxReNResult(
            samples=results.samples,
            logwt=results.logwt,
            logz=results.logz,
        )