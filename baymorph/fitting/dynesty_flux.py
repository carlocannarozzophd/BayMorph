"""Minimal Dynesty flux fitter."""

import dynesty
import numpy as np

from baymorph.likelihood import gaussian_log_likelihood
from baymorph.results import PosteriorResult


class DynestyFluxResult(PosteriorResult):
    """Posterior result for a flux-only fit."""


class DynestyFluxFitter:
    """Fit only the flux parameter with Dynesty.

    Parameters
    ----------
    model_factory : callable
        Callable that receives ``flux`` and returns a model object with a
        ``render(shape=...)`` method.
    data : array-like
        Observed image data.
    noise : array-like or float
        Noise map or scalar noise level used in the Gaussian likelihood.
    flux_min, flux_max : float
        Lower and upper bounds of the uniform prior on total flux.
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
        """Transform a unit-cube sample to physical flux.

        Parameters
        ----------
        u : array-like
            Unit-cube sample with one value.

        Returns
        -------
        ndarray
            Physical parameter array ``[flux]``.
        """
        flux = self.flux_min + u[0] * (self.flux_max - self.flux_min)
        return np.array([flux])

    def loglikelihood(self, theta):
        """Evaluate the Gaussian log-likelihood for one flux value.

        Parameters
        ----------
        theta : array-like
            Physical parameter array ``[flux]``.

        Returns
        -------
        float
            Gaussian log-likelihood value.
        """
        flux = theta[0]

        model = self.model_factory(flux)
        image = model.render(shape=self.data.shape)

        return gaussian_log_likelihood(
            data=self.data,
            model=image,
            noise=self.noise,
        )

    def run(self, nlive=100):
        """Run nested sampling.

        Parameters
        ----------
        nlive : int, default=100
            Number of live points used by Dynesty.

        Returns
        -------
        DynestyFluxResult
            Posterior samples, log-weights, log-likelihoods, evidence sequence,
            and parameter names for the fit.
        """
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
            logl=results.logl,
            parameter_names=["flux"],
        )