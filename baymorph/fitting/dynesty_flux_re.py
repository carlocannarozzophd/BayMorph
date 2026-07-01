"""Minimal Dynesty fitter for flux and effective radius."""

import dynesty
import numpy as np

from baymorph.likelihood import gaussian_log_likelihood
from baymorph.results import PosteriorResult


class DynestyFluxReResult(PosteriorResult):
    """Posterior result for a flux and effective-radius fit."""


class DynestyFluxReFitter:
    """Fit flux and effective radius with Dynesty.

    Parameters
    ----------
    model_factory : callable
        Callable that receives ``flux`` and ``re`` and returns a model object
        with a ``render(shape=...)`` method.
    data : array-like
        Observed image data.
    noise : array-like or float
        Noise map or scalar noise level used in the Gaussian likelihood.
    flux_min, flux_max : float
        Lower and upper bounds of the uniform prior on total flux.
    re_min, re_max : float
        Lower and upper bounds of the uniform prior on effective radius.
    """

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
        """Transform unit-cube samples to physical model parameters.

        Parameters
        ----------
        u : array-like
            Unit-cube sample with two values corresponding to flux and
            effective radius.

        Returns
        -------
        ndarray
            Physical parameters ``[flux, re]``.
        """
        flux = self.flux_min + u[0] * (self.flux_max - self.flux_min)
        re = self.re_min + u[1] * (self.re_max - self.re_min)

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
        """Evaluate the Gaussian log-likelihood for one parameter vector.

        Parameters
        ----------
        theta : array-like
            Physical model parameters ``[flux, re]``.

        Returns
        -------
        float
            Gaussian log-likelihood value.
        """
        flux, re = theta

        model = self.model_factory(
            flux,
            re,
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
        """Run nested sampling.

        Parameters
        ----------
        nlive : int, default=100
            Number of live points used by Dynesty.

        Returns
        -------
        DynestyFluxReResult
            Posterior samples, log-weights, log-likelihoods, evidence sequence,
            and parameter names for the fit.
        """
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