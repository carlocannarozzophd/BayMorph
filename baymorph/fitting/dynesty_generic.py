"""Generic Dynesty fitter."""

import dynesty
import numpy as np

from multiprocessing import Pool

from baymorph.likelihood import gaussian_log_likelihood
from baymorph.results import PosteriorResult
from baymorph.simulation.psf import convolve_with_psf


class DynestyResult(PosteriorResult):
    """Generic Dynesty posterior result."""


class DynestyFitter:
    """Generic Dynesty fitter for arbitrary model parameters."""

    def __init__(
        self,
        model_factory,
        data,
        noise,
        parameter_names: list[str] | None = None,
        prior_bounds: dict[str, tuple[float, float]] | None = None,
        parameters=None,
        psf=None,
        mask=None,
    ):
        if parameters is not None:
            parameter_names = parameters.names
            prior_bounds = parameters.bounds

        if parameter_names is None:
            raise ValueError(
                "parameter_names must be provided when parameters is None."
            )

        if prior_bounds is None:
            raise ValueError(
                "prior_bounds must be provided when parameters is None."
            )

        self.model_factory = model_factory
        self.data = np.asarray(data, dtype=float)
        self.noise = np.asarray(noise, dtype=float)
        self.parameter_names = list(parameter_names)
        self.prior_bounds = dict(prior_bounds)
        self.parameters = parameters
        self.psf = psf
        self.mask = mask

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.data.shape != self.noise.shape:
            raise ValueError("Data and noise must have the same shape.")

        if len(self.parameter_names) == 0:
            raise ValueError("At least one parameter is required.")

        for name in self.parameter_names:
            if name not in self.prior_bounds:
                raise ValueError(f"Missing prior bounds for parameter: {name}")

            lower, upper = self.prior_bounds[name]

            if upper <= lower:
                raise ValueError(f"Invalid prior bounds for parameter: {name}")

    @property
    def ndim(self) -> int:
        """Number of fitted parameters."""
        return len(self.parameter_names)

    def prior_transform(self, u):
        """Transform unit-cube samples to physical parameters."""
        if self.parameters is not None:
            return self.parameters.transform(u)

        theta = []

        for value, name in zip(u, self.parameter_names):
            lower, upper = self.prior_bounds[name]

            theta.append(
                lower
                + value
                * (
                    upper
                    - lower
                )
            )

        return np.asarray(theta)

    def theta_to_parameters(
        self,
        theta,
    ) -> dict[str, float]:
        """Convert parameter vector to dictionary."""
        return {
            name: float(value)
            for name, value in zip(
                self.parameter_names,
                theta,
            )
        }

    def loglikelihood(
        self,
        theta,
    ):
        """Evaluate log-likelihood."""
        parameters = self.theta_to_parameters(
            theta,
        )

        model = self.model_factory(
            **parameters,
        )

        model_image = model.render(
            shape=self.data.shape,
        )

        if self.psf is not None:
            model_image = convolve_with_psf(
                model_image,
                self.psf,
            )

        return gaussian_log_likelihood(
            data=self.data,
            model=model_image,
            noise=self.noise,
            mask=self.mask,
        )

    def run(
        self,
        nlive: int = 100,
        sample: str = "auto",
        bound: str = "multi",
        dlogz: float | None = None,
        bootstrap: int | None = None,
        print_progress: bool = True,
        n_processes: int = 1,
        queue_size: int | None = None,
        seed: int | None = None,
    ) -> DynestyResult:
        """Run nested sampling."""
        rstate = None

        if seed is not None:
            rstate = np.random.default_rng(seed)
        
        sampler_kwargs = {
            "nlive": nlive,
            "sample": sample,
            "bound": bound,
        }

        if rstate is not None:
            sampler_kwargs["rstate"] = rstate

        if bootstrap is not None:
            sampler_kwargs["bootstrap"] = bootstrap

        run_kwargs = {
            "print_progress": print_progress,
        }

        if dlogz is not None:
            run_kwargs["dlogz"] = dlogz

        if n_processes > 1:
            if queue_size is None:
                queue_size = n_processes

            with Pool(processes=n_processes) as pool:
                sampler = dynesty.NestedSampler(
                    self.loglikelihood,
                    self.prior_transform,
                    ndim=self.ndim,
                    pool=pool,
                    queue_size=queue_size,
                    **sampler_kwargs,
                )

                sampler.run_nested(
                    **run_kwargs,
                )

                results = sampler.results

        else:
            sampler = dynesty.NestedSampler(
                self.loglikelihood,
                self.prior_transform,
                ndim=self.ndim,
                **sampler_kwargs,
            )

            sampler.run_nested(
                **run_kwargs,
            )

            results = sampler.results

        return DynestyResult(
            samples=results.samples,
            logwt=results.logwt,
            logz=results.logz,
            logl=results.logl,
            logzerr=results.logzerr,
            parameter_names=self.parameter_names,
        )