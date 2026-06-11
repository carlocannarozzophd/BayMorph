"""Generic Dynesty fitter."""

import dynesty
import numpy as np

from baymorph.likelihood import gaussian_log_likelihood
from baymorph.results import PosteriorResult


class DynestyResult(PosteriorResult):
    """Generic Dynesty posterior result."""


class DynestyFitter:
    """Generic Dynesty fitter for arbitrary model parameters."""

    def __init__(
        self,
        model_factory,
        data,
        noise,
        parameter_names: list[str],
        prior_bounds: dict[str, tuple[float, float]],
        mask=None,
    ):
        self.model_factory = model_factory
        self.data = np.asarray(data, dtype=float)
        self.noise = np.asarray(noise, dtype=float)
        self.parameter_names = list(parameter_names)
        self.prior_bounds = dict(prior_bounds)
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
        theta = []

        for value, name in zip(u, self.parameter_names):
            lower, upper = self.prior_bounds[name]
            theta.append(lower + value * (upper - lower))

        return np.asarray(theta)

    def theta_to_parameters(self, theta) -> dict[str, float]:
        """Convert parameter vector to dictionary."""
        return {
            name: float(value)
            for name, value in zip(self.parameter_names, theta)
        }

    def loglikelihood(self, theta):
        """Evaluate log-likelihood."""
        parameters = self.theta_to_parameters(theta)

        model = self.model_factory(**parameters)

        model_image = model.render(
            shape=self.data.shape,
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
    ) -> DynestyResult:
        """Run nested sampling."""
        sampler_kwargs = {
            "nlive": nlive,
            "sample": sample,
            "bound": bound,
        }

        if bootstrap is not None:
            sampler_kwargs["bootstrap"] = bootstrap

        sampler = dynesty.NestedSampler(
            self.loglikelihood,
            self.prior_transform,
            ndim=self.ndim,
            **sampler_kwargs,
        )

        run_kwargs = {
            "print_progress": print_progress,
        }

        if dlogz is not None:
            run_kwargs["dlogz"] = dlogz

        sampler.run_nested(
            **run_kwargs,
        )

        results = sampler.results

        return DynestyResult(
            samples=results.samples,
            logwt=results.logwt,
            logz=results.logz,
            logl=results.logl,
            parameter_names=self.parameter_names,
        )