"""Posterior result containers."""

from dataclasses import dataclass

import numpy as np
from dynesty.utils import resample_equal

from baymorph.stats import dynesty_weights, posterior_summary


@dataclass
class PosteriorResult:
    """Generic posterior result."""

    samples: np.ndarray
    logwt: np.ndarray
    logz: np.ndarray
    parameter_names: list[str]
    logl: np.ndarray | None = None
    logzerr: np.ndarray | None = None

    @property
    def weights(self) -> np.ndarray:
        """Posterior weights."""
        return dynesty_weights(
            self.logwt,
            self.logz,
        )

    @property
    def posterior_samples(self) -> np.ndarray:
        """Equal-weight posterior samples."""
        return resample_equal(
            self.samples,
            self.weights,
        )

    def summary(
        self,
        parameter_name: str,
    ):
        """Posterior summary for one parameter."""
        idx = self.parameter_names.index(
            parameter_name
        )

        return posterior_summary(
            self.samples[:, idx],
            self.weights,
        )

    @property
    def best_index(self) -> int:
        """Index of the maximum-likelihood sample."""
        if self.logl is None:
            raise ValueError(
                "logl is required to compute best-fit parameters."
            )

        return int(
            np.argmax(self.logl)
        )

    @property
    def best_sample(self) -> np.ndarray:
        """Maximum-likelihood sample."""
        return self.samples[
            self.best_index
        ]

    @property
    def best_parameters(self) -> dict[str, float]:
        """Maximum-likelihood parameters as a dictionary."""
        return {
            name: float(value)
            for name, value in zip(
                self.parameter_names,
                self.best_sample,
            )
        }

    def median_parameters(
        self,
    ) -> dict[str, float]:
        """Posterior-median parameters as a dictionary."""
        return {
            name: self.summary(name)["median"]
            for name in self.parameter_names
        }

    def __getattr__(
        self,
        name,
    ):
        """Dynamic attributes such as flux_median, flux_q16, flux_q84."""
        suffixes = [
            "_median",
            "_q16",
            "_q84",
        ]

        for suffix in suffixes:
            if name.endswith(
                suffix
            ):
                parameter = name[
                    : -len(suffix)
                ]

                if parameter not in self.parameter_names:
                    break

                summary = self.summary(
                    parameter
                )

                key = suffix.replace(
                    "_",
                    ""
                )

                return summary[key]

        raise AttributeError(
            name
        )