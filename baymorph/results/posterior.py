"""Posterior result containers."""

from dataclasses import dataclass

import numpy as np
from dynesty.utils import resample_equal

from baymorph.stats import (
    dynesty_weights,
    posterior_summary,
)


@dataclass
class PosteriorResult:
    """
    Generic posterior result.

    Parameters
    ----------
    samples
        Raw Dynesty samples.

    logwt
        Dynesty log-weights.

    logz
        Dynesty evidence history.

    parameter_names
        Names of fitted parameters.
    """

    samples: np.ndarray
    logwt: np.ndarray
    logz: np.ndarray
    parameter_names: list[str]

    @property
    def weights(self) -> np.ndarray:
        """Posterior weights."""
        return dynesty_weights(
            self.logwt,
            self.logz,
        )

    @property
    def posterior_samples(self) -> np.ndarray:
        """
        Equal-weight posterior samples.

        Useful for corner plots and MCMC-like workflows.
        """
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

    def __getattr__(
        self,
        name,
    ):
        """
        Dynamic attributes.

        Examples
        --------
        result.flux_median
        result.flux_q16
        result.flux_q84
        """

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

                if (
                    parameter
                    not in self.parameter_names
                ):
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