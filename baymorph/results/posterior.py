"""Posterior result containers."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from dynesty.utils import resample_equal

from baymorph.stats import posterior_summary


@dataclass
class PosteriorResult:
    """Container for Bayesian posterior samples.

    Parameters
    ----------
    samples : ndarray
        Posterior samples with shape ``(n_samples, n_parameters)``.
    logwt : ndarray
        Logarithmic Dynesty sample weights.
    logz : ndarray
        Cumulative logarithmic Bayesian evidence.
    parameter_names : list of str
        Names of the fitted parameters in the same order as the sample columns.
    logl : ndarray, optional
        Log-likelihood values for each sample.
    logzerr : ndarray, optional
        Estimated uncertainty on the cumulative evidence.
    """

    samples: np.ndarray
    logwt: np.ndarray
    logz: np.ndarray
    parameter_names: list[str]
    logl: np.ndarray | None = None
    logzerr: np.ndarray | None = None

    @property
    def weights(self) -> np.ndarray:
        """Return normalized posterior sample weights.

        The weights are computed directly from ``logwt`` and the final
        evidence, making this result container independent of the sampler
        backend that generated the samples.
        """
        logz_final = float(np.asarray(self.logz)[-1])
        weights = np.exp(np.asarray(self.logwt) - logz_final)

        normalization = float(np.sum(weights))
        if not np.isfinite(normalization) or normalization <= 0.0:
            raise ValueError(
                "Posterior weights cannot be normalized because their "
                "total is not finite and positive."
            )

        return weights / normalization

    @cached_property
    def posterior_samples(self) -> np.ndarray:
        """ndarray : Cached equal-weight posterior samples."""
        return resample_equal(
            self.samples,
            self.weights,
        )

    def summary(
        self,
        parameter_name: str,
    ):
        """Return summary statistics for one fitted parameter.

        Parameters
        ----------
        parameter_name : str
            Name of the fitted parameter.

        Returns
        -------
        dict
            Posterior summary returned by
            :func:`baymorph.stats.posterior_summary`.

        Raises
        ------
        ValueError
            If ``parameter_name`` is not present in the posterior.
        """
        idx = self.parameter_names.index(
            parameter_name
        )

        return posterior_summary(
            self.samples[:, idx],
            self.weights,
        )

    @property
    def best_index(self) -> int:
        """int : Index of the maximum-likelihood sample.

        Raises
        ------
        ValueError
            If log-likelihood values are not available.
        """
        if self.logl is None:
            raise ValueError(
                "logl is required to compute best-fit parameters."
            )

        return int(
            np.argmax(self.logl)
        )

    @property
    def best_sample(self) -> np.ndarray:
        """ndarray : Maximum-likelihood parameter vector."""
        return self.samples[
            self.best_index
        ]

    @property
    def best_parameters(self) -> dict[str, float]:
        """dict of str to float : Maximum-likelihood parameter values."""
        return {
            name: float(value)
            for name, value in zip(
                self.parameter_names,
                self.best_sample,
            )
        }

    @property
    def median_parameters(
        self,
    ) -> dict[str, float]:
        """dict of str to float : Posterior-median parameter values."""
        return {
            name: self.summary(name)["median"]
            for name in self.parameter_names
        }

    def __getattr__(
        self,
        name,
    ):
        """Provide dynamic posterior-summary attributes.

        Dynamic attributes of the form

        - ``<parameter>_median``
        - ``<parameter>_q16``
        - ``<parameter>_q84``

        are resolved automatically from the posterior summaries.

        Parameters
        ----------
        name : str
            Requested attribute name.

        Returns
        -------
        float
            Requested posterior statistic.

        Raises
        ------
        AttributeError
            If the requested attribute cannot be resolved.
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

    @property
    def final_logz(self) -> float:
        """Return the final logarithmic Bayesian evidence."""
        return float(np.asarray(self.logz)[-1])
