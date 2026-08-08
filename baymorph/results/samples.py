"""Backend-neutral posterior-sample representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from dynesty.utils import resample_equal

from baymorph.stats import posterior_summary


@dataclass
class PosteriorSamples:
    """Backend-neutral weighted posterior samples.

    Parameters
    ----------
    samples : ndarray
        Posterior samples with shape ``(n_samples, n_parameters)``.
    weights : ndarray
        Non-negative posterior sample weights. They are normalized on
        construction.
    parameter_names : sequence of str
        Parameter names in sample-column order.
    log_likelihood : ndarray, optional
        Log-likelihood value associated with each posterior sample.
    log_prior : ndarray, optional
        Log-prior density associated with each posterior sample.

    Notes
    -----
    This representation intentionally contains no sampler-specific concepts
    such as nested-sampling log weights or cumulative evidence trajectories.
    """

    samples: np.ndarray
    weights: np.ndarray
    parameter_names: list[str]
    log_likelihood: np.ndarray | None = None
    log_prior: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate and normalize posterior data."""
        samples = np.asarray(
            self.samples,
            dtype=float,
        )

        if samples.ndim != 2:
            raise ValueError(
                "Posterior samples must be a two-dimensional array."
            )

        n_samples, n_parameters = samples.shape

        names = [
            str(name)
            for name in self.parameter_names
        ]

        if len(names) != n_parameters:
            raise ValueError(
                "The number of parameter names must match the "
                "number of sample columns."
            )

        weights = np.asarray(
            self.weights,
            dtype=float,
        )

        if weights.ndim != 1:
            raise ValueError(
                "Posterior weights must be a one-dimensional array."
            )

        if weights.shape[0] != n_samples:
            raise ValueError(
                "Posterior weights must contain one value per sample."
            )

        if not np.all(
            np.isfinite(weights)
        ):
            raise ValueError(
                "Posterior weights must be finite."
            )

        if np.any(
            weights < 0.0
        ):
            raise ValueError(
                "Posterior weights must be non-negative."
            )

        normalization = float(
            np.sum(weights)
        )

        if (
            not np.isfinite(normalization)
            or normalization <= 0.0
        ):
            raise ValueError(
                "Posterior weights must have a finite positive sum."
            )

        self.samples = samples
        self.weights = weights / normalization
        self.parameter_names = names

        self.log_likelihood = self._validate_optional_sample_quantity(
            self.log_likelihood,
            name="log_likelihood",
            n_samples=n_samples,
        )

        self.log_prior = self._validate_optional_sample_quantity(
            self.log_prior,
            name="log_prior",
            n_samples=n_samples,
        )

    @staticmethod
    def _validate_optional_sample_quantity(
        value: np.ndarray | None,
        *,
        name: str,
        n_samples: int,
    ) -> np.ndarray | None:
        """Validate one optional sample-level quantity."""
        if value is None:
            return None

        array = np.asarray(
            value,
            dtype=float,
        )

        if array.ndim != 1:
            raise ValueError(
                f"{name} must be a one-dimensional array."
            )

        if array.shape[0] != n_samples:
            raise ValueError(
                f"{name} must contain one value per posterior sample."
            )

        return array

    @property
    def ndim(self) -> int:
        """Return posterior dimensionality."""
        return int(
            self.samples.shape[1]
        )

    @property
    def n_samples(self) -> int:
        """Return the number of weighted posterior samples."""
        return int(
            self.samples.shape[0]
        )

    @property
    def equal_weight_samples(self) -> np.ndarray:
        """Return equal-weight posterior samples."""
        return resample_equal(
            self.samples,
            self.weights,
        )

    @property
    def effective_sample_size(self) -> float:
        """Return the importance-weight effective sample size."""
        return float(
            1.0
            / np.sum(
                self.weights**2
            )
        )

    def summary(
        self,
        parameter_name: str,
    ) -> dict[str, float]:
        """Return weighted posterior summary statistics for one parameter."""
        try:
            index = self.parameter_names.index(
                parameter_name
            )
        except ValueError as exc:
            raise ValueError(
                f"Unknown posterior parameter {parameter_name!r}."
            ) from exc

        return posterior_summary(
            self.samples[:, index],
            self.weights,
        )

    @property
    def best_index(self) -> int:
        """Return the maximum-likelihood sample index."""
        if self.log_likelihood is None:
            raise ValueError(
                "log_likelihood is required to identify "
                "the maximum-likelihood sample."
            )

        return int(
            np.argmax(
                self.log_likelihood
            )
        )

    @property
    def best_sample(self) -> np.ndarray:
        """Return the maximum-likelihood parameter vector."""
        return self.samples[
            self.best_index
        ]

    @property
    def best_parameters(self) -> dict[str, float]:
        """Return maximum-likelihood parameters by name."""
        return {
            name: float(value)
            for name, value in zip(
                self.parameter_names,
                self.best_sample,
                strict=True,
            )
        }

    @property
    def median_parameters(self) -> dict[str, float]:
        """Return posterior-median parameters by name."""
        return {
            name: float(
                self.summary(name)["median"]
            )
            for name in self.parameter_names
        }
