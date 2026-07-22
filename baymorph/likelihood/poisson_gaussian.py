"""Poisson-Gaussian likelihood for detector image data."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import poisson

from baymorph.likelihood.base import ImageLikelihood
    

class PoissonGaussianImageLikelihood(ImageLikelihood):
    """Likelihood for Poisson counts with additive Gaussian read noise.

    The detector model for each evaluated pixel is

    .. math::

        K_i \\sim \\operatorname{Poisson}(\\mu_i),

    followed by

    .. math::

        D_i \\mid K_i
        \\sim \\mathcal{N}(K_i, \\sigma_i^2).

    The latent integer count ``K_i`` is marginalized numerically:

    .. math::

        p(D_i \\mid \\mu_i, \\sigma_i)
        =
        \\sum_{k=0}^{\\infty}
        p(k \\mid \\mu_i)
        p(D_i \\mid k, \\sigma_i).

    Parameters
    ----------
    data : array-like
        Observed image in detector-count units.
    read_noise : array-like or float
        Standard deviation of the additive Gaussian read noise, expressed in
        the same count units as ``data`` and the model.
    mask : array-like of bool, optional
        Boolean mask identifying pixels excluded from likelihood evaluation.
    tail_probability : float, default=1e-12
        Upper-tail probability used to truncate the latent Poisson sum.
        Smaller values increase accuracy and computational cost.
    minimum_terms : int, default=20
        Minimum number of latent count values included for each pixel.
    maximum_terms : int, default=100000
        Safety limit for the largest latent count included in a pixel sum.
        If the required truncation exceeds this value, evaluation raises an
        exception instead of silently returning an uncontrolled
        approximation.

    Notes
    -----
    Data, model expectations, and read noise must all use the same linear
    detector-count units.

    Any conversion from ADU to electrons must be performed explicitly before
    constructing this likelihood.

    The model must include every Poisson-generating contribution, including
    source and sky counts. A background-subtracted model is generally not
    appropriate unless the subtraction procedure is represented explicitly
    in the statistical model.

    This exact marginalized likelihood is substantially more expensive than
    the Gaussian or Cash likelihoods.
    """

    def __init__(
        self,
        data: np.ndarray,
        read_noise: np.ndarray | float,
        mask: np.ndarray | None = None,
        tail_probability: float = 1.0e-12,
        minimum_terms: int = 20,
        maximum_terms: int = 100000,
    ) -> None:
        super().__init__(
            data=data,
            mask=mask,
        )

        if not 0.0 < tail_probability < 1.0:
            raise ValueError(
                "tail_probability must lie strictly between zero and one."
            )

        if minimum_terms < 1:
            raise ValueError(
                "minimum_terms must be at least one."
            )

        if maximum_terms < minimum_terms:
            raise ValueError(
                "maximum_terms must be greater than or equal to "
                "minimum_terms."
            )

        self.read_noise_values = self.prepare_fixed_array(
            read_noise,
            name="Read noise",
            require_positive=True,
        )

        self.tail_probability = float(
            tail_probability
        )
        self.minimum_terms = int(
            minimum_terms
        )
        self.maximum_terms = int(
            maximum_terms
        )

        self._log_gaussian_normalization = -np.log(
            np.sqrt(2.0 * np.pi)
            * self.read_noise_values
        )

    def log_likelihood(
        self,
        model_values: np.ndarray,
    ) -> float:
        """Evaluate the marginalized Poisson-Gaussian log-likelihood.

        Parameters
        ----------
        model_values : ndarray
            Non-negative expected source-plus-background counts in the
            unmasked pixels.

        Returns
        -------
        float
            Poisson-Gaussian log-likelihood.

        Raises
        ------
        ValueError
            If a model expectation is negative or if controlled truncation
            requires more latent-count terms than ``maximum_terms``.
        """
        if np.any(model_values < 0.0):
            raise ValueError(
                "Poisson-Gaussian likelihood requires non-negative model "
                "expectations."
            )

        total_loglike = 0.0

        for (
            data_value,
            model_value,
            read_noise,
            log_gaussian_normalization,
        ) in zip(
            self.data_values,
            model_values,
            self.read_noise_values,
            self._log_gaussian_normalization,
            strict=True,
        ):
            total_loglike += self._pixel_log_likelihood(
                data_value=float(data_value),
                model_value=float(model_value),
                read_noise=float(read_noise),
                log_gaussian_normalization=float(
                    log_gaussian_normalization
                ),
            )

        return float(total_loglike)

    def _pixel_log_likelihood(
        self,
        *,
        data_value: float,
        model_value: float,
        read_noise: float,
        log_gaussian_normalization: float,
    ) -> float:
        """Evaluate one marginalized pixel likelihood."""
        if model_value == 0.0:
            return float(
                log_gaussian_normalization
                - 0.5
                * (data_value / read_noise) ** 2
            )

        maximum_count = self._resolve_maximum_count(
            data_value=data_value,
            model_value=model_value,
            read_noise=read_noise,
        )

        counts = np.arange(
            maximum_count + 1,
            dtype=float,
        )

        log_poisson = (
            counts * np.log(model_value)
            - model_value
            - gammaln(counts + 1.0)
        )

        log_gaussian = (
            log_gaussian_normalization
            - 0.5
            * ((data_value - counts) / read_noise) ** 2
        )

        return float(
            logsumexp(
                log_poisson + log_gaussian
            )
        )

    def _resolve_maximum_count(
        self,
        *,
        data_value: float,
        model_value: float,
        read_noise: float,
    ) -> int:
        """Resolve a controlled upper bound for the latent count sum."""
        poisson_upper = poisson.ppf(
            1.0 - self.tail_probability,
            model_value,
        )

        if not np.isfinite(poisson_upper):
            raise ValueError(
                "Could not determine a finite Poisson truncation bound."
            )

        gaussian_upper = np.ceil(
            max(
                0.0,
                data_value + 8.0 * read_noise,
            )
        )

        maximum_count = max(
            self.minimum_terms - 1,
            int(np.ceil(poisson_upper)),
            int(gaussian_upper),
        )

        if maximum_count > self.maximum_terms:
            raise ValueError(
                "Poisson-Gaussian likelihood requires more latent-count "
                "terms than allowed by maximum_terms. Increase "
                "maximum_terms or use a scientifically justified "
                "approximation."
            )

        return maximum_count