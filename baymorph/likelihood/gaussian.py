"""Gaussian image likelihood."""

from __future__ import annotations

import numpy as np

from baymorph.likelihood.base import ImageLikelihood


class GaussianImageLikelihood(ImageLikelihood):
    """Gaussian likelihood for pixel-based image data.

    The likelihood assumes statistically independent Gaussian uncertainties
    in all evaluated pixels.

    Parameters
    ----------
    data : array-like
        Observed image.
    noise : array-like or float
        Gaussian standard deviation. It may be a scalar or an array
        broadcastable to the observed image shape. All evaluated values must
        be strictly positive.
    mask : array-like of bool, optional
        Boolean mask identifying pixels excluded from likelihood evaluation.
    include_normalization : bool, default=True
        If True, include the Gaussian normalization term. If False, evaluate
        only the chi-squared contribution.
    """

    def __init__(
        self,
        data: np.ndarray,
        noise: np.ndarray | float,
        mask: np.ndarray | None = None,
        include_normalization: bool = True,
    ) -> None:
        super().__init__(
            data=data,
            mask=mask,
        )

        self.noise_values = self.prepare_fixed_array(
            noise,
            name="Noise",
            require_positive=True,
        )
        self.include_normalization = bool(
            include_normalization
        )

        self._inverse_variance = 1.0 / self.noise_values**2

        if self.include_normalization:
            self._normalization = -0.5 * np.sum(
                np.log(
                    2.0 * np.pi * self.noise_values**2
                )
            )
        else:
            self._normalization = 0.0

    def _log_likelihood(
        self,
        model_values: np.ndarray,
    ) -> float:
        """Evaluate the Gaussian image log-likelihood.

        Parameters
        ----------
        model_values : ndarray
            Model values in the unmasked pixels.

        Returns
        -------
        float
            Gaussian log-likelihood.
        """
        residual = self.data_values - model_values

        chi2 = np.sum(
            residual**2 * self._inverse_variance
        )

        return float(
            -0.5 * chi2 + self._normalization
        )