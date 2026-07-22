"""Cash likelihood for Poisson-distributed image data."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, xlogy

from baymorph.likelihood.base import ImageLikelihood


class CashImageLikelihood(ImageLikelihood):
    """Poisson likelihood expressed through the Cash statistic.

    For observed counts ``data`` and expected counts ``model``, the complete
    Poisson log-likelihood is

    .. math::

        \\log \\mathcal{L}
        =
        \\sum_i
        \\left[
            d_i \\log(m_i)
            - m_i
            - \\log\\Gamma(d_i + 1)
        \\right].

    The factorial term depends only on the observed data. It may therefore be
    omitted when comparing models fitted to exactly the same dataset.

    Parameters
    ----------
    data : array-like
        Observed non-negative counts.
    mask : array-like of bool, optional
        Boolean mask identifying pixels excluded from likelihood evaluation.
    include_data_term : bool, default=False
        If True, include the model-independent factorial term
        ``-gammaln(data + 1)``.
    require_integer_data : bool, default=True
        If True, require observed values to be integer-valued counts.
        Setting this to False enables the continuous extension commonly used
        with the Cash statistic, but does not make background-subtracted data
        physically Poisson-distributed.
    integer_tolerance : float, default=1e-10
        Absolute tolerance used when checking whether observed values are
        integer-valued.
    """

    def __init__(
        self,
        data: np.ndarray,
        mask: np.ndarray | None = None,
        include_data_term: bool = False,
        require_integer_data: bool = True,
        integer_tolerance: float = 1.0e-10,
    ) -> None:
        super().__init__(
            data=data,
            mask=mask,
        )

        if integer_tolerance < 0.0:
            raise ValueError(
                "integer_tolerance must be non-negative."
            )

        if np.any(self.data_values < 0.0):
            raise ValueError(
                "Cash likelihood requires non-negative observed counts."
            )

        if require_integer_data:
            rounded_data = np.rint(
                self.data_values
            )

            if not np.allclose(
                self.data_values,
                rounded_data,
                rtol=0.0,
                atol=integer_tolerance,
            ):
                raise ValueError(
                    "Cash likelihood requires integer-valued observed counts "
                    "when require_integer_data=True."
                )

        self.include_data_term = bool(
            include_data_term
        )
        self.require_integer_data = bool(
            require_integer_data
        )
        self.integer_tolerance = float(
            integer_tolerance
        )

        if self.include_data_term:
            self._data_term = -float(
                np.sum(
                    gammaln(
                        self.data_values + 1.0
                    )
                )
            )
        else:
            self._data_term = 0.0

    def _log_likelihood(
        self,
        model_values: np.ndarray,
    ) -> float:
        """Evaluate the Cash log-likelihood.

        Parameters
        ----------
        model_values : ndarray
            Expected counts in the unmasked pixels.

        Returns
        -------
        float
            Cash or complete Poisson log-likelihood.

        Raises
        ------
        ValueError
            If any model expectation is negative.
        """
        if np.any(model_values < 0.0):
            raise ValueError(
                "Cash likelihood requires non-negative model expectations."
            )

        loglike = np.sum(
            xlogy(
                self.data_values,
                model_values,
            )
            - model_values
        )

        return float(
            loglike + self._data_term
        )