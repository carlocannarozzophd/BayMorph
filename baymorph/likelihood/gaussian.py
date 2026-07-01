"""Gaussian image likelihood."""

import numpy as np


def gaussian_log_likelihood(
    data: np.ndarray,
    model: np.ndarray,
    noise: np.ndarray,
    mask: np.ndarray | None = None,
    include_normalization: bool = True,
) -> float:
    """Compute the Gaussian log-likelihood for image data.

    Parameters
    ----------
    data : ndarray
        Observed image.
    model : ndarray
        Model image evaluated on the same pixel grid as ``data``.
    noise : ndarray
        Per-pixel Gaussian standard deviation. All values must be strictly
        positive.
    mask : ndarray of bool, optional
        Boolean mask identifying pixels to exclude from the likelihood
        evaluation.
    include_normalization : bool, default=True
        If True, include the Gaussian normalization term in the
        log-likelihood. If False, only the chi-squared term is evaluated.

    Returns
    -------
    float
        Gaussian log-likelihood.

    Raises
    ------
    ValueError
        If the input arrays do not have matching shapes, if any noise value is
        non-positive, or if the mask has an incompatible shape.
    """
    data = np.asarray(data, dtype=float)
    model = np.asarray(model, dtype=float)
    noise = np.asarray(noise, dtype=float)

    if data.shape != model.shape:
        raise ValueError("Data and model must have the same shape.")

    if data.shape != noise.shape:
        raise ValueError("Data and noise must have the same shape.")

    if np.any(noise <= 0):
        raise ValueError("Noise values must be positive.")

    residual = data - model

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

        if mask.shape != data.shape:
            raise ValueError("Mask must have the same shape as data.")

        valid = ~mask
        residual = residual[valid]
        noise = noise[valid]

    chi2 = np.sum((residual / noise) ** 2)

    loglike = -0.5 * chi2

    if include_normalization:
        loglike -= 0.5 * np.sum(
            np.log(2.0 * np.pi * noise**2)
        )

    return float(loglike)