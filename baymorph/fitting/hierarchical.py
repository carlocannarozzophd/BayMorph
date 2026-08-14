"""Physical-prior-preserving hierarchical dependence."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from baymorph.fitting.priors import Prior


def equicorrelated_gaussian_latents(
    independent_latents: np.ndarray,
    correlation: float,
) -> np.ndarray:
    """Construct equicorrelated standard-normal latent coordinates.

    Parameters
    ----------
    independent_latents : numpy.ndarray
        One independent standard-normal coordinate per observation.
    correlation : float
        Common pairwise correlation in latent Gaussian space.

    Returns
    -------
    numpy.ndarray
        Correlated standard-normal coordinates.

    Notes
    -----
    The target correlation matrix is

    ``C = (1 - rho) I + rho J``.

    The implementation applies its analytic symmetric square root without
    explicitly constructing or factorizing the matrix.
    """
    values = np.asarray(
        independent_latents,
        dtype=float,
    )

    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "Hierarchical latent coordinates must be a non-empty "
            "one-dimensional array."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Hierarchical latent coordinates must be finite."
        )

    rho = float(correlation)

    if (
        not np.isfinite(rho)
        or rho < 0.0
        or rho > 1.0
    ):
        raise ValueError(
            "Hierarchical latent correlation must be finite "
            "and lie in [0, 1]."
        )

    n_observations = values.size
    mean_latent = float(np.mean(values))

    orthogonal_scale = float(
        np.sqrt(1.0 - rho)
    )
    common_scale = float(
        np.sqrt(
            1.0
            + (n_observations - 1) * rho
        )
    )

    correlated = (
        orthogonal_scale
        * (values - mean_latent)
        + common_scale
        * mean_latent
    )

    if not np.all(np.isfinite(correlated)):
        raise ValueError(
            "Hierarchical Gaussian dependence produced "
            "non-finite latent coordinates."
        )

    return np.asarray(
        correlated,
        dtype=float,
    )


def physical_prior_copula_values(
    prior: Prior,
    independent_latents: np.ndarray,
    correlation: float,
) -> np.ndarray:
    """Map hierarchical dependence through a physical marginal prior.

    The observation-level values retain ``prior`` as their marginal
    distribution for every allowed value of ``correlation``.
    """
    if not isinstance(prior, Prior):
        raise TypeError(
            "prior must be a Prior instance."
        )

    if prior.is_fixed:
        raise ValueError(
            "A fixed prior cannot define hierarchical variation."
        )

    correlated = equicorrelated_gaussian_latents(
        independent_latents,
        correlation,
    )

    unit_values = ndtr(
        correlated
    )

    physical = np.asarray(
        [
            prior.transform(
                float(unit_value)
            )
            for unit_value in unit_values
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(physical)):
        raise ValueError(
            "Physical prior transform produced non-finite "
            "hierarchical values."
        )

    return physical
