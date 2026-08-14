"""Physical-prior-preserving hierarchical dependence."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from baymorph.fitting.priors import Prior


def correlated_gaussian_latents(
    independent_latents: np.ndarray,
    correlation_matrix: np.ndarray,
) -> np.ndarray:
    """Map independent standard normals to a target correlation matrix.

    Parameters
    ----------
    independent_latents : numpy.ndarray
        One independent standard-normal coordinate per observation.
    correlation_matrix : numpy.ndarray
        Symmetric positive-semidefinite correlation matrix with unit diagonal.

    Returns
    -------
    numpy.ndarray
        Correlated standard-normal coordinates.
    """
    values = np.asarray(
        independent_latents,
        dtype=float,
    )

    matrix = np.asarray(
        correlation_matrix,
        dtype=float,
    )

    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "Gaussian latent coordinates must be a non-empty "
            "one-dimensional array."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Gaussian latent coordinates must be finite."
        )

    expected_shape = (
        values.size,
        values.size,
    )

    if matrix.shape != expected_shape:
        raise ValueError(
            "Correlation matrix shape must match the number "
            "of Gaussian latent coordinates."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "Correlation matrix must contain only finite values."
        )

    if not np.allclose(
        matrix,
        matrix.T,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Correlation matrix must be symmetric."
        )

    if not np.allclose(
        np.diag(matrix),
        1.0,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError(
            "Correlation matrix must have unit diagonal."
        )

    eigenvalues, eigenvectors = np.linalg.eigh(
        matrix
    )

    if float(np.min(eigenvalues)) < -1.0e-10:
        raise ValueError(
            "Correlation matrix must be positive semidefinite."
        )

    eigenvalues = np.clip(
        eigenvalues,
        0.0,
        None,
    )

    square_root = (
        eigenvectors
        * np.sqrt(eigenvalues)
    ) @ eigenvectors.T

    correlated = square_root @ values

    if not np.all(np.isfinite(correlated)):
        raise ValueError(
            "Gaussian correlation transform produced "
            "non-finite latent coordinates."
        )

    return np.asarray(
        correlated,
        dtype=float,
    )


def equicorrelated_gaussian_latents(
    independent_latents: np.ndarray,
    correlation: float,
) -> np.ndarray:
    """Construct equicorrelated standard-normal latent coordinates."""
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
    mean_latent = float(
        np.mean(values)
    )

    orthogonal_scale = float(
        np.sqrt(
            1.0 - rho
        )
    )

    common_scale = float(
        np.sqrt(
            1.0
            + (n_observations - 1) * rho
        )
    )

    correlated = (
        orthogonal_scale
        * (
            values
            - mean_latent
        )
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


def squared_exponential_correlation_matrix(
    covariate: np.ndarray,
    *,
    correlation: float,
    length_scale: float,
) -> np.ndarray:
    """Build a smooth covariate-dependent correlation matrix.

    Parameters
    ----------
    covariate : numpy.ndarray
        One finite covariate value per observation.
    correlation : float
        Maximum smooth-component amplitude in [0, 1].
    length_scale : float
        Positive covariate correlation length.

    Returns
    -------
    numpy.ndarray
        Correlation matrix with unit diagonal.
    """
    values = np.asarray(
        covariate,
        dtype=float,
    )

    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "Covariate values must form a non-empty "
            "one-dimensional array."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Covariate values must be finite."
        )

    rho = float(
        correlation
    )

    if (
        not np.isfinite(rho)
        or rho < 0.0
        or rho > 1.0
    ):
        raise ValueError(
            "Correlation amplitude must be finite "
            "and lie in [0, 1]."
        )

    scale = float(
        length_scale
    )

    if (
        not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError(
            "Correlation length scale must be finite "
            "and positive."
        )

    separation = (
        values[:, None]
        - values[None, :]
    ) / scale

    smooth = np.exp(
        -0.5
        * separation**2
    )

    matrix = (
        (1.0 - rho)
        * np.eye(
            values.size,
            dtype=float,
        )
        + rho
        * smooth
    )

    return np.asarray(
        matrix,
        dtype=float,
    )


def _physical_values_from_correlated_latents(
    prior: Prior,
    correlated_latents: np.ndarray,
) -> np.ndarray:
    """Map standard-normal marginals through a physical prior quantile."""
    if not isinstance(
        prior,
        Prior,
    ):
        raise TypeError(
            "prior must be a Prior instance."
        )

    if prior.is_fixed:
        raise ValueError(
            "A fixed prior cannot define hierarchical variation."
        )

    unit_values = ndtr(
        correlated_latents
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


def physical_prior_copula_values(
    prior: Prior,
    independent_latents: np.ndarray,
    correlation: float,
) -> np.ndarray:
    """Map equicorrelated Gaussian latents through a physical prior."""
    correlated = equicorrelated_gaussian_latents(
        independent_latents,
        correlation,
    )

    return _physical_values_from_correlated_latents(
        prior,
        correlated,
    )


def physical_prior_copula_values_from_matrix(
    prior: Prior,
    independent_latents: np.ndarray,
    correlation_matrix: np.ndarray,
) -> np.ndarray:
    """Map a general Gaussian copula through a physical marginal prior."""
    correlated = correlated_gaussian_latents(
        independent_latents,
        correlation_matrix,
    )

    return _physical_values_from_correlated_latents(
        prior,
        correlated,
    )
