"""Dense-reference tests for the Vecchia correlated Gaussian likelihood."""

import numpy as np
import pytest

from baymorph.likelihood import GaussianImageLikelihood
from baymorph.likelihood.vecchia_gaussian import (
    VecchiaCorrelatedGaussianImageLikelihood,
    build_vecchia_factorization,
)


def _exponential_kernel(shape, rho_y=0.35, rho_x=0.25):
    half_y = shape[0] // 2
    half_x = shape[1] // 2
    yy, xx = np.indices(shape)
    return rho_y ** np.abs(yy - half_y) * rho_x ** np.abs(xx - half_x)


def _dense_correlation(mask, kernel):
    coordinates = np.argwhere(~mask)
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    half_y = kernel.shape[0] // 2
    half_x = kernel.shape[1] // 2
    output = np.zeros(differences.shape[:2])
    inside = (
        (np.abs(differences[..., 0]) <= half_y)
        & (np.abs(differences[..., 1]) <= half_x)
    )
    output[inside] = kernel[
        differences[..., 0][inside] + half_y,
        differences[..., 1][inside] + half_x,
    ]
    return output


def test_full_predecessor_factorization_matches_dense_masked_reference():
    """All predecessors must reproduce the exact dense Gaussian chain rule."""
    shape = (4, 5)
    mask = np.zeros(shape, dtype=bool)
    mask[0, 3] = True
    mask[2, 1] = True
    kernel = _exponential_kernel((7, 9))
    data = np.linspace(-0.7, 1.1, np.prod(shape)).reshape(shape)
    model = np.sin(np.arange(np.prod(shape)).reshape(shape)) * 0.1
    noise = np.linspace(0.8, 1.3, np.prod(shape)).reshape(shape)

    likelihood = VecchiaCorrelatedGaussianImageLikelihood(
        data=data,
        noise=noise,
        correlation_kernel=kernel,
        mask=mask,
        max_neighbors=64,
    )

    correlation = _dense_correlation(mask, kernel)
    noise_values = noise[~mask]
    covariance = noise_values[:, None] * correlation * noise_values[None, :]
    residual = data[~mask] - model[~mask]
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0.0
    quadratic = float(residual @ np.linalg.solve(covariance, residual))
    expected = -0.5 * (
        quadratic + logdet + residual.size * np.log(2.0 * np.pi)
    )

    assert likelihood(model) == pytest.approx(expected, abs=1.0e-11)
    assert likelihood.quadratic_form(model) == pytest.approx(
        quadratic,
        abs=1.0e-11,
    )
    assert np.sum(likelihood.quadratic_innovations(model)) == pytest.approx(
        quadratic,
        abs=1.0e-11,
    )
    assert likelihood.factorization.correlation_logdet == pytest.approx(
        np.linalg.slogdet(correlation)[1],
        abs=1.0e-11,
    )


def test_identity_kernel_matches_normalized_independent_gaussian():
    """An identity correlation kernel must reduce exactly to white noise."""
    data = np.arange(12, dtype=float).reshape(3, 4) / 10.0
    model = np.full((3, 4), 0.4)
    noise = np.linspace(0.7, 1.5, 12).reshape(3, 4)
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 2] = True

    independent = GaussianImageLikelihood(
        data=data,
        noise=noise,
        mask=mask,
        include_normalization=True,
    )
    vecchia = VecchiaCorrelatedGaussianImageLikelihood(
        data=data,
        noise=noise,
        correlation_kernel=np.ones((1, 1)),
        mask=mask,
        max_neighbors=64,
    )

    assert vecchia(model) == pytest.approx(independent(model), abs=1.0e-12)


def test_full_m_matches_dense_for_compact_banded_covariance():
    """Global predecessors must recover conditionals beyond kernel support."""
    data = np.linspace(-0.4, 0.8, 8)[None, :]
    model = np.linspace(0.1, -0.2, 8)[None, :]
    mask = np.zeros_like(data, dtype=bool)
    kernel = np.array([[0.2, 1.0, 0.2]])
    correlation = _dense_correlation(mask, kernel)
    residual = (data - model).ravel()
    sign, logdet = np.linalg.slogdet(correlation)
    assert sign > 0.0
    quadratic = float(residual @ np.linalg.solve(correlation, residual))
    expected = -0.5 * (
        quadratic + logdet + residual.size * np.log(2.0 * np.pi)
    )

    likelihood = VecchiaCorrelatedGaussianImageLikelihood(
        data=data,
        noise=1.0,
        correlation_kernel=kernel,
        mask=mask,
        max_neighbors=7,
    )

    assert likelihood(model) == pytest.approx(expected, abs=1.0e-12)
    last_start = likelihood.factorization.neighbor_indptr[-2]
    assert likelihood.factorization.neighbor_indices[last_start:].size == 7


def test_row_major_neighbors_are_previous_and_respect_exact_mask():
    """Stored neighbors must address only prior valid row-major pixels."""
    mask = np.array(
        [
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, False],
        ]
    )
    factorization = build_vecchia_factorization(
        mask,
        _exponential_kernel((5, 7)),
        max_neighbors=3,
    )

    assert np.array_equal(
        factorization.valid_flat_indices,
        np.flatnonzero(~mask.ravel(order="C")),
    )
    assert np.array_equal(
        np.diff(factorization.neighbor_indptr),
        np.minimum(np.arange(factorization.n_valid_pixels), 3),
    )
    for row in range(factorization.n_valid_pixels):
        start = factorization.neighbor_indptr[row]
        stop = factorization.neighbor_indptr[row + 1]
        active = factorization.neighbor_indices[start:stop]
        assert np.all(active < row)
        assert np.unique(active).size == active.size


def test_equal_distance_tie_prefers_most_recent_predecessor():
    """Equal-distance candidates use descending predecessor C-index."""
    factorization = build_vecchia_factorization(
        np.zeros((3, 3), dtype=bool),
        _exponential_kernel((5, 5)),
        max_neighbors=2,
    )
    start = factorization.neighbor_indptr[4]
    stop = factorization.neighbor_indptr[5]

    assert np.array_equal(
        factorization.neighbor_indices[start:stop],
        np.array([3, 1]),
    )


def test_factorization_arrays_cannot_be_made_writeable():
    """Cached statistical factors must be immutable after validation."""
    factorization = build_vecchia_factorization(
        np.zeros((3, 4), dtype=bool),
        _exponential_kernel((5, 7)),
        max_neighbors=4,
    )

    arrays = (
        factorization.valid_flat_indices,
        factorization.neighbor_indptr,
        factorization.neighbor_indices,
        factorization.weights,
        factorization.conditional_variances,
    )
    for values in arrays:
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.setflags(write=True)


def test_factorization_rejects_different_mask_or_kernel():
    """A persisted factorization must remain bound to its exact domain."""
    mask = np.zeros((3, 4), dtype=bool)
    kernel = _exponential_kernel((5, 7))
    factorization = build_vecchia_factorization(
        mask,
        kernel,
        max_neighbors=4,
    )
    different_mask = mask.copy()
    different_mask[1, 1] = True

    with pytest.raises(ValueError, match="does not match the fitting mask"):
        factorization.validate_domain(
            mask=different_mask,
            correlation_kernel=kernel,
        )

    different_kernel = kernel.copy()
    different_kernel[2, 1] *= 0.9
    different_kernel[2, -2] *= 0.9
    with pytest.raises(ValueError, match="fingerprint does not match"):
        factorization.validate_domain(
            mask=mask,
            correlation_kernel=different_kernel,
        )


def test_non_positive_conditional_variance_is_rejected():
    """An invalid empirical kernel must fail rather than receive jitter."""
    kernel = np.array([[0.99, 1.0, 0.99]])
    mask = np.zeros((1, 3), dtype=bool)

    with pytest.raises(ValueError, match="not safely positive"):
        build_vecchia_factorization(
            mask,
            kernel,
            max_neighbors=2,
        )


def test_non_exact_kernel_contract_is_rejected():
    """The target variance and point symmetry must be exact."""
    with pytest.raises(ValueError, match="exactly unit"):
        build_vecchia_factorization(
            np.zeros((2, 2), dtype=bool),
            np.array([[0.2, 1.0 + 1.0e-14, 0.2]]),
            max_neighbors=2,
        )

    with pytest.raises(ValueError, match="bitwise covariance symmetry"):
        build_vecchia_factorization(
            np.zeros((2, 2), dtype=bool),
            np.array([[0.2, 1.0, 0.2 + 1.0e-14]]),
            max_neighbors=2,
        )


def test_approximation_metadata_is_explicit():
    """Results must identify the approximation and its normalization."""
    likelihood = VecchiaCorrelatedGaussianImageLikelihood(
        data=np.zeros((2, 3)),
        noise=1.0,
        correlation_kernel=_exponential_kernel((3, 5)),
        max_neighbors=2,
    )

    metadata = likelihood.approximation_metadata
    assert metadata["approximate"] is True
    assert metadata["approximation"] == "nearest_neighbor_conditional_gaussian"
    assert metadata["ordering"] == "row_major"
    assert metadata["max_neighbors"] == 2
    assert metadata["normalization_complete"] is True
    assert metadata["evidence_scope"] == "approximate_absolute"
