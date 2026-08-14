"""Tests for general correlation-matrix hierarchical coupling."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.hierarchical import (
    correlated_gaussian_latents,
    equicorrelated_gaussian_latents,
    physical_prior_copula_values_from_matrix,
    squared_exponential_correlation_matrix,
)
from baymorph.fitting.priors import UniformPrior


def test_general_matrix_matches_equicorrelation_transform():
    latents = np.array(
        [-1.2, 0.3, 1.7, 0.8],
        dtype=float,
    )
    rho = 0.65

    matrix = (
        (1.0 - rho)
        * np.eye(
            latents.size
        )
        + rho
        * np.ones(
            (
                latents.size,
                latents.size,
            )
        )
    )

    general = correlated_gaussian_latents(
        latents,
        matrix,
    )

    analytic = equicorrelated_gaussian_latents(
        latents,
        rho,
    )

    assert general == pytest.approx(
        analytic,
        abs=1.0e-12,
    )


def test_general_transform_recovers_target_covariance():
    target = np.array(
        [
            [1.0, 0.8, 0.4, 0.1],
            [0.8, 1.0, 0.5, 0.2],
            [0.4, 0.5, 1.0, 0.6],
            [0.1, 0.2, 0.6, 1.0],
        ],
        dtype=float,
    )

    rng = np.random.default_rng(
        41392
    )

    draws = np.asarray(
        [
            correlated_gaussian_latents(
                rng.normal(
                    size=4
                ),
                target,
            )
            for _ in range(
                15000
            )
        ]
    )

    empirical = np.corrcoef(
        draws,
        rowvar=False,
    )

    assert empirical == pytest.approx(
        target,
        abs=0.025,
    )


def test_squared_exponential_kernel_is_covariate_aware():
    covariate = np.array(
        [-1.0, -0.5, 0.0, 1.0],
        dtype=float,
    )

    matrix = squared_exponential_correlation_matrix(
        covariate,
        correlation=0.8,
        length_scale=0.6,
    )

    assert np.diag(
        matrix
    ) == pytest.approx(
        np.ones(4)
    )

    assert matrix[0, 1] > matrix[0, 2]
    assert matrix[0, 2] > matrix[0, 3]


def test_zero_smooth_amplitude_is_independent():
    matrix = squared_exponential_correlation_matrix(
        np.array(
            [-1.0, 0.0, 1.0]
        ),
        correlation=0.0,
        length_scale=1.0,
    )

    assert matrix == pytest.approx(
        np.eye(3)
    )


def test_large_length_scale_approaches_equicorrelation():
    rho = 0.7

    matrix = squared_exponential_correlation_matrix(
        np.array(
            [-1.0, 0.0, 1.0]
        ),
        correlation=rho,
        length_scale=1.0e6,
    )

    expected = (
        (1.0 - rho)
        * np.eye(3)
        + rho
        * np.ones(
            (3, 3)
        )
    )

    assert matrix == pytest.approx(
        expected,
        abs=2.0e-12,
    )


def test_general_copula_preserves_physical_marginals():
    prior = UniformPrior(
        lower=0.1,
        upper=6.0,
    )

    covariate = np.array(
        [-1.0, -0.5, 0.0, 0.5, 1.0],
        dtype=float,
    )

    matrix = squared_exponential_correlation_matrix(
        covariate,
        correlation=0.85,
        length_scale=0.5,
    )

    rng = np.random.default_rng(
        82931
    )

    samples = np.asarray(
        [
            physical_prior_copula_values_from_matrix(
                prior,
                rng.normal(
                    size=5
                ),
                matrix,
            )
            for _ in range(
                12000
            )
        ]
    )

    expected_mean = (
        0.1 + 6.0
    ) / 2.0

    expected_quantiles = np.array(
        [
            0.1 + 0.1 * 5.9,
            0.1 + 0.5 * 5.9,
            0.1 + 0.9 * 5.9,
        ]
    )

    for observation in range(5):
        marginal = samples[
            :,
            observation,
        ]

        assert np.mean(
            marginal
        ) == pytest.approx(
            expected_mean,
            abs=0.05,
        )

        assert np.quantile(
            marginal,
            [
                0.1,
                0.5,
                0.9,
            ],
        ) == pytest.approx(
            expected_quantiles,
            abs=0.08,
        )


@pytest.mark.parametrize(
    "matrix",
    [
        np.array(
            [
                [1.0, 0.5],
                [0.1, 1.0],
            ]
        ),
        np.array(
            [
                [1.0, 2.0],
                [2.0, 1.0],
            ]
        ),
        np.array(
            [
                [0.9, 0.2],
                [0.2, 1.0],
            ]
        ),
    ],
)
def test_general_correlation_rejects_invalid_matrix(
    matrix,
):
    with pytest.raises(
        ValueError,
    ):
        correlated_gaussian_latents(
            np.array(
                [0.0, 1.0]
            ),
            matrix,
        )
