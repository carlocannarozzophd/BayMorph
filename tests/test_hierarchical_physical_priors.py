"""Tests for physical-prior-preserving hierarchical dependence."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.hierarchical import (
    equicorrelated_gaussian_latents,
    physical_prior_copula_values,
)
from baymorph.fitting.priors import (
    NormalPrior,
    UniformPrior,
)


def test_zero_correlation_preserves_independent_latents():
    latents = np.array(
        [-1.2, 0.3, 2.1],
        dtype=float,
    )

    correlated = equicorrelated_gaussian_latents(
        latents,
        correlation=0.0,
    )

    assert correlated == pytest.approx(
        latents
    )


def test_unit_correlation_produces_common_standardized_latent():
    latents = np.array(
        [-1.2, 0.3, 2.1],
        dtype=float,
    )

    correlated = equicorrelated_gaussian_latents(
        latents,
        correlation=1.0,
    )

    expected = float(
        np.sum(latents)
        / np.sqrt(latents.size)
    )

    assert correlated == pytest.approx(
        np.full(
            latents.size,
            expected,
        )
    )


@pytest.mark.parametrize(
    "correlation",
    [
        0.0,
        0.25,
        0.75,
        1.0,
    ],
)
def test_uniform_physical_prior_respects_bounds(
    correlation,
):
    prior = UniformPrior(
        lower=0.1,
        upper=6.0,
    )

    values = physical_prior_copula_values(
        prior,
        np.array(
            [-2.0, -0.5, 0.5, 2.0],
            dtype=float,
        ),
        correlation,
    )

    assert np.all(
        values >= 0.1
    )
    assert np.all(
        values <= 6.0
    )


def test_unit_correlation_produces_identical_physical_values():
    prior = UniformPrior(
        lower=0.1,
        upper=6.0,
    )

    values = physical_prior_copula_values(
        prior,
        np.array(
            [-1.5, 0.2, 1.8],
            dtype=float,
        ),
        correlation=1.0,
    )

    assert values == pytest.approx(
        np.full(
            3,
            values[0],
        )
    )


@pytest.mark.parametrize(
    "correlation",
    [
        0.0,
        0.3,
        0.7,
        0.95,
    ],
)
def test_uniform_marginal_prior_is_preserved(
    correlation,
):
    rng = np.random.default_rng(
        812734
    )

    n_draws = 30000
    n_observations = 4

    prior = UniformPrior(
        lower=0.1,
        upper=6.0,
    )

    samples = np.empty(
        (
            n_draws,
            n_observations,
        ),
        dtype=float,
    )

    for draw in range(
        n_draws
    ):
        samples[draw] = physical_prior_copula_values(
            prior,
            rng.normal(
                size=n_observations
            ),
            correlation,
        )

    expected_mean = (
        0.1 + 6.0
    ) / 2.0
    expected_variance = (
        (6.0 - 0.1) ** 2
        / 12.0
    )

    for observation in range(
        n_observations
    ):
        marginal = samples[
            :,
            observation,
        ]

        assert np.mean(
            marginal
        ) == pytest.approx(
            expected_mean,
            abs=0.035,
        )

        assert np.var(
            marginal
        ) == pytest.approx(
            expected_variance,
            abs=0.08,
        )

        quantiles = np.quantile(
            marginal,
            [
                0.1,
                0.5,
                0.9,
            ],
        )

        expected_quantiles = np.array(
            [
                0.1 + 0.1 * 5.9,
                0.1 + 0.5 * 5.9,
                0.1 + 0.9 * 5.9,
            ]
        )

        assert quantiles == pytest.approx(
            expected_quantiles,
            abs=0.06,
        )


@pytest.mark.parametrize(
    "correlation",
    [
        0.0,
        0.4,
        0.9,
    ],
)
def test_normal_marginal_prior_is_preserved(
    correlation,
):
    rng = np.random.default_rng(
        391247
    )

    prior = NormalPrior(
        mean=3.0,
        sigma=1.2,
    )

    samples = np.asarray(
        [
            physical_prior_copula_values(
                prior,
                rng.normal(
                    size=3
                ),
                correlation,
            )
            for _ in range(
                30000
            )
        ]
    )

    for observation in range(
        samples.shape[1]
    ):
        marginal = samples[
            :,
            observation,
        ]

        assert np.mean(
            marginal
        ) == pytest.approx(
            3.0,
            abs=0.03,
        )

        assert np.std(
            marginal
        ) == pytest.approx(
            1.2,
            abs=0.03,
        )


def test_increasing_latent_correlation_increases_dependence():
    rng = np.random.default_rng(
        127381
    )

    n_draws = 30000

    prior = UniformPrior(
        lower=0.1,
        upper=6.0,
    )

    low = np.asarray(
        [
            physical_prior_copula_values(
                prior,
                rng.normal(
                    size=2
                ),
                correlation=0.1,
            )
            for _ in range(
                n_draws
            )
        ]
    )

    rng = np.random.default_rng(
        127381
    )

    high = np.asarray(
        [
            physical_prior_copula_values(
                prior,
                rng.normal(
                    size=2
                ),
                correlation=0.9,
            )
            for _ in range(
                n_draws
            )
        ]
    )

    low_correlation = float(
        np.corrcoef(
            low[:, 0],
            low[:, 1],
        )[0, 1]
    )
    high_correlation = float(
        np.corrcoef(
            high[:, 0],
            high[:, 1],
        )[0, 1]
    )

    assert high_correlation > low_correlation
    assert high_correlation > 0.8
