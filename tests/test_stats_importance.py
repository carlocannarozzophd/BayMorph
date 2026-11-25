"""Tests of Pareto-smoothed importance sampling against known answers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from baymorph.stats.importance import (
    generalized_pareto_fit,
    pareto_smoothed_log_weights,
)


def _gaussian_log_weights(proposal_sigma: float, count: int, seed: int):
    rng = np.random.default_rng(seed)
    draws = rng.normal(0.0, proposal_sigma, count)
    log_weights = stats.norm.logpdf(draws) - stats.norm.logpdf(
        draws, 0.0, proposal_sigma
    )

    return draws, log_weights


def test_constant_weights_are_bounded_and_unchanged():
    result = pareto_smoothed_log_weights(np.full(500, 3.7))

    assert result.pareto_k == 0.0
    assert result.effective_sample_size == pytest.approx(500.0)
    assert np.allclose(result.weights, 1.0 / 500.0)


def test_weights_are_normalized_and_zero_weights_are_kept():
    _, log_weights = _gaussian_log_weights(1.5, 1000, seed=3)
    log_weights[:10] = -np.inf
    result = pareto_smoothed_log_weights(log_weights)

    assert np.sum(result.weights) == pytest.approx(1.0)
    assert np.all(result.weights[:10] == 0.0)
    assert result.log_weights.shape == (1000,)


def test_wide_proposal_gives_bounded_weights():
    # A proposal wider than the target bounds the weights: k is negative.
    shapes = [
        pareto_smoothed_log_weights(_gaussian_log_weights(2.0, 4000, seed)[1]).pareto_k
        for seed in range(10)
    ]

    assert np.median(shapes) < 0.0


def test_narrow_proposal_is_diagnosed():
    # For a Gaussian target and proposal the tail shape is 1 - s_q^2 / s_p^2:
    # 0.36 at s_q = 0.8 and 0.75 at s_q = 0.5.
    moderate = [
        pareto_smoothed_log_weights(_gaussian_log_weights(0.8, 4000, seed)[1]).pareto_k
        for seed in range(20)
    ]
    heavy = [
        pareto_smoothed_log_weights(_gaussian_log_weights(0.5, 4000, seed)[1]).pareto_k
        for seed in range(20)
    ]

    assert np.median(moderate) == pytest.approx(0.36, abs=0.08)
    assert np.median(heavy) > 0.55


def test_smoothed_estimate_of_a_moment():
    estimates = []

    for seed in range(20):
        draws, log_weights = _gaussian_log_weights(0.8, 4000, seed)
        weights = pareto_smoothed_log_weights(log_weights).weights
        estimates.append(np.sum(weights * draws**2))

    assert np.mean(estimates) == pytest.approx(1.0, abs=0.02)


def test_generalized_pareto_fit_recovers_a_known_shape():
    rng = np.random.default_rng(11)
    values = np.sort(stats.genpareto.rvs(0.3, scale=2.0, size=4000, random_state=rng))
    shape, scale = generalized_pareto_fit(values)

    assert shape == pytest.approx(0.3, abs=0.06)
    assert scale == pytest.approx(2.0, rel=0.1)


@pytest.mark.parametrize(
    "values",
    [
        np.array([0.0, np.nan, 1.0]),
        np.array([0.0, np.inf, 1.0]),
        np.full(10, -np.inf),
        np.array([1.0]),
        np.zeros((2, 2)),
    ],
)
def test_invalid_log_weights_are_refused(values):
    with pytest.raises(ValueError):
        pareto_smoothed_log_weights(values)


def test_pareto_fit_refuses_invalid_exceedances():
    with pytest.raises(ValueError):
        generalized_pareto_fit(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError):
        generalized_pareto_fit(np.array([3.0, 2.0, 1.0, 4.0, 5.0, 6.0]))

    with pytest.raises(ValueError):
        generalized_pareto_fit(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
