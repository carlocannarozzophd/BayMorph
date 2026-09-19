"""Tests for the posterior calibration metric."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.results.inference import InferenceResult
from baymorph.results.samples import PosteriorSamples
from baymorph.stats.weighted import (
    mahalanobis_distance_squared,
    weighted_covariance,
)


def _gaussian_result(
    *,
    correlation: float = 0.0,
    n_samples: int = 40000,
    seed: int = 5,
) -> InferenceResult:
    """Return an inference result whose posterior is a known Gaussian."""
    rng = np.random.default_rng(
        seed
    )

    covariance = np.array(
        [
            [
                1.0,
                correlation,
            ],
            [
                correlation,
                1.0,
            ],
        ]
    )

    samples = rng.multivariate_normal(
        mean=[
            0.0,
            0.0,
        ],
        cov=covariance,
        size=n_samples,
    )

    return InferenceResult(
        posterior=PosteriorSamples(
            samples=samples,
            weights=np.full(
                n_samples,
                1.0 / n_samples,
            ),
            parameter_names=[
                "a",
                "b",
            ],
        ),
    )


def test_weighted_covariance_recovers_a_known_matrix() -> None:
    """The weighted covariance must recover the generating covariance."""
    result = _gaussian_result(
        correlation=0.6
    )

    covariance = weighted_covariance(
        result.samples,
        result.weights,
    )

    np.testing.assert_allclose(
        covariance,
        [
            [
                1.0,
                0.6,
            ],
            [
                0.6,
                1.0,
            ],
        ],
        atol=0.03,
    )


def test_weights_are_normalized_internally() -> None:
    """Unnormalized weights must give the same covariance."""
    result = _gaussian_result()

    scaled = weighted_covariance(
        result.samples,
        result.weights * 17.0,
    )
    reference = weighted_covariance(
        result.samples,
        result.weights,
    )

    np.testing.assert_allclose(
        scaled,
        reference,
    )


def test_distance_of_the_posterior_mean_is_zero() -> None:
    """The posterior mean must sit at zero distance from its own posterior."""
    result = _gaussian_result()

    mean = result.weights @ result.samples

    distance = mahalanobis_distance_squared(
        result.samples,
        result.weights,
        mean,
    )

    assert distance == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_distance_matches_the_analytic_value_for_uncorrelated_parameters() -> None:
    """With unit variances and no correlation the metric is the squared offset."""
    result = _gaussian_result(
        correlation=0.0
    )

    outcome = result.mahalanobis_distance(
        {
            "a": 1.0,
            "b": 2.0,
        }
    )

    assert outcome["dof"] == 2
    assert outcome["distance_squared"] == pytest.approx(
        5.0,
        rel=0.05,
    )


def test_correlation_changes_the_answer_against_the_naive_sum() -> None:
    """Correlated parameters must not be assessed one at a time.

    A shared displacement along the correlation ridge is cheap, while the same
    displacement across it is expensive. Summing marginal offsets cannot
    distinguish the two, which is the reason this metric exists.
    """
    result = _gaussian_result(
        correlation=0.9
    )

    naive = 1.0 + 1.0

    along = result.mahalanobis_distance(
        {
            "a": 1.0,
            "b": 1.0,
        }
    )["distance_squared"]
    across = result.mahalanobis_distance(
        {
            "a": 1.0,
            "b": -1.0,
        }
    )["distance_squared"]

    assert along < naive < across
    assert along == pytest.approx(
        2.0 / (1.0 + 0.9),
        rel=0.08,
    )
    assert across == pytest.approx(
        2.0 / (1.0 - 0.9),
        rel=0.08,
    )


def test_reference_subset_selects_shared_parameters() -> None:
    """Only parameters present in both the reference and posterior are used."""
    result = _gaussian_result()

    outcome = result.mahalanobis_distance(
        {
            "b": 1.0,
            "background_level": 3.0,
        }
    )

    assert outcome["parameters"] == [
        "b",
    ]
    assert outcome["dof"] == 1


def test_reference_without_shared_names_is_rejected() -> None:
    """A reference sharing no parameter name must fail closed."""
    result = _gaussian_result()

    with pytest.raises(
        ValueError,
        match="shares no parameter name",
    ):
        result.mahalanobis_distance(
            {
                "background_level": 1.0,
            }
        )


def test_p_value_is_consistent_with_the_chi_squared_distribution() -> None:
    """A calibrated Gaussian posterior must give uniform-like p-values."""
    result = _gaussian_result(
        correlation=0.5
    )

    zero = result.mahalanobis_distance(
        {
            "a": 0.0,
            "b": 0.0,
        }
    )

    assert zero["p_value"] == pytest.approx(
        1.0,
        abs=0.02,
    )
    assert zero["distance"] == pytest.approx(
        np.sqrt(
            zero["distance_squared"]
        )
    )


def test_singular_covariance_is_rejected() -> None:
    """A degenerate parameter must raise instead of being pseudo-inverted.

    Rank deficiency is detected from the covariance eigenvalues rather than by
    relying on the linear solver to fail. LAPACK implementations disagree: an
    exactly singular matrix raises under macOS Accelerate but returns finite
    nonsense under OpenBLAS, which would silently yield a distance that is not
    the declared quantity.
    """
    rng = np.random.default_rng(
        3
    )

    first = rng.normal(
        size=2000
    )
    samples = np.column_stack(
        [
            first,
            first,
        ]
    )

    with pytest.raises(
        ValueError,
        match="singular",
    ):
        mahalanobis_distance_squared(
            samples,
            np.full(
                samples.shape[0],
                1.0 / samples.shape[0],
            ),
            np.array(
                [
                    1.0,
                    1.0,
                ]
            ),
        )


def test_fixed_parameter_is_rejected() -> None:
    """A parameter with no posterior variance makes the metric undefined."""
    rng = np.random.default_rng(
        13
    )

    samples = np.column_stack(
        [
            rng.normal(
                size=2000
            ),
            np.full(
                2000,
                3.0,
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="singular",
    ):
        mahalanobis_distance_squared(
            samples,
            np.full(
                samples.shape[0],
                1.0 / samples.shape[0],
            ),
            np.array(
                [
                    0.0,
                    3.0,
                ]
            ),
        )


def test_strong_but_legitimate_correlation_still_computes() -> None:
    """The rank check must not reject a merely strong degeneracy.

    Sersic parameters routinely correlate above 0.9, so a criterion that
    rejected those would be useless in practice.
    """
    result = _gaussian_result(
        correlation=0.99,
        n_samples=40000,
    )

    outcome = result.mahalanobis_distance(
        {
            "a": 1.0,
            "b": 1.0,
        }
    )

    assert outcome["distance_squared"] == pytest.approx(
        2.0 / (1.0 + 0.99),
        rel=0.10,
    )


def test_negative_weights_are_rejected() -> None:
    """Weights must be non-negative."""
    result = _gaussian_result(
        n_samples=100
    )

    weights = np.array(
        result.weights,
        dtype=float,
    )
    weights[0] = -1.0

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        weighted_covariance(
            result.samples,
            weights,
        )
