import numpy as np
import pytest

from baymorph.validation import (
    bias,
    coverage_fraction,
    fractional_error,
    interval_contains_truth,
    parameter_bias,
    scatter,
)


def test_scalar_recovery_metrics():
    assert parameter_bias(12.0, 10.0) == 2.0
    assert fractional_error(12.0, 10.0) == 0.2
    assert interval_contains_truth(8.0, 12.0, 10.0)


def test_fractional_error_rejects_zero_truth():
    with pytest.raises(ValueError):
        fractional_error(1.0, 0.0)


def test_campaign_metrics():
    estimates = np.array([9.0, 11.0, 13.0])
    truths = np.array([10.0, 10.0, 10.0])

    assert np.isclose(bias(estimates, truths), 1.0)
    assert np.isclose(scatter(estimates, truths), np.sqrt(8.0 / 3.0))
    assert coverage_fraction(np.array([True, False, True])) == pytest.approx(2 / 3)


def test_weighted_posterior_rank():
    from baymorph.validation.metrics import weighted_posterior_rank

    samples = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([1.0, 1.0, 2.0, 0.0])

    assert weighted_posterior_rank(
        samples,
        weights,
        truth=3.0,
    ) == pytest.approx(0.5)


def test_weighted_credible_interval():
    from baymorph.validation.metrics import weighted_credible_interval

    samples = np.linspace(0.0, 1.0, 10001)
    weights = np.ones_like(samples)

    lower, upper = weighted_credible_interval(
        samples,
        weights,
        probability=0.68,
    )

    assert lower == pytest.approx(0.16, abs=2.0e-4)
    assert upper == pytest.approx(0.84, abs=2.0e-4)


def test_weighted_posterior_metrics_reject_invalid_weights():
    from baymorph.validation.metrics import (
        weighted_credible_interval,
        weighted_posterior_rank,
    )

    samples = np.array([1.0, 2.0, 3.0])
    invalid_weights = np.array([1.0, -1.0, 1.0])

    with pytest.raises(ValueError, match="non-negative"):
        weighted_posterior_rank(
            samples,
            invalid_weights,
            truth=2.0,
        )

    with pytest.raises(ValueError, match="non-negative"):
        weighted_credible_interval(
            samples,
            invalid_weights,
            probability=0.68,
        )
