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
