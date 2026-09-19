"""Tests for the truncation-corrected Gaussian scale estimator."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.stats.robust import (
    truncated_gaussian_sigma,
    truncated_variance_ratio,
)


def test_variance_ratio_tends_to_one_for_wide_thresholds() -> None:
    """A threshold far in the tails must leave the variance untouched."""
    assert truncated_variance_ratio(
        8.0
    ) == pytest.approx(
        1.0,
        abs=1e-6,
    )
    assert truncated_variance_ratio(
        np.inf
    ) == 1.0


def test_variance_ratio_matches_the_analytic_value_at_three_sigma() -> None:
    """Clipping at three sigma must remove a known fraction of the variance."""
    ratio = truncated_variance_ratio(
        3.0
    )

    assert ratio == pytest.approx(
        0.97333692,
        rel=1e-6,
    )

    # A plain standard deviation of the clipped sample is low by this much.
    assert np.sqrt(
        ratio
    ) == pytest.approx(
        0.98658,
        rel=1e-4,
    )


def test_variance_ratio_rejects_a_non_positive_threshold() -> None:
    """A non-positive threshold has no meaning."""
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        truncated_variance_ratio(
            0.0
        )


@pytest.mark.parametrize(
    "relative_threshold",
    [
        2.0,
        2.5,
        3.0,
        4.0,
    ],
)
def test_correction_recovers_a_known_sigma(
    relative_threshold: float,
) -> None:
    """The parent sigma must be recovered from a truncated sample."""
    rng = np.random.default_rng(
        17
    )
    sigma = 2.0

    estimates = []

    for _ in range(60):
        values = rng.normal(
            0.0,
            sigma,
            20000,
        )
        threshold = relative_threshold * sigma
        estimates.append(
            truncated_gaussian_sigma(
                values[
                    np.abs(values) < threshold
                ],
                threshold,
            )
        )

    assert float(
        np.mean(estimates)
    ) == pytest.approx(
        sigma,
        rel=0.004,
    )


def test_uncorrected_standard_deviation_would_be_biased() -> None:
    """The correction must change the answer, not merely pass it through.

    Without this the recovery test above could be satisfied by returning the
    plain standard deviation.
    """
    rng = np.random.default_rng(
        23
    )
    sigma = 2.0
    threshold = 3.0 * sigma

    values = rng.normal(
        0.0,
        sigma,
        200000,
    )
    clipped = values[
        np.abs(values) < threshold
    ]

    uncorrected = float(
        np.std(clipped)
    )
    corrected = truncated_gaussian_sigma(
        clipped,
        threshold,
    )

    assert uncorrected < 0.995 * sigma
    assert corrected > uncorrected
    assert abs(
        corrected / sigma - 1.0
    ) < abs(
        uncorrected / sigma - 1.0
    )


def test_unclipped_sample_is_returned_unchanged() -> None:
    """An infinite threshold means nothing was truncated."""
    rng = np.random.default_rng(
        29
    )

    values = rng.normal(
        0.0,
        2.0,
        5000,
    )

    assert truncated_gaussian_sigma(
        values,
        np.inf,
    ) == pytest.approx(
        float(
            np.std(values)
        )
    )


def test_empty_and_degenerate_samples() -> None:
    """Degenerate input must not produce a misleading scale."""
    assert np.isnan(
        truncated_gaussian_sigma(
            [],
            3.0,
        )
    )
    assert truncated_gaussian_sigma(
        np.zeros(100),
        3.0,
    ) == 0.0


def test_non_positive_threshold_is_rejected() -> None:
    """A non-positive clipping half-width must fail closed."""
    rng = np.random.default_rng(
        31
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        truncated_gaussian_sigma(
            rng.normal(
                size=100
            ),
            -1.0,
        )
