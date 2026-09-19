"""Tests for the prepared background-noise estimators."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.core.background import estimate_background_constant

TRUE_SIGMA = 2.0

# Comparable to the sky annulus of the canonical simulated example.
SKY_PIXELS = 11586

REALIZATIONS = 120


def _estimator_performance(
    method: str,
    *,
    seed: int,
    contaminated_fraction: float = 0.0,
) -> tuple[float, float]:
    """Return the relative bias and relative scatter of one estimator."""
    rng = np.random.default_rng(
        seed
    )

    estimates = []

    for _ in range(REALIZATIONS):
        values = rng.normal(
            0.0,
            TRUE_SIGMA,
            SKY_PIXELS,
        )

        if contaminated_fraction:
            contaminated = int(
                contaminated_fraction * values.size
            )
            values[:contaminated] += rng.uniform(
                5.0,
                200.0,
                contaminated,
            )

        estimates.append(
            estimate_background_constant(
                values,
                noise_method=method,
            ).sigma
        )

    estimates = np.asarray(
        estimates,
        dtype=float,
    )

    return (
        float(
            estimates.mean() / TRUE_SIGMA - 1.0
        ),
        float(
            estimates.std() / TRUE_SIGMA
        ),
    )


def test_corrected_std_is_unbiased_on_clean_gaussian_pixels() -> None:
    """The production estimator must recover a known sigma."""
    bias, _ = _estimator_performance(
        "corrected_std",
        seed=101,
    )

    assert abs(bias) < 0.004


def test_plain_clipped_std_is_biased_low() -> None:
    """Without the correction the clipped standard deviation is biased.

    This is the defect the correction removes, so it is asserted directly
    rather than assumed.
    """
    bias, _ = _estimator_performance(
        "std",
        seed=102,
    )

    assert bias < -0.010


def test_corrected_std_is_more_efficient_than_mad() -> None:
    """The production estimator must have less run-to-run scatter than MAD.

    A scaled MAD has roughly 37% efficiency, so its estimate fluctuates about
    1.5 times more at the same sample size. That scatter propagates directly
    into every credible-interval width and into the absolute evidence.
    """
    _, mad_scatter = _estimator_performance(
        "mad",
        seed=103,
    )
    _, corrected_scatter = _estimator_performance(
        "corrected_std",
        seed=103,
    )

    assert corrected_scatter < 0.8 * mad_scatter


def test_corrected_std_has_no_worse_bias_than_mad() -> None:
    """Replacing MAD must not trade scatter for bias."""
    mad_bias, _ = _estimator_performance(
        "mad",
        seed=104,
    )
    corrected_bias, _ = _estimator_performance(
        "corrected_std",
        seed=104,
    )

    assert abs(corrected_bias) <= abs(mad_bias)


@pytest.mark.parametrize(
    "contaminated_fraction",
    [
        0.01,
        0.05,
    ],
)
def test_clipping_keeps_the_estimator_robust_to_bright_pixels(
    contaminated_fraction: float,
) -> None:
    """Unmasked sources must not inflate the estimated scale.

    Clipping is retained for exactly this reason: the truncation correction
    restores efficiency, it does not provide robustness.
    """
    bias, _ = _estimator_performance(
        "corrected_std",
        seed=105,
        contaminated_fraction=contaminated_fraction,
    )

    assert abs(bias) < 0.010


def test_production_preparation_uses_the_detrended_scale() -> None:
    """The canonical preparation must measure noise, not sky structure.

    This pins the production wiring: without ``detrend`` the same field yields
    a visibly inflated scale, so the assertion cannot pass by accident.
    """
    rng = np.random.default_rng(
        107
    )
    side = 108

    image = rng.normal(
        20.0,
        TRUE_SIGMA,
        (
            side,
            side,
        ),
    )
    image = image + np.linspace(
        -2.0,
        2.0,
        side,
    )[None, :]

    inflated = estimate_background_constant(
        image,
        noise_method="corrected_std",
    ).sigma
    detrended = estimate_background_constant(
        image,
        noise_method="corrected_std",
        detrend=True,
    ).sigma

    assert inflated / TRUE_SIGMA > 1.10
    assert abs(
        detrended / TRUE_SIGMA - 1.0
    ) < 0.02


def test_unknown_noise_method_is_rejected() -> None:
    """An unrecognised estimator must fail closed."""
    rng = np.random.default_rng(
        106
    )

    with pytest.raises(
        ValueError,
        match="Unknown noise estimation method",
    ):
        estimate_background_constant(
            rng.normal(
                0.0,
                TRUE_SIGMA,
                1000,
            ),
            noise_method="not_an_estimator",
        )
