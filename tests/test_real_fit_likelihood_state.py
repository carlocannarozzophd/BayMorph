"""Tests for real-fit likelihood-state preparation."""

from types import SimpleNamespace

import numpy as np
import pytest

from baymorph.pipeline.likelihood_state import (
    prepare_real_fit_likelihood_state,
)


def _inference_region():
    """Return a minimal deterministic inference-region stand-in."""
    return SimpleNamespace(
        noise=np.full(
            (3, 4),
            2.0,
        ),
        shape=(3, 4),
        bounds={
            "xmin": 2,
            "xmax": 6,
            "ymin": 1,
            "ymax": 4,
        },
    )


def test_independent_gaussian_reuses_inference_noise():
    """Independent Gaussian state must preserve the inference noise map."""
    config = {
        "fit": {
            "likelihood": {
                "type": "gaussian",
            },
        },
    }

    region = _inference_region()

    state = (
        prepare_real_fit_likelihood_state(
            config,
            region,
        )
    )

    assert (
        state.likelihood_type
        == "gaussian"
    )

    np.testing.assert_array_equal(
        state.noise,
        region.noise,
    )

    assert (
        state.correlation_kernel
        is None
    )

    assert (
        state.metadata[
            "normalization_complete"
        ]
        is True
    )


def test_correlated_gaussian_crops_persisted_scale(
    monkeypatch,
):
    """Correlated state must use exactly the persisted inference bounds."""
    marginal_sigma = np.arange(
        42,
        dtype=float,
    ).reshape(
        6,
        7,
    ) + 1.0

    kernel = np.array(
        [
            [0.0, 0.1, 0.0],
            [0.2, 1.0, 0.2],
            [0.0, 0.1, 0.0],
        ]
    )

    calibration = SimpleNamespace(
        marginal_sigma=(
            marginal_sigma
        ),
        correlation_kernel=kernel,
        metadata={
            "schema_version": 1,
            "factorization": "C = S R S",
            "correlation_model": (
                "empirical_stationary_2d"
            ),
            "marginal_model": (
                "quadratic_log_sigma"
            ),
            "max_lag": 1,
        },
    )

    monkeypatch.setattr(
        "baymorph.pipeline.likelihood_state."
        "load_correlated_noise_calibration",
        lambda config: calibration,
    )

    monkeypatch.setattr(
        "baymorph.pipeline.likelihood_state."
        "correlated_noise_calibration_path",
        lambda config: (
            "correlated_gaussian_noise.npz"
        ),
    )

    config = {
        "fit": {
            "likelihood": {
                "type": (
                    "correlated_gaussian"
                ),
                "include_normalization": False,
            },
        },
    }

    region = _inference_region()

    state = (
        prepare_real_fit_likelihood_state(
            config,
            region,
        )
    )

    expected = marginal_sigma[
        1:4,
        2:6,
    ]

    np.testing.assert_array_equal(
        state.noise,
        expected,
    )

    np.testing.assert_array_equal(
        state.correlation_kernel,
        kernel,
    )

    assert (
        state.metadata[
            "fit_statistic"
        ]
        == "mahalanobis_chi2"
    )

    assert (
        state.metadata[
            "residual_diagnostic"
        ]
        == "marginal_standardized"
    )

    assert (
        state.metadata[
            "normalization_complete"
        ]
        is False
    )


def test_correlated_gaussian_requires_persisted_calibration(
    monkeypatch,
):
    """Correlated fitting must never silently fall back to diagonal noise."""
    def raise_missing(
        config,
    ):
        del config

        raise FileNotFoundError(
            "missing"
        )

    monkeypatch.setattr(
        "baymorph.pipeline.likelihood_state."
        "load_correlated_noise_calibration",
        raise_missing,
    )

    config = {
        "fit": {
            "likelihood": {
                "type": (
                    "correlated_gaussian"
                ),
                "include_normalization": False,
            },
        },
    }

    with pytest.raises(
        FileNotFoundError,
        match="requires a persisted",
    ):
        prepare_real_fit_likelihood_state(
            config,
            _inference_region(),
        )


def test_correlated_gaussian_rejects_incomplete_absolute_normalization():
    """Absolute normalization must not be claimed without log|R_V|."""
    config = {
        "fit": {
            "likelihood": {
                "type": (
                    "correlated_gaussian"
                ),
                "include_normalization": True,
            },
        },
    }

    with pytest.raises(
        ValueError,
        match="correlation_logdet",
    ):
        prepare_real_fit_likelihood_state(
            config,
            _inference_region(),
        )
