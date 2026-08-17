"""Tests for explicit scientific priors in FITS configuration generation."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from baymorph.pipeline.config import resolve_config
from baymorph.pipeline.config_maker import make_config_from_fits


def _write_test_fits(
    path,
) -> None:
    """Write one minimal observed-image fixture."""
    fits.writeto(
        path,
        np.zeros(
            (
                21,
                21,
            ),
            dtype=float,
        ),
    )


def test_config_maker_preserves_explicit_typed_model_prior(
    tmp_path,
) -> None:
    """Observed-data prior overrides must be copied without data scaling."""
    path = tmp_path / "galaxy.fits"

    _write_test_fits(
        path
    )

    flux_prior = {
        "type": "uniform",
        "lower": 100.0,
        "upper": 1000.0,
    }

    config = make_config_from_fits(
        path,
        target_position=(
            10.0,
            10.0,
        ),
        model_priors={
            "flux": flux_prior,
        },
    )

    assert config[
        "fit"
    ][
        "priors"
    ][
        "single_sersic"
    ][
        "flux"
    ] == flux_prior

    resolved = resolve_config(
        config
    )

    assert resolved[
        "fit"
    ][
        "priors"
    ][
        "single_sersic"
    ][
        "flux"
    ] == flux_prior


def test_config_maker_rejects_legacy_data_scaled_prior(
    tmp_path,
) -> None:
    """Observed config generation must reject min-factor flux priors."""
    path = tmp_path / "galaxy.fits"

    _write_test_fits(
        path
    )

    with pytest.raises(
        ValueError,
        match="Prior type must be one of",
    ):
        make_config_from_fits(
            path,
            target_position=(
                10.0,
                10.0,
            ),
            model_priors={
                "flux": {
                    "min_factor": 0.0,
                    "max_factor": 2.0,
                },
            },
        )


def test_config_maker_does_not_modify_explicit_prior(
    tmp_path,
) -> None:
    """FITS image values must not modify an explicitly supplied prior."""
    low_path = tmp_path / "low.fits"
    high_path = tmp_path / "high.fits"

    fits.writeto(
        low_path,
        np.zeros(
            (
                21,
                21,
            ),
            dtype=float,
        ),
    )

    fits.writeto(
        high_path,
        np.full(
            (
                21,
                21,
            ),
            1.0e12,
            dtype=float,
        ),
    )

    prior = {
        "type": "uniform",
        "lower": 50.0,
        "upper": 5000.0,
    }

    low = make_config_from_fits(
        low_path,
        target_position=(
            10.0,
            10.0,
        ),
        model_priors={
            "flux": prior,
        },
    )

    high = make_config_from_fits(
        high_path,
        target_position=(
            10.0,
            10.0,
        ),
        model_priors={
            "flux": prior,
        },
    )

    assert low[
        "fit"
    ][
        "priors"
    ][
        "single_sersic"
    ][
        "flux"
    ] == high[
        "fit"
    ][
        "priors"
    ][
        "single_sersic"
    ][
        "flux"
    ]


def test_config_maker_requires_non_default_model_priors(
    tmp_path,
) -> None:
    """Missing scientific priors must fail during observed config creation."""
    path = tmp_path / "galaxy.fits"

    _write_test_fits(
        path
    )

    with pytest.raises(
        ValueError,
        match=r"Missing prior configuration.*flux",
    ):
        make_config_from_fits(
            path,
            target_position=(
                10.0,
                10.0,
            ),
        )
