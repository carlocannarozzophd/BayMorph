"""Tests for the PSF displayed by residual-map diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

import baymorph.pipeline.residual_maps as residual_maps


def test_gaussian_psf_display_does_not_load_empirical_psf(
    monkeypatch,
) -> None:
    """Analytic PSF runs must not require empirical-PSF provenance."""

    def fail_if_called(config):
        raise AssertionError("load_empirical_psf must not be called.")

    monkeypatch.setattr(
        residual_maps,
        "load_empirical_psf",
        fail_if_called,
    )

    config = {
        "psf": {
            "type": "gaussian",
            "gaussian": {
                "fwhm": 2.0,
            },
        },
    }

    assert residual_maps._load_display_psf(config) is None

    kernel = residual_maps._psf_to_image(
        None,
        config,
    )

    assert np.all(np.isfinite(kernel))
    assert np.isclose(kernel.sum(), 1.0)


def test_empirical_psf_display_propagates_invalid_provenance(
    monkeypatch,
) -> None:
    """Invalid empirical-PSF provenance must still fail closed."""

    def reject_metadata(config):
        raise ValueError(
            "Empirical PSF metadata schema/type is unsupported or missing."
        )

    monkeypatch.setattr(
        residual_maps,
        "load_empirical_psf",
        reject_metadata,
    )

    with pytest.raises(
        ValueError,
        match="schema/type",
    ):
        residual_maps._load_display_psf(
            {
                "psf": {
                    "type": "empirical",
                },
            }
        )


def test_missing_empirical_psf_is_not_displayed(
    monkeypatch,
) -> None:
    """A missing empirical PSF leaves the diagnostic PSF panel empty."""

    def missing_psf(config):
        raise FileNotFoundError("psf.fits")

    monkeypatch.setattr(
        residual_maps,
        "load_empirical_psf",
        missing_psf,
    )

    assert residual_maps._load_display_psf({}) is None
