"""Tests for PSF-aware radial-profile binning."""

from __future__ import annotations

import numpy as np
import pytest

import baymorph.pipeline.radial_profile as radial_pipeline
from baymorph.pipeline.radial_profile import (
    _default_radial_dr,
    _psf_fwhm_pixels,
    _radial_edges,
)
from baymorph.simulation.psf import GaussianPSF


def test_default_radial_dr_uses_half_gaussian_psf_fwhm():
    """Gaussian diagnostics should sample at half the configured FWHM."""
    psf = GaussianPSF(
        fwhm=4.0
    )

    assert _psf_fwhm_pixels(
        psf
    ) == pytest.approx(
        4.0
    )

    assert _default_radial_dr(
        psf
    ) == pytest.approx(
        2.0
    )


def test_empirical_psf_uses_core_fwhm(
    monkeypatch,
):
    """Empirical PSFs should use the measured core FWHM."""
    monkeypatch.setattr(
        radial_pipeline,
        "estimate_psf_core_fwhm",
        lambda kernel: 6.0,
    )

    psf = np.ones(
        (9, 9),
        dtype=float,
    )

    assert _psf_fwhm_pixels(
        psf
    ) == pytest.approx(
        6.0
    )

    assert _default_radial_dr(
        psf
    ) == pytest.approx(
        3.0
    )


def test_default_radial_dr_has_one_pixel_floor():
    """PSF-aware binning should not default to sub-pixel annuli."""
    psf = GaussianPSF(
        fwhm=1.2
    )

    assert _default_radial_dr(
        psf
    ) == pytest.approx(
        1.0
    )


def test_default_radial_dr_falls_back_without_psf():
    """Missing PSF resolution should retain a one-pixel fallback."""
    assert _default_radial_dr(
        None
    ) == pytest.approx(
        1.0
    )


def test_radial_edges_use_psf_aware_default():
    """Absent explicit dr, radial edges should use half the PSF FWHM."""
    radii = _radial_edges(
        (100, 100),
        {},
        axis_ratio=0.8,
        psf=GaussianPSF(
            fwhm=4.0
        ),
    )

    assert np.diff(
        radii
    )[:-1] == pytest.approx(
        2.0
    )


def test_explicit_radial_dr_overrides_psf_default():
    """A user-configured dr must always override the automatic default."""
    radii = _radial_edges(
        (100, 100),
        {
            "binning": "fixed",
            "dr": 5.0,
        },
        axis_ratio=0.8,
        psf=GaussianPSF(
            fwhm=4.0
        ),
    )

    assert np.diff(
        radii
    )[:-1] == pytest.approx(
        5.0
    )
