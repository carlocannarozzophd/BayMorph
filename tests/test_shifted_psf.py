"""Tests for sub-pixel shifted PSF rendering."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.observation import render_shifted_psf
from baymorph.simulation.psf import EmpiricalPSF, GaussianPSF


def _centroid(image: np.ndarray) -> tuple[float, float]:
    """Return the flux-weighted image centroid."""
    y, x = np.indices(image.shape)
    total = float(image.sum())

    return (
        float(np.sum(x * image) / total),
        float(np.sum(y * image) / total),
    )


def test_shifted_psf_preserves_flux_away_from_edges() -> None:
    """Sub-pixel shifting must preserve flux before footprint clipping."""
    image = render_shifted_psf(
        psf=GaussianPSF(fwhm=2.5),
        flux=100.0,
        x0=20.35,
        y0=19.65,
        shape=(41, 43),
    )

    assert image.sum() == pytest.approx(
        100.0,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_shifted_psf_centroid_tracks_requested_position() -> None:
    """The rendered centroid must follow the requested sub-pixel centre."""
    expected_x = 20.35
    expected_y = 19.65

    image = render_shifted_psf(
        psf=GaussianPSF(fwhm=3.0),
        flux=100.0,
        x0=expected_x,
        y0=expected_y,
        shape=(41, 43),
    )

    measured_x, measured_y = _centroid(image)

    assert measured_x == pytest.approx(expected_x, abs=0.05)
    assert measured_y == pytest.approx(expected_y, abs=0.05)


def test_shifted_psf_loses_flux_at_footprint_edge() -> None:
    """Flux outside the output footprint must not be redistributed."""
    image = render_shifted_psf(
        psf=GaussianPSF(fwhm=3.0),
        flux=100.0,
        x0=-0.25,
        y0=10.0,
        shape=(21, 21),
    )

    assert 0.0 < image.sum() < 100.0


def test_shifted_psf_respects_global_origin() -> None:
    """Global coordinates must map correctly onto an extended domain."""
    image = render_shifted_psf(
        psf=GaussianPSF(fwhm=2.5),
        flux=50.0,
        x0=3.25,
        y0=4.75,
        shape=(31, 31),
        origin=(-10.0, -10.0),
    )

    measured_x, measured_y = _centroid(image)

    assert measured_x == pytest.approx(13.25, abs=0.05)
    assert measured_y == pytest.approx(14.75, abs=0.05)


def test_shifted_psf_supports_empirical_kernel() -> None:
    """Empirical PSFs must use the same shifted-rendering contract.

    The kernel must be a physically valid, adequately sampled band-limited
    PSF (not a single-occupied-pixel delta), otherwise cubic-spline
    sub-pixel shifting produces ringing large enough to legitimately exceed
    ``PSF_SHIFT_RINGING_TOLERANCE``.
    """
    kernel = GaussianPSF(fwhm=2.5).kernel(size=7)

    image = render_shifted_psf(
        psf=EmpiricalPSF(kernel),
        flux=25.0,
        x0=10.2,
        y0=9.8,
        shape=(21, 21),
    )

    assert image.shape == (21, 21)
    assert image.sum() == pytest.approx(
        25.0,
        rel=1.0e-12,
        abs=1.0e-12,
    )
    assert np.all(image >= 0.0)


def test_shifted_psf_rejects_excessive_ringing() -> None:
    """A pathologically undersampled kernel must fail closed, not be repaired."""
    kernel = np.zeros((7, 7), dtype=float)
    kernel[3, 3] = 1.0

    with pytest.raises(ValueError, match="ringing tolerance"):
        render_shifted_psf(
            psf=EmpiricalPSF(kernel),
            flux=25.0,
            x0=10.2,
            y0=9.8,
            shape=(21, 21),
        )


@pytest.mark.parametrize(
    ("flux", "x0", "y0"),
    [
        (0.0, 5.0, 5.0),
        (-1.0, 5.0, 5.0),
        (1.0, np.nan, 5.0),
        (1.0, 5.0, np.inf),
    ],
)
def test_shifted_psf_rejects_invalid_parameters(
    flux: float,
    x0: float,
    y0: float,
) -> None:
    """Invalid point-source parameters must fail explicitly."""
    with pytest.raises(ValueError):
        render_shifted_psf(
            psf=GaussianPSF(fwhm=2.0),
            flux=flux,
            x0=x0,
            y0=y0,
            shape=(11, 11),
        )
