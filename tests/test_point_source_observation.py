"""Tests for observation-level point-source rendering."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.models import PointSourceModel
from baymorph.observation import PSFPointSourceRenderer
from baymorph.simulation.psf import EmpiricalPSF, GaussianPSF


def _centroid(image: np.ndarray) -> tuple[float, float]:
    """Return the flux-weighted image centroid."""
    y, x = np.indices(image.shape)
    total = float(image.sum())

    return (
        float(np.sum(x * image) / total),
        float(np.sum(y * image) / total),
    )


def test_renderer_preserves_contained_point_source_flux() -> None:
    """A fully contained observed point source must preserve flux."""
    source = PointSourceModel(
        flux=100.0,
        x0=20.35,
        y0=19.65,
    )
    renderer = PSFPointSourceRenderer(
        GaussianPSF(fwhm=2.5),
    )

    image = renderer.render(
        source,
        shape=(41, 43),
    )

    assert image.sum() == pytest.approx(
        source.flux,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_renderer_tracks_subpixel_position() -> None:
    """The observed centroid must follow the intrinsic source position."""
    source = PointSourceModel(
        flux=100.0,
        x0=20.35,
        y0=19.65,
    )
    renderer = PSFPointSourceRenderer(
        GaussianPSF(fwhm=3.0),
    )

    measured_x, measured_y = _centroid(
        renderer.render(
            source,
            shape=(41, 43),
        )
    )

    assert measured_x == pytest.approx(source.x0, abs=0.05)
    assert measured_y == pytest.approx(source.y0, abs=0.05)


def test_renderer_respects_extended_domain_origin() -> None:
    """Global source coordinates must respect the rendering origin."""
    source = PointSourceModel(
        flux=50.0,
        x0=3.25,
        y0=4.75,
    )
    renderer = PSFPointSourceRenderer(
        GaussianPSF(fwhm=2.5),
    )

    image = renderer.render(
        source,
        shape=(31, 31),
        origin=(-10.0, -10.0),
    )

    measured_x, measured_y = _centroid(image)

    assert measured_x == pytest.approx(13.25, abs=0.05)
    assert measured_y == pytest.approx(14.75, abs=0.05)


def test_renderer_supports_empirical_psf() -> None:
    """Empirical PSFs must use the same observation-level renderer.

    The kernel must be a physically valid, adequately sampled band-limited
    PSF (not a single-occupied-pixel delta), otherwise cubic-spline
    sub-pixel shifting produces ringing large enough to legitimately exceed
    the declared ringing tolerance.
    """
    kernel = GaussianPSF(fwhm=2.5).kernel(size=7)

    source = PointSourceModel(
        flux=25.0,
        x0=10.2,
        y0=9.8,
    )
    renderer = PSFPointSourceRenderer(
        EmpiricalPSF(kernel),
    )

    image = renderer.render(
        source,
        shape=(21, 21),
    )

    assert image.sum() == pytest.approx(
        source.flux,
        rel=1.0e-12,
        abs=1.0e-12,
    )
    assert np.all(image >= 0.0)


def test_renderer_rejects_invalid_source_object() -> None:
    """Only intrinsic point-source models may be rendered."""
    renderer = PSFPointSourceRenderer(
        GaussianPSF(fwhm=2.0),
    )

    with pytest.raises(
        TypeError,
        match="PointSourceModel",
    ):
        renderer.render(
            object(),
            shape=(11, 11),
        )
