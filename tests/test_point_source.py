"""Tests for unresolved point-source models."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.models import (
    DecomposableMorphology,
    MorphologicalComponent,
    PointSourceModel,
)
from baymorph.simulation.psf import (
    EmpiricalPSF,
    GaussianPSF,
)


def _centroid(image: np.ndarray) -> tuple[float, float]:
    """Return the flux-weighted image centroid."""
    y, x = np.indices(image.shape)
    total = float(image.sum())

    return (
        float(np.sum(x * image) / total),
        float(np.sum(y * image) / total),
    )


def test_point_source_satisfies_model_protocols() -> None:
    """Point sources must support total and named rendering."""
    model = PointSourceModel(
        flux=100.0,
        x0=10.0,
        y0=11.0,
        psf=GaussianPSF(fwhm=2.5),
    )

    assert isinstance(model, MorphologicalComponent)
    assert isinstance(model, DecomposableMorphology)


def test_point_source_preserves_flux_away_from_edges() -> None:
    """A fully contained point source must preserve its total flux."""
    model = PointSourceModel(
        flux=100.0,
        x0=20.35,
        y0=19.65,
        psf=GaussianPSF(fwhm=2.5),
    )

    image = model.render(
        shape=(41, 43),
    )

    assert image.sum() == pytest.approx(
        100.0,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_point_source_tracks_subpixel_position() -> None:
    """The observed centroid must follow the requested source centre."""
    model = PointSourceModel(
        flux=100.0,
        x0=20.35,
        y0=19.65,
        psf=GaussianPSF(fwhm=3.0),
    )

    image = model.render(
        shape=(41, 43),
    )

    measured_x, measured_y = _centroid(image)

    assert measured_x == pytest.approx(
        20.35,
        abs=0.05,
    )
    assert measured_y == pytest.approx(
        19.65,
        abs=0.05,
    )


def test_point_source_respects_render_origin() -> None:
    """Global coordinates must map correctly onto extended domains."""
    model = PointSourceModel(
        flux=50.0,
        x0=3.25,
        y0=4.75,
        psf=GaussianPSF(fwhm=2.5),
    )

    image = model.render(
        shape=(31, 31),
        origin=(-10.0, -10.0),
    )

    measured_x, measured_y = _centroid(image)

    assert measured_x == pytest.approx(
        13.25,
        abs=0.05,
    )
    assert measured_y == pytest.approx(
        14.75,
        abs=0.05,
    )


def test_point_source_loses_flux_at_footprint_edge() -> None:
    """Flux outside the requested footprint must not be redistributed."""
    model = PointSourceModel(
        flux=100.0,
        x0=-0.25,
        y0=10.0,
        psf=GaussianPSF(fwhm=3.0),
    )

    image = model.render(
        shape=(21, 21),
    )

    assert 0.0 < image.sum() < 100.0


def test_point_source_supports_empirical_psf() -> None:
    """Empirical kernels must use the same point-source interface."""
    kernel = np.zeros((7, 7), dtype=float)
    kernel[3, 3] = 1.0

    model = PointSourceModel(
        flux=25.0,
        x0=10.2,
        y0=9.8,
        psf=EmpiricalPSF(kernel),
    )

    image = model.render(
        shape=(21, 21),
    )
    components = model.render_components(
        shape=(21, 21),
    )

    assert set(components) == {
        "point_source",
    }
    assert np.array_equal(
        components["point_source"],
        image,
    )
    assert image.sum() == pytest.approx(
        25.0,
        rel=1.0e-12,
        abs=1.0e-12,
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
def test_point_source_rejects_invalid_parameters(
    flux: float,
    x0: float,
    y0: float,
) -> None:
    """Invalid physical parameters must fail explicitly."""
    with pytest.raises(ValueError):
        PointSourceModel(
            flux=flux,
            x0=x0,
            y0=y0,
            psf=GaussianPSF(fwhm=2.0),
        )


def test_point_source_rejects_invalid_psf() -> None:
    """A point source requires an explicit observation PSF."""
    with pytest.raises(
        TypeError,
        match=r"exposing kernel",
    ):
        PointSourceModel(
            flux=1.0,
            x0=5.0,
            y0=5.0,
            psf=object(),
        )
