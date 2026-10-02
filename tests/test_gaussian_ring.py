"""The Gaussian ring (AD-048)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from baymorph.models import GaussianRingComponent

PARAMETERS = dict(
    flux=1000.0,
    r=6.0,
    sigma=1.5,
    q=0.6,
    theta=0.55,
    x0=40.31,
    y0=39.77,
)


def _ring(**changes):
    parameters = dict(PARAMETERS)
    parameters.update(changes)

    return GaussianRingComponent(**parameters)


def _major_axis(ring, distances):
    distances = np.asarray(distances, dtype=float)

    return (
        ring.x0 + distances * np.cos(ring.theta),
        ring.y0 + distances * np.sin(ring.theta),
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("flux", 0.0),
        ("r", 0.0),
        ("r", -2.0),
        ("sigma", 0.0),
        ("q", 0.0),
        ("q", 1.5),
        ("theta", np.nan),
        ("y0", np.inf),
    ],
)
def test_invalid_parameters_are_rejected(name, value):
    with pytest.raises(ValueError):
        _ring(**{name: value})


def test_the_crest_lies_at_r_on_the_major_axis():
    ring = _ring()
    values = ring.intensity(*_major_axis(ring, ring.r + np.array([-0.1, 0.0, 0.1])))

    assert values[1] == pytest.approx(np.exp(ring.log_crest_intensity))
    assert values[1] > values[0]
    assert values[1] > values[2]


@pytest.mark.parametrize(("r", "sigma"), [(6.0, 1.5), (1.0, 3.0), (40.0, 0.5)])
def test_the_crest_intensity_gives_the_flux(r, sigma):
    """The flux is the radial integral over the ellipses, centre included."""
    ring = _ring(r=r, sigma=sigma)
    crest = np.exp(ring.log_crest_intensity)
    flux, _ = quad(
        lambda m: 2.0 * np.pi * ring.q * m * crest
        * np.exp(-0.5 * ((m - r) / sigma) ** 2),
        0.0,
        r + 40.0 * sigma,
        points=[r],
        epsabs=0.0,
        epsrel=1.0e-12,
    )

    assert flux == pytest.approx(ring.flux, rel=1.0e-10)


def test_a_wide_ring_holds_its_circumference_times_its_profile():
    ring = _ring(r=200.0, sigma=1.0)
    crest = np.exp(ring.log_crest_intensity)
    circumference = 2.0 * np.pi * ring.r * ring.q
    profile = np.sqrt(2.0 * np.pi) * ring.sigma * crest

    assert circumference * profile == pytest.approx(ring.flux, rel=1.0e-12)


@pytest.mark.parametrize(
    ("r", "sigma", "q"),
    [(4.0, 0.7, 0.45), (6.0, 1.5, 0.8), (2.0, 2.5, 0.6), (9.0, 0.5, 1.0)],
)
def test_the_total_flux_is_the_flux_parameter(r, sigma, q):
    ring = _ring(r=r, sigma=sigma, q=q)
    image = ring.render((81, 81))

    assert image.sum() == pytest.approx(ring.flux, rel=1.0e-8)


def test_a_narrow_ring_is_integrated_adaptively_about_its_crest(monkeypatch):
    """Without its band, a ring narrower than a cell is integrated poorly.

    The ring lies far from the adaptive block about the centre, which would
    otherwise widen over it.
    """
    ring = _ring(r=25.0, sigma=0.8, q=0.45)
    adaptive = ring.render((81, 81), renderer="adaptive", rtol=1.0e-10)
    banded = ring.render((81, 81))
    monkeypatch.setattr(
        GaussianRingComponent,
        "nonsmooth_cells",
        lambda self, x, y: None,
    )
    unbanded = ring.render((81, 81))

    assert np.max(np.abs(banded - adaptive)) <= 1.0e-8 * adaptive.max()
    assert np.max(np.abs(unbanded - adaptive)) > 1.0e-6 * adaptive.max()


def test_the_band_is_declared_only_for_rings_narrower_than_a_cell():
    y, x = np.mgrid[0:81, 0:81].astype(float)
    narrow = _ring(sigma=1.5, q=0.6)
    columns, rows = (
        np.rint(values).astype(int)
        for values in _major_axis(narrow, [narrow.r, 30.0])
    )

    assert _ring(sigma=2.0, q=0.5).nonsmooth_cells(x, y) is None
    np.testing.assert_array_equal(
        narrow.nonsmooth_cells(x, y)[rows, columns],
        [True, False],
    )


def test_scaled_ring_keeps_the_flux_of_each_cell():
    """Cells of the finer grid sum to the cells of the native grid."""
    ring = _ring(sigma=0.8, q=0.45)
    native = ring.render((41, 41), origin=(20.0, 20.0))
    factor = 5
    fine = ring.scaled(factor).render(
        (41 * factor, 41 * factor),
        origin=(factor * 20.0, factor * 20.0),
    )
    blocks = fine.reshape(41, factor, 41, factor).sum(axis=(1, 3))

    assert (ring.scaled(factor).r, ring.scaled(factor).sigma) == pytest.approx(
        (30.0, 4.0)
    )
    assert np.max(np.abs(blocks - native)) <= 1.0e-6 * native.max()
