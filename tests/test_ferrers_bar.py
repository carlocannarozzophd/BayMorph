"""The Ferrers bar (AD-048)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from baymorph.models import FerrersBarComponent
from baymorph.models.isophote import area_factor

PARAMETERS = dict(
    flux=1000.0,
    a=8.0,
    alpha=1.5,
    c0=0.0,
    q=0.3,
    theta=0.55,
    x0=40.31,
    y0=39.77,
)


def _bar(**changes):
    parameters = dict(PARAMETERS)
    parameters.update(changes)

    return FerrersBarComponent(**parameters)


def _major_axis(bar, distances):
    distances = np.asarray(distances, dtype=float)

    return (
        bar.x0 + distances * np.cos(bar.theta),
        bar.y0 + distances * np.sin(bar.theta),
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("flux", 0.0),
        ("a", -1.0),
        ("alpha", 0.0),
        ("alpha", np.nan),
        ("c0", -1.5),
        ("q", 0.0),
        ("q", 1.2),
        ("theta", np.nan),
        ("x0", np.inf),
    ],
)
def test_invalid_parameters_are_rejected(name, value):
    with pytest.raises(ValueError):
        _bar(**{name: value})


def test_the_light_ends_at_the_semi_major_length():
    bar = _bar()
    values = bar.intensity(*_major_axis(bar, [0.0, 4.0, 7.9, 8.0, 8.1, 20.0]))

    assert values[0] == pytest.approx(np.exp(bar.log_central_intensity))
    assert values[2] > 0.0
    np.testing.assert_array_equal(values[3:], 0.0)


@pytest.mark.parametrize(("alpha", "c0"), [(0.5, 0.0), (2.0, 2.0), (4.0, 3.5)])
def test_the_central_intensity_gives_the_flux(alpha, c0):
    """The flux is the radial integral over the generalized ellipses."""
    bar = _bar(alpha=alpha, c0=c0)
    central = np.exp(bar.log_central_intensity)
    flux, _ = quad(
        lambda m: 2.0 * area_factor(c0) * bar.q * m * central
        * (1.0 - (m / bar.a) ** 2) ** alpha,
        0.0,
        bar.a,
        epsabs=0.0,
        epsrel=1.0e-12,
    )

    assert flux == pytest.approx(bar.flux, rel=1.0e-10)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 3.0])
def test_alpha_is_the_slope_of_the_edge(alpha):
    """Near the edge the profile falls as the distance to it raised to alpha."""
    bar = _bar(alpha=alpha)
    values = bar.intensity(*_major_axis(bar, bar.a - np.array([1.0e-4, 2.0e-4])))
    slope = np.log(values[1] / values[0]) / np.log(2.0)

    assert slope == pytest.approx(alpha, rel=1.0e-3)


@pytest.mark.parametrize(
    ("alpha", "c0", "q"),
    [
        (0.5, 0.0, 0.15),
        (0.5, 2.5, 0.25),
        (1.0, 3.5, 0.15),
        (2.0, 0.0, 0.12),
        (5.0, 1.5, 0.4),
    ],
)
def test_the_total_flux_is_the_flux_parameter(alpha, c0, q):
    """A bar inside the grid holds its analytic flux, edge included."""
    bar = _bar(a=7.0, alpha=alpha, c0=c0, q=q)
    image = bar.render((81, 81))

    assert image.sum() == pytest.approx(bar.flux, rel=1.0e-7)


def test_cells_the_edge_crosses_are_split(monkeypatch):
    """Without the declared edge, adaptive integration loses part of the bar.

    A sliver of the bar inside a cell, or of empty plane beyond its edge, can
    escape every node at two successive levels.
    """
    bar = _bar(a=6.0, alpha=0.5, q=0.12)
    split = bar.render((81, 81))
    monkeypatch.setattr(
        FerrersBarComponent,
        "edge_cells",
        lambda self, x, y, half: None,
    )
    unsplit = bar.render((81, 81))

    assert abs(split.sum() / bar.flux - 1.0) <= 1.0e-7
    assert abs(unsplit.sum() / bar.flux - 1.0) > 1.0e-5


def test_the_hybrid_rule_agrees_with_adaptive_integration():
    bar = _bar(alpha=0.5, c0=2.0)
    hybrid = bar.render((81, 81))
    adaptive = bar.render((81, 81), renderer="adaptive", rtol=1.0e-10)

    assert np.max(np.abs(hybrid - adaptive)) <= 1.0e-8 * adaptive.max()


def test_the_edge_and_the_axes_inside_the_bar_are_declared():
    """A wide bar, whose centre lies more than two cells from its edge.

    With c0 = 1.5 the isophotes are not smooth on their axes; with c0 = 2,
    an even exponent, they would be.
    """
    y, x = np.mgrid[0:81, 0:81].astype(float)
    boxy = _bar(q=0.7, c0=1.5)
    elliptical = _bar(q=0.7, c0=0.0)
    columns, rows = (
        np.rint(values).astype(int)
        for values in _major_axis(boxy, [3.0, 8.0, 30.0])
    )

    # Inside the bar, on the major axis; on the edge; far beyond it.
    np.testing.assert_array_equal(
        boxy.nonsmooth_cells(x, y)[rows, columns],
        [True, True, False],
    )
    np.testing.assert_array_equal(
        elliptical.nonsmooth_cells(x, y)[rows, columns],
        [False, True, False],
    )


def test_edge_cells_stop_at_a_fraction_of_the_bar():
    bar = _bar()
    x, y = _major_axis(bar, [bar.a, 0.0])

    np.testing.assert_array_equal(bar.edge_cells(x, y, 0.25), [True, False])
    assert bar.edge_cells(x, y, 0.5 * bar.a / 1024.0) is None


def test_scaled_bar_keeps_the_flux_of_each_cell():
    """Cells of the finer grid sum to the cells of the native grid."""
    bar = _bar(alpha=1.0, c0=1.5)
    native = bar.render((41, 41), origin=(20.0, 20.0))
    factor = 5
    fine = bar.scaled(factor).render(
        (41 * factor, 41 * factor),
        origin=(factor * 20.0, factor * 20.0),
    )
    blocks = fine.reshape(41, factor, 41, factor).sum(axis=(1, 3))

    assert bar.scaled(factor).a == pytest.approx(40.0)
    assert np.max(np.abs(blocks - native)) <= 1.0e-6 * native.max()
