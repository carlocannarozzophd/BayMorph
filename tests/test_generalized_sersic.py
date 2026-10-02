"""The Sérsic profile on generalized ellipses (AD-048)."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.models.generalized_sersic import (
    GeneralizedSersicComponent,
    axis_strips,
)
from baymorph.models.isophote import generalized_radius, major_minor_offsets
from baymorph.models.sersic import SersicModel

PARAMETERS = dict(
    flux=1000.0, re=6.0, n=2.5, q=0.7, theta=0.4, x0=40.27, y0=39.81
)


def _component(**changes):
    parameters = dict(PARAMETERS, c0=0.0)
    parameters.update(changes)

    return GeneralizedSersicComponent(**parameters)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("flux", 0.0),
        ("re", -1.0),
        ("n", 0.0),
        ("q", 1.5),
        ("q", 0.0),
        ("c0", -1.2),
        ("c0", np.nan),
        ("theta", np.inf),
        ("x0", np.nan),
    ],
)
def test_invalid_parameters_are_rejected(name, value):
    with pytest.raises(ValueError):
        _component(**{name: value})


def test_ellipses_give_the_sersic_profile():
    """With c0 = 0 the component is the Sérsic profile of the package."""
    component = _component()
    sersic = SersicModel(**PARAMETERS)
    y, x = np.mgrid[30:51, 30:51].astype(float)

    np.testing.assert_allclose(
        component.intensity(x, y),
        sersic.surface_brightness(sersic._elliptical_radius(x, y)),
        rtol=1.0e-13,
    )

    image = component.render((81, 81))
    reference = sersic.render((81, 81))

    assert np.max(np.abs(image - reference)) <= 1.0e-6 * reference.max()


@pytest.mark.parametrize("c0", [-0.5, 0.5, 1.0, 2.0])
def test_the_total_flux_is_the_flux_parameter(c0):
    """A compact profile on a large grid holds all of its flux."""
    component = _component(re=2.0, n=1.0, c0=c0)
    image = component.render((81, 81))

    assert image.sum() == pytest.approx(component.flux, rel=1.0e-8)


def test_rhombic_isophotes_are_accurate_to_a_few_millionths():
    """At c0 = -1 the radius has kinks along the axes, oblique to the cells.

    The error estimate of a cell crossed by an oblique kink can agree
    between two levels while both are off, so the cell rules converge to
    within a few millionths of the flux instead of a billionth. A fine
    midpoint rule, which a kink cannot mislead, measures the error at
    1.1e-6 for this profile.
    """
    component = _component(re=2.0, n=1.0, c0=-1.0)
    image = component.render((81, 81))

    assert image.sum() == pytest.approx(component.flux, rel=3.0e-6)


@pytest.mark.parametrize("c0", [-0.6, 0.0, 0.8])
def test_re_is_the_half_light_isophote(c0):
    """Half of the light lies within the generalized isophote m = re."""
    component = _component(c0=c0, n=1.5)
    step = 0.05
    axis = np.arange(0.0, 81.0, step) - 0.5 + step / 2
    total = 0.0
    inside = 0.0

    for row in np.array_split(axis, 40):
        y = row[:, None]
        x = axis[None, :]
        values = component.intensity(x, y) * step**2
        major, minor = major_minor_offsets(
            x, y, x0=component.x0, y0=component.y0, theta=component.theta
        )
        radius = generalized_radius(major, minor, q=component.q, c0=c0)
        total += values.sum()
        inside += values[radius <= component.re].sum()

    assert inside / total == pytest.approx(0.5, abs=2.0e-3)


def test_boxy_and_discy_isophotes_differ_in_the_expected_sense():
    """Relative to the major axis, boxy isophotes are brighter on the diagonal."""
    distance = 6.0
    x0, y0, theta = PARAMETERS["x0"], PARAMETERS["y0"], PARAMETERS["theta"]

    def contrast(c0):
        component = _component(c0=c0, q=1.0)
        diagonal = component.intensity(
            np.array([x0 + distance * np.cos(theta + np.pi / 4)]),
            np.array([y0 + distance * np.sin(theta + np.pi / 4)]),
        )
        major = component.intensity(
            np.array([x0 + distance * np.cos(theta)]),
            np.array([y0 + distance * np.sin(theta)]),
        )

        return float(diagonal[0] / major[0])

    assert contrast(0.5) > contrast(0.0) > contrast(-0.5)
    assert contrast(0.0) == pytest.approx(1.0, rel=1.0e-12)


def test_axis_strips_are_declared_only_where_needed():
    major = np.array([[0.2, 3.0], [5.0, 4.0]])
    minor = np.array([[3.0, 0.1], [5.0, 4.0]])

    assert axis_strips(major, minor, 0.0) is None
    assert axis_strips(major, minor, 2.0) is None
    np.testing.assert_array_equal(
        axis_strips(major, minor, 0.5),
        [[True, True], [False, False]],
    )


def test_scaled_component_keeps_the_flux_of_each_cell():
    """Cells of the finer grid sum to the cells of the native grid."""
    component = _component(c0=0.5)
    native = component.render((41, 41), origin=(20.0, 20.0))
    factor = 5
    fine = component.scaled(factor).render(
        (41 * factor, 41 * factor),
        origin=(factor * 20.0, factor * 20.0),
    )
    blocks = fine.reshape(41, factor, 41, factor).sum(axis=(1, 3))

    assert component.scaled(factor).re == pytest.approx(30.0)
    assert np.max(np.abs(blocks - native)) <= 1.0e-6 * native.max()
