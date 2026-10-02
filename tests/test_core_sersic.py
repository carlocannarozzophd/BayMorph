"""The core-Sérsic profile (AD-048)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from baymorph.models.core_sersic import CoreSersicComponent
from baymorph.models.sersic import SersicModel

PARAMETERS = dict(
    flux=1000.0,
    re=6.0,
    n=4.0,
    rb=2.0,
    gamma=0.3,
    alpha=2.0,
    q=0.8,
    theta=0.3,
    x0=40.27,
    y0=39.87,
)


def _component(**changes):
    parameters = dict(PARAMETERS)
    parameters.update(changes)

    return CoreSersicComponent(**parameters)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("flux", -1.0),
        ("re", 0.0),
        ("n", -2.0),
        ("rb", 0.0),
        ("alpha", 0.0),
        ("gamma", 2.0),
        ("gamma", np.nan),
        ("q", 1.1),
        ("theta", np.nan),
    ],
)
def test_invalid_parameters_are_rejected(name, value):
    with pytest.raises(ValueError):
        _component(**{name: value})


def _radial_flux(component):
    """Total flux by an independent integral in the radius."""

    def integrand(radius):
        return (
            float(component.surface_brightness(np.asarray(radius)))
            * 2.0
            * np.pi
            * component.q
            * radius
        )

    edges = [0.0, component.rb, component.re, 50.0 * component.re]
    inner = sum(
        quad(integrand, low, high, limit=500)[0]
        for low, high in zip(edges[:-1], edges[1:], strict=True)
    )

    return inner + quad(integrand, edges[-1], np.inf, limit=500)[0]


@pytest.mark.parametrize(
    ("gamma", "alpha", "rb"),
    [(0.0, 2.0, 2.0), (0.3, 0.5, 0.5), (0.9, 2.0, 5.0), (0.9, 0.5, 2.0)],
)
def test_the_total_flux_is_the_flux_parameter(gamma, alpha, rb):
    component = _component(gamma=gamma, alpha=alpha, rb=rb)

    assert _radial_flux(component) == pytest.approx(component.flux, rel=1.0e-9)


def test_a_vanishing_core_gives_the_sersic_profile():
    component = _component(rb=1.0e-6)
    sersic = SersicModel(
        flux=PARAMETERS["flux"],
        re=PARAMETERS["re"],
        n=PARAMETERS["n"],
        q=PARAMETERS["q"],
        theta=PARAMETERS["theta"],
        x0=PARAMETERS["x0"],
        y0=PARAMETERS["y0"],
    )
    image = component.render((81, 81))
    reference = sersic.render((81, 81))

    assert np.max(np.abs(image - reference)) <= 1.0e-6 * reference.max()


@pytest.mark.parametrize("gamma", [0.0, 0.3, 0.9])
def test_the_inner_slope_is_gamma(gamma):
    """Well inside the break the profile is the power law R**-gamma."""
    component = _component(gamma=gamma)
    radii = np.array([1.0e-4, 2.0e-4]) * component.rb
    values = component.surface_brightness(radii)
    slope = np.log(values[1] / values[0]) / np.log(2.0)

    assert slope == pytest.approx(-gamma, abs=1.0e-3)


def test_the_rendering_conserves_the_flux_of_a_compact_core():
    """A compact profile on a large grid holds all of its flux."""
    component = _component(re=2.0, n=1.0, rb=0.5, gamma=0.9)
    image = component.render((81, 81))

    assert image.sum() == pytest.approx(component.flux, rel=1.0e-7)


def test_scaled_component_scales_both_radii():
    fine = _component().scaled(11)

    assert (fine.re, fine.rb) == pytest.approx((66.0, 22.0))
    assert fine.log_intensity_scale == pytest.approx(
        _component().log_intensity_scale - 2.0 * np.log(11.0),
        rel=1.0e-9,
    )


def test_cusps_steeper_than_the_rendering_resolves_fail_closed():
    """An inner slope near 2 does not converge and raises."""
    component = _component(gamma=1.9)

    with pytest.raises(ValueError, match="did not converge"):
        component.render((41, 41))
