"""The exponential disc with two logarithmic spiral arms (AD-048)."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting import (
    make_bulge_disk_psf_model,
    make_bulge_disk_spiral_psf_model,
)
from baymorph.models import ExponentialComponent, LogSpiralDiskComponent
from baymorph.simulation.psf import GaussianPSF

PARAMETERS = dict(
    flux=1000.0,
    re=7.0,
    q=0.7,
    theta=0.55,
    amplitude=0.8,
    winding=3.0,
    phase=0.9,
    r_in=2.0,
    x0=60.31,
    y0=59.77,
)


def _disc(**changes):
    parameters = dict(PARAMETERS)
    parameters.update(changes)

    return LogSpiralDiskComponent(**parameters)


def _on_circle(disc, radius, azimuth):
    """Return detector coordinates at a radius and azimuth in the disc plane."""
    major = radius * np.cos(azimuth)
    minor = disc.q * radius * np.sin(azimuth)
    cos_theta, sin_theta = np.cos(disc.theta), np.sin(disc.theta)

    return (
        disc.x0 + major * cos_theta - minor * sin_theta,
        disc.y0 + major * sin_theta + minor * cos_theta,
    )


def _crest(disc, radius):
    """Return the disc-plane azimuth of a crest of the arms at ``radius``."""
    azimuth = np.linspace(-np.pi, np.pi, 200001)
    values = disc.modulation.intensity(*_on_circle(disc, radius, azimuth))

    return azimuth[np.argmax(values)]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("flux", 0.0),
        ("re", 0.0),
        ("q", 1.5),
        ("theta", np.nan),
        ("amplitude", -0.1),
        ("amplitude", 1.2),
        ("winding", np.inf),
        ("phase", np.nan),
        ("r_in", 0.0),
        ("x0", np.nan),
    ],
)
def test_invalid_parameters_are_rejected(name, value):
    with pytest.raises(ValueError):
        _disc(**{name: value})


@pytest.mark.parametrize("renderer", ["hybrid", "point"])
def test_without_arms_the_disc_is_the_exponential_disc(renderer):
    """With A0 = 0 the disc renders bit for bit as the production disc."""
    disc = _disc(amplitude=0.0)
    exponential = ExponentialComponent(
        flux=disc.flux,
        re=disc.re,
        q=disc.q,
        theta=disc.theta,
        x0=disc.x0,
        y0=disc.y0,
    )

    assert np.array_equal(
        disc.render((121, 121), renderer=renderer),
        exponential.render((121, 121), renderer=renderer),
    )


@pytest.mark.parametrize("effective", [False, True])
def test_bulge_disk_spiral_without_arms_is_bulge_disk(
    effective,
    subpixel_gaussian_psf,
):
    """Bit for bit, under an analytic PSF and under the effective PSF."""
    psf = subpixel_gaussian_psf if effective else GaussianPSF(fwhm=2.5)
    shared = dict(
        flux_bulge=300.0,
        re_bulge=2.0,
        n_bulge=3.0,
        q_bulge=0.9,
        theta_bulge=0.2,
        flux_disk=700.0,
        re_disk=9.0,
        q_disk=0.7,
        theta_disk=0.5,
        x0=40.2,
        y0=39.7,
        psf=psf,
    )
    spiral = make_bulge_disk_spiral_psf_model(
        **shared,
        amplitude_spiral=0.0,
        winding_spiral=2.0,
        phase_spiral=0.4,
        r_in_spiral=3.0,
    )

    assert np.array_equal(
        spiral.render((81, 81)),
        make_bulge_disk_psf_model(**shared).render((81, 81)),
    )


def test_the_arms_add_no_flux():
    """The modulation of a compact disc integrates to zero."""
    disc = _disc(re=4.0)

    assert abs(disc.modulation.render((121, 121)).sum()) <= 1.0e-8 * disc.flux
    assert disc.render((121, 121)).sum() == pytest.approx(disc.flux, rel=1.0e-8)


@pytest.mark.parametrize("radius", [4.0, 7.0, 12.0])
def test_the_crests_follow_the_logarithmic_spiral(radius):
    """At disc-plane radius R a crest lies at phi0 + w ln(R / re), modulo pi."""
    disc = _disc()
    expected = disc.phase + disc.winding * np.log(radius / disc.re)
    offset = (_crest(disc, radius) - expected + 0.5 * np.pi) % np.pi - 0.5 * np.pi

    assert offset == pytest.approx(0.0, abs=1.0e-4)


@pytest.mark.parametrize(("winding", "turn"), [(2.0, 1.0), (-2.0, -1.0)])
def test_positive_winding_draws_a_z_and_negative_an_s(winding, turn):
    """On the detector, with x to the right and y up, w > 0 winds counter-clockwise.

    A crest farther out lies at a larger detector azimuth for w > 0, the
    arms of a Z, and at a smaller one for w < 0, the arms of an S.
    """
    disc = _disc(q=1.0, winding=winding, amplitude=1.0, r_in=0.5)
    azimuths = []

    for radius in (7.0, 12.0):
        x, y = _on_circle(disc, radius, _crest(disc, radius))
        azimuths.append(np.arctan2(y - disc.y0, x - disc.x0))

    change = (azimuths[1] - azimuths[0] + 0.5 * np.pi) % np.pi - 0.5 * np.pi

    assert np.sign(change) == turn
    assert abs(change) == pytest.approx(abs(winding) * np.log(12.0 / 7.0), rel=1e-3)


def test_the_local_wavelength_is_that_of_the_arms():
    """It is 2 pi over the gradient of the phase, and vanishes at the centre."""
    disc = _disc()
    modulation = disc.modulation
    x = disc.x0 + np.array([0.0, 3.0, -5.0, 9.0])
    y = disc.y0 + np.array([0.0, 4.0, 2.0, -6.0])
    wavelength = modulation.local_wavelength(x, y)

    def phase(xx, yy):
        major = (xx - disc.x0) * np.cos(disc.theta) + (yy - disc.y0) * np.sin(
            disc.theta
        )
        minor = -(xx - disc.x0) * np.sin(disc.theta) + (yy - disc.y0) * np.cos(
            disc.theta
        )
        radius = np.hypot(major, minor / disc.q)

        return 2.0 * (
            np.arctan2(minor / disc.q, major) - disc.winding * np.log(radius)
        )

    step = 1.0e-6
    gradient = np.hypot(
        (phase(x[1:] + step, y[1:]) - phase(x[1:] - step, y[1:])) / (2 * step),
        (phase(x[1:], y[1:] + step) - phase(x[1:], y[1:] - step)) / (2 * step),
    )

    assert wavelength[0] == 0.0
    np.testing.assert_allclose(wavelength[1:], 2.0 * np.pi / gradient, rtol=1e-6)


def test_the_hybrid_rule_agrees_with_adaptive_integration():
    modulation = _disc().modulation
    hybrid = modulation.render((121, 121))
    adaptive = modulation.render((121, 121), renderer="adaptive", rtol=1.0e-10)

    assert np.max(np.abs(hybrid - adaptive)) <= 1.0e-7 * np.abs(adaptive).max()


def test_scaled_disc_keeps_the_flux_of_each_cell():
    """Cells of the finer grid sum to the cells of the native grid."""
    disc = _disc(re=5.0)
    native = disc.render((41, 41), origin=(40.0, 40.0))
    factor = 5
    fine = disc.scaled(factor).render(
        (41 * factor, 41 * factor),
        origin=(factor * 40.0, factor * 40.0),
    )
    blocks = fine.reshape(41, factor, 41, factor).sum(axis=(1, 3))

    assert (disc.scaled(factor).re, disc.scaled(factor).r_in) == pytest.approx(
        (25.0, 10.0)
    )
    assert np.max(np.abs(blocks - native)) <= 1.0e-6 * native.max()
