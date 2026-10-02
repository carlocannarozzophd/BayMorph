"""The broken exponential disc (AD-048)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import gammaincinv

from baymorph.models import BrokenExponentialComponent, ExponentialComponent

PARAMETERS = dict(
    flux=1000.0,
    h_in=4.0,
    h_out=2.0,
    rb=10.0,
    wb=1.0,
    q=0.6,
    theta=0.3,
    x0=60.27,
    y0=59.81,
)


def _disk(**changes):
    parameters = dict(PARAMETERS)
    parameters.update(changes)

    return BrokenExponentialComponent(**parameters)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("flux", 0.0),
        ("h_in", -1.0),
        ("h_out", 0.0),
        ("rb", 0.0),
        ("wb", 0.0),
        ("q", 1.5),
        ("theta", np.nan),
        ("y0", np.inf),
    ],
)
def test_invalid_parameters_are_rejected(name, value):
    with pytest.raises(ValueError):
        _disk(**{name: value})


def test_an_unbroken_disc_is_the_exponential_disc():
    """With h_in = h_out the profile is the production exponential disc."""
    disk = _disk(h_in=4.0, h_out=4.0)
    exponential = ExponentialComponent(
        flux=PARAMETERS["flux"],
        re=4.0 * float(gammaincinv(2.0, 0.5)),
        q=PARAMETERS["q"],
        theta=PARAMETERS["theta"],
        x0=PARAMETERS["x0"],
        y0=PARAMETERS["y0"],
    )
    image = disk.render((121, 121))
    reference = exponential.render((121, 121))

    assert np.max(np.abs(image - reference)) <= 1.0e-6 * reference.max()


@pytest.mark.parametrize(
    ("h_in", "h_out", "wb"),
    [(4.0, 2.0, 0.5), (4.0, 2.0, 5.0), (2.0, 3.0, 1.0)],
)
def test_the_total_flux_is_the_flux_parameter(h_in, h_out, wb):
    """A compact disc on a large grid holds all of its flux."""
    disk = _disk(h_in=h_in, h_out=h_out, wb=wb)
    image = disk.render((121, 121))

    assert image.sum() == pytest.approx(disk.flux, rel=1.0e-8)


@pytest.mark.parametrize(("h_in", "h_out"), [(4.0, 2.0), (3.0, 6.0)])
def test_the_scale_lengths_hold_on_either_side_of_the_break(h_in, h_out):
    disk = _disk(h_in=h_in, h_out=h_out, rb=20.0, wb=0.5)
    inner = np.log(disk.surface_brightness(np.array([2.0, 3.0])))
    outer = np.log(disk.surface_brightness(np.array([60.0, 61.0])))

    assert inner[1] - inner[0] == pytest.approx(-1.0 / h_in, rel=1.0e-6)
    assert outer[1] - outer[0] == pytest.approx(-1.0 / h_out, rel=1.0e-6)


def test_scaled_disc_scales_every_length():
    fine = _disk().scaled(11)

    assert (fine.h_in, fine.h_out, fine.rb, fine.wb) == pytest.approx(
        (44.0, 22.0, 110.0, 11.0)
    )
    assert fine.log_central_intensity == pytest.approx(
        _disk().log_central_intensity - 2.0 * np.log(11.0),
        rel=1.0e-9,
    )
