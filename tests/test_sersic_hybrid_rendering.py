"""Validation tests for experimental hybrid Sérsic rendering."""

from math import erf

import numpy as np
import pytest

from baymorph.models import SersicModel
from baymorph.rendering.sersic_hybrid import (
    _semianalytic_central_pixel_integral,
    _semianalytic_pixel_integral,
    _split_central_pixel_integral,
    render_sersic_hybrid,
)


def gaussian_exact_pixel(
    model,
    *,
    pixel_x,
    pixel_y,
):
    """Return the exact detector-pixel integral for a circular n=0.5 Sérsic."""
    scale = (
        np.sqrt(model.bn)
        / model.re
    )

    xmin = (
        pixel_x - 0.5 - model.x0
    )
    xmax = (
        pixel_x + 0.5 - model.x0
    )
    ymin = (
        pixel_y - 0.5 - model.y0
    )
    ymax = (
        pixel_y + 0.5 - model.y0
    )

    return (
        0.25
        * model.flux
        * (
            erf(scale * xmax)
            - erf(scale * xmin)
        )
        * (
            erf(scale * ymax)
            - erf(scale * ymin)
        )
    )


@pytest.mark.parametrize(
    ("pixel_x", "pixel_y"),
    [
        (10.0, 10.0),
        (11.0, 10.0),
        (8.0, 12.0),
        (15.0, 4.0),
    ],
)
def test_semianalytic_pixel_matches_gaussian_closed_form(
    pixel_x,
    pixel_y,
):
    model = SersicModel(
        flux=3.7,
        re=2.3,
        n=0.5,
        q=1.0,
        theta=0.73,
        x0=10.37,
        y0=9.82,
    )

    value = _semianalytic_pixel_integral(
        model,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
    )
    expected = gaussian_exact_pixel(
        model,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
    )

    np.testing.assert_allclose(
        value,
        expected,
        rtol=5.0e-13,
        atol=1.0e-15,
    )


def test_central_wrapper_matches_general_semianalytic_integral():
    model = SersicModel(
        flux=1.0,
        re=0.5,
        n=10.0,
        q=0.2,
        theta=0.67,
        x0=20.49,
        y0=19.51,
    )

    central = _semianalytic_central_pixel_integral(
        model,
        pixel_x=20.0,
        pixel_y=20.0,
    )
    general = _semianalytic_pixel_integral(
        model,
        pixel_x=20.0,
        pixel_y=20.0,
    )

    np.testing.assert_allclose(
        central,
        general,
        rtol=0.0,
        atol=0.0,
    )


def test_high_order_split_integral_converges_towards_semianalytic_value():
    model = SersicModel(
        flux=1.0,
        re=0.5,
        n=10.0,
        q=0.2,
        theta=0.0,
        x0=20.49,
        y0=19.51,
    )

    reference = _semianalytic_pixel_integral(
        model,
        pixel_x=20.0,
        pixel_y=20.0,
    )

    value_512 = _split_central_pixel_integral(
        model,
        pixel_x=20.0,
        pixel_y=20.0,
        integration_order=512,
    )
    value_1024 = _split_central_pixel_integral(
        model,
        pixel_x=20.0,
        pixel_y=20.0,
        integration_order=1024,
    )

    assert (
        abs(value_1024 - reference)
        < abs(value_512 - reference)
    )


def test_semianalytic_pixel_scales_linearly_with_flux():
    parameters = {
        "re": 0.5,
        "n": 10.0,
        "q": 0.2,
        "theta": 0.67,
        "x0": 20.49,
        "y0": 19.51,
    }

    unit_model = SersicModel(
        flux=1.0,
        **parameters,
    )
    scaled_model = SersicModel(
        flux=1234.0,
        **parameters,
    )

    unit_value = _semianalytic_pixel_integral(
        unit_model,
        pixel_x=21.0,
        pixel_y=20.0,
    )
    scaled_value = _semianalytic_pixel_integral(
        scaled_model,
        pixel_x=21.0,
        pixel_y=20.0,
    )

    np.testing.assert_allclose(
        scaled_value,
        1234.0 * unit_value,
        rtol=2.0e-13,
        atol=0.0,
    )


def test_hybrid_renderer_replaces_inner_patch_with_semianalytic_values():
    model = SersicModel(
        flux=1.0,
        re=0.5,
        n=10.0,
        q=0.2,
        theta=0.67,
        x0=20.49,
        y0=19.51,
    )

    image = render_sersic_hybrid(
        model,
        shape=(41, 41),
        semianalytic_half_width=1,
    )

    central_x = int(
        np.floor(
            model.x0 + 0.5
        )
    )
    central_y = int(
        np.floor(
            model.y0 + 0.5
        )
    )

    for pixel_y in range(
        central_y - 1,
        central_y + 2,
    ):
        for pixel_x in range(
            central_x - 1,
            central_x + 2,
        ):
            expected = (
                _semianalytic_pixel_integral(
                    model,
                    pixel_x=float(pixel_x),
                    pixel_y=float(pixel_y),
                )
            )

            np.testing.assert_allclose(
                image[pixel_y, pixel_x],
                expected,
                rtol=2.0e-12,
                atol=1.0e-14,
            )


def test_hybrid_renderer_returns_finite_non_negative_image():
    model = SersicModel(
        flux=1.0e5,
        re=5.0,
        n=8.0,
        q=0.3,
        theta=0.6,
        x0=20.35,
        y0=20.40,
    )

    image = render_sersic_hybrid(
        model,
        shape=(41, 41),
    )

    assert image.shape == (41, 41)
    assert np.all(np.isfinite(image))
    assert np.all(image >= 0.0)


@pytest.mark.parametrize(
    "half_width",
    [
        -1,
        1.5,
    ],
)
def test_hybrid_renderer_rejects_invalid_half_width(
    half_width,
):
    model = SersicModel(
        flux=1.0,
        re=3.0,
        n=2.0,
        q=0.8,
        theta=0.2,
        x0=10.0,
        y0=10.0,
    )

    with pytest.raises(
        ValueError,
        match="semianalytic_half_width",
    ):
        render_sersic_hybrid(
            model,
            shape=(21, 21),
            semianalytic_half_width=half_width,
        )


def test_central_wrapper_rejects_noncentral_pixel():
    model = SersicModel(
        flux=1.0,
        re=3.0,
        n=2.0,
        q=0.8,
        theta=0.2,
        x0=10.0,
        y0=10.0,
    )

    with pytest.raises(
        ValueError,
        match="containing the Sersic centre",
    ):
        _semianalytic_central_pixel_integral(
            model,
            pixel_x=8.0,
            pixel_y=8.0,
        )
