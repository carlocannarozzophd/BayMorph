"""Validation tests for experimental hybrid Sérsic rendering."""

from math import erf

import numpy as np
import pytest

from baymorph.models import SersicModel
from baymorph.rendering.sersic_hybrid import (
    _apply_automatic_semianalytic_region,
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


def selector_result(
    model,
    *,
    tolerance,
    shape=(41, 41),
):
    """Run the automatic selector and expose its diagnostics."""
    baseline = model.render(
        shape=shape,
        renderer="fixed",
        integration_order=1,
    )

    return _apply_automatic_semianalytic_region(
        model,
        baseline,
        origin=(0.0, 0.0),
        unit_flux_pixel_tolerance=tolerance,
        guard_rings=2,
        max_half_width=16,
        epsabs=1.0e-13,
        epsrel=1.0e-12,
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


def test_automatic_selector_is_flux_scale_invariant():
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
    bright_model = SersicModel(
        flux=1.0e6,
        **parameters,
    )

    (
        _,
        unit_half_width,
        unit_pixels,
        unit_errors,
    ) = selector_result(
        unit_model,
        tolerance=1.0e-4,
    )

    (
        _,
        bright_half_width,
        bright_pixels,
        bright_errors,
    ) = selector_result(
        bright_model,
        tolerance=1.0e-4,
    )

    assert (
        bright_half_width
        == unit_half_width
    )
    assert bright_pixels == unit_pixels

    np.testing.assert_allclose(
        bright_errors,
        unit_errors,
        rtol=2.0e-10,
        atol=1.0e-15,
    )


def test_difficult_profile_requires_at_least_as_much_refinement_as_smooth_profile():
    smooth = SersicModel(
        flux=1.0,
        re=5.0,
        n=0.5,
        q=1.0,
        theta=0.0,
        x0=20.35,
        y0=20.40,
    )
    difficult = SersicModel(
        flux=1.0,
        re=0.5,
        n=10.0,
        q=0.2,
        theta=0.67,
        x0=20.49,
        y0=19.51,
    )

    (
        _,
        smooth_half_width,
        smooth_pixels,
        _,
    ) = selector_result(
        smooth,
        tolerance=1.0e-4,
    )

    (
        _,
        difficult_half_width,
        difficult_pixels,
        _,
    ) = selector_result(
        difficult,
        tolerance=1.0e-4,
    )

    assert (
        difficult_half_width
        >= smooth_half_width
    )
    assert (
        difficult_pixels
        >= smooth_pixels
    )


def test_automatic_selector_terminal_guard_rings_meet_tolerance():
    model = SersicModel(
        flux=1.0,
        re=0.5,
        n=10.0,
        q=0.2,
        theta=0.67,
        x0=20.49,
        y0=19.51,
    )

    (
        _,
        _,
        _,
        ring_errors,
    ) = selector_result(
        model,
        tolerance=1.0e-4,
    )

    assert len(ring_errors) >= 2
    assert ring_errors[-1] <= 1.0e-4
    assert ring_errors[-2] <= 1.0e-4


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
        unit_flux_pixel_tolerance=1.0e-4,
    )

    assert image.shape == (41, 41)
    assert np.all(np.isfinite(image))
    assert np.all(image >= 0.0)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        (
            "unit_flux_pixel_tolerance",
            0.0,
        ),
        (
            "unit_flux_pixel_tolerance",
            -1.0,
        ),
        (
            "guard_rings",
            0,
        ),
        (
            "max_half_width",
            -1,
        ),
    ],
)
def test_hybrid_renderer_rejects_invalid_selector_configuration(
    keyword,
    value,
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

    arguments = {
        "unit_flux_pixel_tolerance": 1.0e-4,
        keyword: value,
    }

    with pytest.raises(ValueError):
        render_sersic_hybrid(
            model,
            shape=(21, 21),
            **arguments,
        )


def test_selector_raises_when_maximum_region_is_insufficient():
    model = SersicModel(
        flux=1.0,
        re=0.5,
        n=10.0,
        q=0.2,
        theta=0.67,
        x0=20.49,
        y0=19.51,
    )

    with pytest.raises(
        RuntimeError,
        match="did not converge",
    ):
        render_sersic_hybrid(
            model,
            shape=(41, 41),
            unit_flux_pixel_tolerance=1.0e-12,
            guard_rings=2,
            max_half_width=1,
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


def test_collinear_pixel_vertices_are_merged_without_integration_warning():
    """Numerically coincident angular breakpoints must form one ray."""
    import warnings

    from scipy.integrate import IntegrationWarning

    from baymorph.rendering.sersic_hybrid import (
        _pixel_angular_breakpoints,
    )

    model = SersicModel(
        flux=1.0,
        re=0.5,
        n=2.0,
        q=1.0,
        theta=0.0,
        x0=32.49,
        y0=31.51,
    )

    breakpoints = _pixel_angular_breakpoints(
        model,
        pixel_x=33.0,
        pixel_y=31.0,
    )

    assert len(breakpoints) == 3

    with warnings.catch_warnings():
        warnings.simplefilter(
            "error",
            IntegrationWarning,
        )

        value = _semianalytic_pixel_integral(
            model,
            pixel_x=33.0,
            pixel_y=31.0,
        )

    assert np.isfinite(value)
    assert value > 0.0


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "flux": 2.3e5,
            "re": 8.7,
            "n": 1.0,
            "q": 0.73,
            "theta": 0.37,
            "x0": 15.2,
            "y0": 14.8,
        },
        {
            "flux": 8.1e5,
            "re": 21.4,
            "n": 4.0,
            "q": 0.41,
            "theta": -0.82,
            "x0": 17.31,
            "y0": 16.67,
        },
        {
            "flux": 1.7e6,
            "re": 32.0,
            "n": 7.5,
            "q": 0.88,
            "theta": 1.13,
            "x0": 18.49,
            "y0": 15.51,
        },
    ],
)
def test_fast_outer_renderer_matches_independent_fixed_reference(
    parameters,
):
    """Production outer rendering must match the generic fixed reference."""
    from baymorph.rendering.sersic_hybrid import (
        _render_outer_fixed_fast,
    )

    model = SersicModel(
        **parameters
    )

    shape = (37, 41)
    origin = (-3.0, 2.0)
    integration_order = 4

    reference = model.render(
        shape,
        renderer="fixed",
        integration_order=integration_order,
        origin=origin,
    )

    fast = _render_outer_fixed_fast(
        model,
        shape,
        origin=origin,
        integration_order=integration_order,
    )

    np.testing.assert_allclose(
        fast,
        reference,
        rtol=5.0e-13,
        atol=1.0e-13,
    )

    np.testing.assert_allclose(
        np.sum(fast),
        np.sum(reference),
        rtol=5.0e-14,
        atol=1.0e-12,
    )
