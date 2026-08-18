import numpy as np
import pytest
from scipy.integrate import quad

from baymorph import SersicModel


def make_model() -> SersicModel:
    return SersicModel(
        flux=1000.0,
        re=10.0,
        n=4.0,
        q=0.7,
        theta=0.5,
        x0=50.0,
        y0=50.0,
    )


def test_sersic_model_shape():
    image = make_model().render(shape=(100, 100))

    assert image.shape == (100, 100)


def test_sersic_continuous_profile_has_requested_total_flux():
    model = make_model()

    radial_integral, _ = quad(
        lambda radius: (
            2.0
            * np.pi
            * model.q
            * float(model.surface_brightness(radius))
            * radius
        ),
        0.0,
        np.inf,
        epsabs=1.0e-8,
        epsrel=1.0e-10,
    )

    assert radial_integral == pytest.approx(
        model.flux,
        rel=1.0e-9,
    )


def test_sersic_effective_radius_encloses_half_total_flux():
    model = make_model()

    enclosed_flux, _ = quad(
        lambda radius: (
            2.0
            * np.pi
            * model.q
            * float(model.surface_brightness(radius))
            * radius
        ),
        0.0,
        model.re,
        epsabs=1.0e-8,
        epsrel=1.0e-10,
    )

    assert enclosed_flux == pytest.approx(
        0.5 * model.flux,
        rel=1.0e-9,
    )


def test_sersic_render_is_invariant_to_cutout_extension():
    model = SersicModel(
        flux=1000.0,
        re=10.0,
        n=4.0,
        q=0.7,
        theta=0.5,
        x0=40.0,
        y0=40.0,
    )

    smaller = model.render(shape=(81, 81))
    larger = model.render(shape=(161, 161))

    np.testing.assert_allclose(
        smaller,
        larger[:81, :81],
        rtol=0.0,
        atol=0.0,
    )


def test_sersic_model_peak_positive():
    image = make_model().render(shape=(100, 100))

    assert image.max() > 0.0


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("flux", 0.0),
        ("re", 0.0),
        ("n", 0.0),
        ("q", 0.0),
        ("flux", np.nan),
        ("re", np.inf),
    ],
)
def test_sersic_model_rejects_invalid_parameters(parameter, value):
    parameters = {
        "flux": 1000.0,
        "re": 10.0,
        "n": 4.0,
        "q": 0.7,
        "theta": 0.5,
        "x0": 50.0,
        "y0": 50.0,
    }
    parameters[parameter] = value

    with pytest.raises(ValueError):
        SersicModel(**parameters)


def test_sersic_render_integrates_over_pixel_area():
    model = SersicModel(
        flux=1000.0,
        re=3.2,
        n=1.5,
        q=0.8,
        theta=0.3,
        x0=10.37,
        y0=9.82,
    )

    image = model.render(
        shape=(21, 21),
        integration_order=9,
    )
    reference = model.render(
        shape=(21, 21),
        integration_order=31,
    )

    relative_l1_error = (
        np.abs(image - reference).sum()
        / reference.sum()
    )

    assert relative_l1_error < 1.0e-4


def test_sersic_pixel_integrator_supports_scalar_and_batched_coordinates():
    model = SersicModel(
        flux=1000.0,
        re=3.2,
        n=1.5,
        q=0.8,
        theta=0.3,
        x0=10.37,
        y0=9.82,
    )

    pixel_x = np.arange(9.0, 13.0)[None, :]
    pixel_y = np.arange(8.0, 11.0)[:, None]

    batched = model._integrate_pixel(
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        integration_order=7,
    )
    scalar = np.array(
        [
            [
                model._integrate_pixel(
                    pixel_x=x,
                    pixel_y=y,
                    integration_order=7,
                )
                for x in pixel_x[0]
            ]
            for y in pixel_y[:, 0]
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        batched,
        scalar,
        rtol=1.0e-14,
        atol=1.0e-15,
    )


def test_sersic_refined_rectangle_matches_explicit_quadrisection():
    model = SersicModel(
        flux=1000.0,
        re=3.2,
        n=1.5,
        q=0.8,
        theta=0.3,
        x0=10.37,
        y0=9.82,
    )

    refined = model._integrate_refined_rectangle(
        xmin=9.0,
        xmax=10.0,
        ymin=8.0,
        ymax=9.0,
        integration_order=7,
    )

    explicit = sum(
        model._integrate_rectangle(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            integration_order=7,
        )
        for ymin, ymax in ((8.0, 8.5), (8.5, 9.0))
        for xmin, xmax in ((9.0, 9.5), (9.5, 10.0))
    )

    np.testing.assert_allclose(
        refined,
        explicit,
        rtol=0.0,
        atol=0.0,
    )


def test_sersic_large_cutout_recovers_total_flux():
    model = SersicModel(
        flux=1000.0,
        re=5.0,
        n=1.0,
        q=0.8,
        theta=0.4,
        x0=100.0,
        y0=100.0,
    )

    image = model.render(
        shape=(201, 201),
        integration_order=7,
    )

    assert image.sum() == pytest.approx(
        model.flux,
        rel=5.0e-5,
    )


def test_sersic_pixel_integration_is_cutout_invariant():
    model = SersicModel(
        flux=1000.0,
        re=8.0,
        n=2.0,
        q=0.65,
        theta=0.7,
        x0=30.4,
        y0=29.7,
    )

    smaller = model.render(
        shape=(61, 61),
        integration_order=5,
    )
    larger = model.render(
        shape=(121, 121),
        integration_order=5,
    )

    np.testing.assert_allclose(
        smaller,
        larger[:61, :61],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("integration_order", [0, -1, 1.5])
def test_sersic_render_rejects_invalid_integration_order(
    integration_order,
):
    with pytest.raises(ValueError):
        make_model().render(
            shape=(21, 21),
            integration_order=integration_order,
        )


def test_sersic_model_rejects_axis_ratio_above_unity():
    with pytest.raises(ValueError):
        SersicModel(
            flux=1000.0,
            re=10.0,
            n=4.0,
            q=1.1,
            theta=0.5,
            x0=50.0,
            y0=50.0,
        )


def test_sersic_rectangle_integrator_recovers_pixel_integral():
    model = SersicModel(
        flux=1000.0,
        re=3.2,
        n=1.5,
        q=0.8,
        theta=0.3,
        x0=10.37,
        y0=9.82,
    )

    pixel_x = np.arange(9.0, 13.0)[None, :]
    pixel_y = np.arange(8.0, 11.0)[:, None]

    pixel_flux = model._integrate_pixel(
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        integration_order=7,
    )
    rectangle_flux = model._integrate_rectangle(
        xmin=pixel_x - 0.5,
        xmax=pixel_x + 0.5,
        ymin=pixel_y - 0.5,
        ymax=pixel_y + 0.5,
        integration_order=7,
    )

    np.testing.assert_allclose(
        rectangle_flux,
        pixel_flux,
        rtol=0.0,
        atol=0.0,
    )


def test_sersic_rectangle_integrator_is_additive_under_quadrisection():
    model = SersicModel(
        flux=1000.0,
        re=3.2,
        n=1.5,
        q=0.8,
        theta=0.3,
        x0=10.37,
        y0=9.82,
    )

    whole = model._integrate_rectangle(
        xmin=9.0,
        xmax=10.0,
        ymin=8.0,
        ymax=9.0,
        integration_order=15,
    )

    quadrants = sum(
        model._integrate_rectangle(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            integration_order=15,
        )
        for ymin, ymax in ((8.0, 8.5), (8.5, 9.0))
        for xmin, xmax in ((9.0, 9.5), (9.5, 10.0))
    )

    np.testing.assert_allclose(
        quadrants,
        whole,
        rtol=1.0e-12,
        atol=1.0e-14,
    )


def test_sersic_default_render_uses_validated_hybrid_renderer():
    """Default Sérsic rendering must use the validated hybrid algorithm."""
    from baymorph.rendering.sersic_hybrid import (
        render_sersic_hybrid,
    )

    model = SersicModel(
        flux=1.0,
        re=0.5,
        n=10.0,
        q=0.2,
        theta=0.67,
        x0=20.49,
        y0=19.51,
    )

    default = model.render(
        shape=(41, 41),
    )

    explicit = render_sersic_hybrid(
        model,
        shape=(41, 41),
    )

    np.testing.assert_allclose(
        default,
        explicit,
        rtol=0.0,
        atol=0.0,
    )


def test_sersic_fixed_renderer_remains_available_for_validation():
    """Fixed quadrature must remain explicitly accessible."""
    model = SersicModel(
        flux=1.0,
        re=0.5,
        n=10.0,
        q=0.2,
        theta=0.67,
        x0=20.49,
        y0=19.51,
    )

    hybrid = model.render(
        shape=(41, 41),
    )

    fixed = model.render(
        shape=(41, 41),
        renderer="fixed",
        integration_order=4,
    )

    assert hybrid.shape == fixed.shape
    assert np.all(np.isfinite(hybrid))
    assert np.all(np.isfinite(fixed))

    assert (
        np.linalg.norm(
            hybrid - fixed
        )
        > 1.0e-3
    )
