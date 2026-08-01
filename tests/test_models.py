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
