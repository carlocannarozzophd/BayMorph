"""Validation tests for experimental hybrid Sérsic rendering."""

import numpy as np
import pytest

from baymorph.models import SersicModel
from baymorph.rendering.sersic_hybrid import (
    _split_central_pixel_integral,
    render_sersic_hybrid,
)


def make_compact_high_index_model() -> SersicModel:
    """Return a numerically demanding compact Sérsic model."""
    return SersicModel(
        flux=1.0e5,
        re=5.0,
        n=8.0,
        q=0.3,
        theta=0.6,
        x0=20.35,
        y0=20.40,
    )


def test_split_central_pixel_quadrature_converges():
    model = make_compact_high_index_model()

    value_256 = _split_central_pixel_integral(
        model,
        pixel_x=20.0,
        pixel_y=20.0,
        integration_order=256,
    )
    value_512 = _split_central_pixel_integral(
        model,
        pixel_x=20.0,
        pixel_y=20.0,
        integration_order=512,
    )

    relative_change = (
        abs(value_512 - value_256)
        / abs(value_512)
    )

    assert relative_change < 5.0e-7


def test_split_central_pixel_matches_validated_reference_value():
    model = make_compact_high_index_model()

    value = _split_central_pixel_integral(
        model,
        pixel_x=20.0,
        pixel_y=20.0,
        integration_order=512,
    )

    assert value == pytest.approx(
        1.6800641049e4,
        rel=5.0e-8,
    )


def test_hybrid_renderer_returns_finite_image():
    model = make_compact_high_index_model()

    image = render_sersic_hybrid(
        model,
        shape=(41, 41),
    )

    assert image.shape == (41, 41)
    assert np.all(np.isfinite(image))
    assert np.all(image >= 0.0)


def test_hybrid_renderer_refines_the_central_pixel():
    model = make_compact_high_index_model()

    image = render_sersic_hybrid(
        model,
        shape=(41, 41),
    )

    expected = _split_central_pixel_integral(
        model,
        pixel_x=20.0,
        pixel_y=20.0,
        integration_order=512,
    )

    assert image[20, 20] == pytest.approx(
        expected,
        rel=1.0e-6,
    )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("base_order", 0),
        ("refinement_order", 0),
        ("refinement_radius", -1.0),
        ("central_min_order", 0),
        ("central_max_order", 0),
        ("central_rtol", -1.0),
        ("central_atol", -1.0),
    ],
)
def test_hybrid_renderer_rejects_invalid_configuration(
    keyword,
    value,
):
    model = make_compact_high_index_model()

    kwargs = {
        keyword: value,
    }

    with pytest.raises(ValueError):
        render_sersic_hybrid(
            model,
            shape=(41, 41),
            **kwargs,
        )


def test_hybrid_renderer_rejects_reversed_central_orders():
    model = make_compact_high_index_model()

    with pytest.raises(
        ValueError,
        match="central_max_order",
    ):
        render_sersic_hybrid(
            model,
            shape=(41, 41),
            central_min_order=128,
            central_max_order=64,
        )
