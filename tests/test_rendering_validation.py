import pytest

from baymorph import SersicModel
from baymorph.validation.rendering import (
    validate_rendering_accuracy,
)


def make_model() -> SersicModel:
    return SersicModel(
        flux=1000.0,
        re=1.5,
        n=8.0,
        q=0.8,
        theta=0.3,
        x0=20.35,
        y0=19.72,
    )


def test_reference_against_itself_is_exact():
    result = validate_rendering_accuracy(
        make_model(),
        shape=(41, 41),
        reference_order=31,
        integration_orders=(31,),
    )[0]

    assert result.l1_error == pytest.approx(0.0)
    assert result.l2_error == pytest.approx(0.0)
    assert result.linf_error == pytest.approx(0.0)
    assert result.flux_error == pytest.approx(0.0)


def test_accuracy_improves_with_quadrature_order():
    results = validate_rendering_accuracy(
        make_model(),
        shape=(41, 41),
        reference_order=61,
        integration_orders=(3, 5, 7, 9, 11),
    )

    errors = [r.l1_error for r in results]

    assert errors[0] > errors[-1]


def test_low_order_is_not_exact():
    result = validate_rendering_accuracy(
        make_model(),
        shape=(41, 41),
        reference_order=61,
        integration_orders=(3,),
    )[0]

    assert result.l1_error > 0.0
