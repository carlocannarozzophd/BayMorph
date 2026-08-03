import math

import pytest

from baymorph.rendering import IntegrationCell, estimate_cell


def test_estimate_cell_accepts_exact_constant_integral():
    cell = IntegrationCell(
        xmin=0.0,
        xmax=2.0,
        ymin=-1.0,
        ymax=3.0,
    )

    def integrate(candidate):
        return 5.0 * candidate.area

    estimate = estimate_cell(
        cell,
        integrate,
        rtol=1.0e-8,
        atol=0.0,
    )

    assert estimate.coarse == pytest.approx(40.0)
    assert estimate.fine == pytest.approx(40.0)
    assert estimate.error == pytest.approx(0.0)
    assert estimate.tolerance == pytest.approx(4.0e-7)
    assert estimate.accepted


def test_estimate_cell_rejects_inaccurate_coarse_estimate():
    cell = IntegrationCell(
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
    )

    def integrate(candidate):
        if candidate.level == 0:
            return 1.0
        return 0.3

    estimate = estimate_cell(
        cell,
        integrate,
        rtol=0.05,
        atol=0.0,
    )

    assert estimate.coarse == pytest.approx(1.0)
    assert estimate.fine == pytest.approx(1.2)
    assert estimate.error == pytest.approx(0.2)
    assert estimate.tolerance == pytest.approx(0.06)
    assert not estimate.accepted


def test_estimate_cell_uses_absolute_tolerance_near_zero():
    cell = IntegrationCell(
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
    )

    def integrate(candidate):
        if candidate.level == 0:
            return 0.0
        return 2.0e-9

    estimate = estimate_cell(
        cell,
        integrate,
        rtol=0.0,
        atol=1.0e-8,
    )

    assert estimate.fine == pytest.approx(8.0e-9)
    assert estimate.error == pytest.approx(8.0e-9)
    assert estimate.accepted


@pytest.mark.parametrize(
    ("rtol", "atol", "message"),
    [
        (-1.0, 0.0, "rtol must be non-negative"),
        (0.0, -1.0, "atol must be non-negative"),
    ],
)
def test_estimate_cell_rejects_negative_tolerances(
    rtol,
    atol,
    message,
):
    cell = IntegrationCell(0.0, 1.0, 0.0, 1.0)

    with pytest.raises(ValueError, match=message):
        estimate_cell(
            cell,
            lambda candidate: candidate.area,
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize(
    ("invalid_value", "message"),
    [
        (math.nan, "Coarse integration estimate must be finite"),
        (math.inf, "Coarse integration estimate must be finite"),
    ],
)
def test_estimate_cell_rejects_invalid_coarse_estimate(
    invalid_value,
    message,
):
    cell = IntegrationCell(0.0, 1.0, 0.0, 1.0)

    with pytest.raises(ValueError, match=message):
        estimate_cell(
            cell,
            lambda candidate: invalid_value,
            rtol=1.0e-6,
            atol=0.0,
        )


def test_estimate_cell_rejects_invalid_refined_estimate():
    cell = IntegrationCell(0.0, 1.0, 0.0, 1.0)

    def integrate(candidate):
        if candidate.level == 0:
            return 1.0
        return math.nan

    with pytest.raises(
        ValueError,
        match="Refined integration estimate must be finite",
    ):
        estimate_cell(
            cell,
            integrate,
            rtol=1.0e-6,
            atol=0.0,
        )
