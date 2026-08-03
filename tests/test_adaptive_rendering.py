"""Tests for adaptive numerical rendering."""

from __future__ import annotations

import math
from collections import Counter

import pytest

from baymorph.rendering import (
    IntegrationCell,
    estimate_cell,
    integrate_adaptively,
)


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


def test_integrate_adaptively_accepts_exact_integral_without_recursion():
    cell = IntegrationCell(
        xmin=0.0,
        xmax=2.0,
        ymin=-1.0,
        ymax=3.0,
    )

    result = integrate_adaptively(
        cell,
        lambda candidate: 5.0 * candidate.area,
        rtol=1.0e-10,
        atol=0.0,
        max_depth=5,
    )

    assert result.value == pytest.approx(40.0)
    assert result.estimated_error == pytest.approx(0.0)
    assert result.accepted_cells == 1
    assert result.refined_cells == 0
    assert result.maximum_level == 0


def test_integrate_adaptively_refines_until_estimate_converges():
    cell = IntegrationCell(
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
    )

    def integrate(candidate):
        exact = candidate.area
        bias = candidate.area * 0.5 ** (candidate.level + 1)
        return exact + bias

    result = integrate_adaptively(
        cell,
        integrate,
        rtol=0.0,
        atol=0.02,
        max_depth=6,
    )

    assert result.value == pytest.approx(1.015625)
    assert result.estimated_error <= 0.02 + 1.0e-15
    assert result.accepted_cells == 256
    assert result.refined_cells == 85
    assert result.maximum_level == 4


def test_integrate_adaptively_stops_at_maximum_depth():
    cell = IntegrationCell(
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
    )

    def integrate(candidate):
        return candidate.area * (
            1.0 + 0.5 ** (candidate.level + 1)
        )

    result = integrate_adaptively(
        cell,
        integrate,
        rtol=0.0,
        atol=0.0,
        max_depth=2,
    )

    assert result.accepted_cells == 16
    assert result.refined_cells == 5
    assert result.maximum_level == 2


def test_integrate_adaptively_integrates_each_cell_once():
    cell = IntegrationCell(
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
    )
    evaluations: Counter[IntegrationCell] = Counter()

    def integrate(candidate):
        evaluations[candidate] += 1

        return candidate.area * (
            1.0 + 0.5 ** (candidate.level + 1)
        )

    result = integrate_adaptively(
        cell,
        integrate,
        rtol=0.0,
        atol=0.0,
        max_depth=3,
    )

    visited_cells = result.accepted_cells + result.refined_cells

    assert all(count == 1 for count in evaluations.values())
    assert len(evaluations) == 1 + 4 * visited_cells


def test_integrate_adaptively_rejects_invalid_tolerances():
    cell = IntegrationCell(
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
    )

    with pytest.raises(
        ValueError,
        match="rtol must be non-negative",
    ):
        integrate_adaptively(
            cell,
            lambda candidate: candidate.area,
            rtol=-1.0,
            atol=0.0,
            max_depth=1,
        )

    with pytest.raises(
        ValueError,
        match="atol must be non-negative",
    ):
        integrate_adaptively(
            cell,
            lambda candidate: candidate.area,
            rtol=0.0,
            atol=-1.0,
            max_depth=1,
        )


def test_integrate_adaptively_rejects_invalid_maximum_depth():
    cell = IntegrationCell(
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        level=2,
    )

    with pytest.raises(
        ValueError,
        match="max_depth must be greater than or equal to the cell level",
    ):
        integrate_adaptively(
            cell,
            lambda candidate: candidate.area,
            rtol=1.0e-6,
            atol=0.0,
            max_depth=1,
        )
