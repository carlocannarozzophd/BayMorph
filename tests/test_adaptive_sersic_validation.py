"""Numerical validation of adaptive Sérsic pixel integration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pytest

from baymorph import SersicModel
from baymorph.rendering import (
    AdaptiveIntegrationResult,
    IntegrationCell,
    integrate_adaptively,
)


@dataclass
class CountingSersicIntegrator:
    """Integrate Sérsic cells while recording deterministic cost."""

    model: SersicModel
    integration_order: int
    calls: int = 0
    point_evaluations: int = 0
    evaluated_cells: Counter[IntegrationCell] = field(
        default_factory=Counter
    )

    def __call__(self, cell: IntegrationCell) -> float:
        """Integrate one cell and update deterministic diagnostics."""
        self.calls += 1
        self.point_evaluations += self.integration_order**2
        self.evaluated_cells[cell] += 1

        return float(
            self.model._integrate_rectangle(
                xmin=cell.xmin,
                xmax=cell.xmax,
                ymin=cell.ymin,
                ymax=cell.ymax,
                integration_order=self.integration_order,
            )
        )


def make_pixel_cell(
    x: float,
    y: float,
) -> IntegrationCell:
    """Create the unit detector pixel centred on ``(x, y)``."""
    return IntegrationCell(
        xmin=x - 0.5,
        xmax=x + 0.5,
        ymin=y - 0.5,
        ymax=y + 0.5,
    )


def reference_integral(
    model: SersicModel,
    cell: IntegrationCell,
) -> float:
    """Return a composite high-order reference integral.

    When the Sérsic centre lies inside the cell, the domain is split at
    ``x0`` and ``y0`` so that the central cusp lies on subdomain boundaries
    rather than inside one Gauss-Legendre integration region.
    """
    x_edges = sorted(
        {
            cell.xmin,
            max(cell.xmin, min(model.x0, cell.xmax)),
            cell.xmax,
        }
    )
    y_edges = sorted(
        {
            cell.ymin,
            max(cell.ymin, min(model.y0, cell.ymax)),
            cell.ymax,
        }
    )

    return float(
        sum(
            model._integrate_rectangle(
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                integration_order=61,
            )
            for xmin, xmax in zip(
                x_edges[:-1],
                x_edges[1:],
                strict=True,
            )
            for ymin, ymax in zip(
                y_edges[:-1],
                y_edges[1:],
                strict=True,
            )
            if xmax > xmin and ymax > ymin
        )
    )


@pytest.mark.parametrize(
    ("pixel_x", "pixel_y"),
    [
        (0.0, 0.0),
        (3.0, 1.0),
        (10.0, -2.0),
    ],
)
def test_adaptive_sersic_integral_matches_high_order_reference(
    pixel_x,
    pixel_y,
):
    model = SersicModel(
        flux=1000.0,
        re=3.0,
        n=2.0,
        q=0.75,
        theta=0.4,
        x0=0.17,
        y0=-0.23,
    )
    cell = make_pixel_cell(pixel_x, pixel_y)
    integrator = CountingSersicIntegrator(
        model=model,
        integration_order=5,
    )

    result = integrate_adaptively(
        cell,
        integrator,
        rtol=1.0e-7,
        atol=1.0e-13,
        max_depth=7,
    )
    reference = reference_integral(model, cell)

    np.testing.assert_allclose(
        result.value,
        reference,
        rtol=5.0e-7,
        atol=1.0e-12,
    )


def test_adaptive_sersic_refinement_is_concentrated_near_centre():
    model = SersicModel(
        flux=1000.0,
        re=3.0,
        n=4.0,
        q=0.7,
        theta=0.3,
        x0=0.13,
        y0=-0.21,
    )

    central_integrator = CountingSersicIntegrator(
        model=model,
        integration_order=3,
    )
    outer_integrator = CountingSersicIntegrator(
        model=model,
        integration_order=3,
    )

    central_result = integrate_adaptively(
        make_pixel_cell(0.0, 0.0),
        central_integrator,
        rtol=1.0e-6,
        atol=1.0e-14,
        max_depth=7,
    )
    outer_result = integrate_adaptively(
        make_pixel_cell(15.0, 0.0),
        outer_integrator,
        rtol=1.0e-6,
        atol=1.0e-14,
        max_depth=7,
    )

    assert central_result.maximum_level > outer_result.maximum_level
    assert (
        central_integrator.point_evaluations
        > outer_integrator.point_evaluations
    )


def test_adaptive_cost_accounting_matches_recursion_tree():
    model = SersicModel(
        flux=1000.0,
        re=3.0,
        n=4.0,
        q=0.7,
        theta=0.3,
        x0=0.13,
        y0=-0.21,
    )
    integration_order = 3
    integrator = CountingSersicIntegrator(
        model=model,
        integration_order=integration_order,
    )

    result: AdaptiveIntegrationResult = integrate_adaptively(
        make_pixel_cell(0.0, 0.0),
        integrator,
        rtol=1.0e-5,
        atol=1.0e-14,
        max_depth=6,
    )

    visited_cells = result.accepted_cells + result.refined_cells
    expected_calls = 1 + 4 * visited_cells

    assert integrator.calls == expected_calls
    assert (
        integrator.point_evaluations
        == expected_calls * integration_order**2
    )
    assert all(
        count == 1
        for count in integrator.evaluated_cells.values()
    )


def test_tighter_tolerance_requires_at_least_as_much_work():
    model = SersicModel(
        flux=1000.0,
        re=3.0,
        n=4.0,
        q=0.7,
        theta=0.3,
        x0=0.13,
        y0=-0.21,
    )
    cell = make_pixel_cell(0.0, 0.0)

    loose_integrator = CountingSersicIntegrator(
        model=model,
        integration_order=3,
    )
    tight_integrator = CountingSersicIntegrator(
        model=model,
        integration_order=3,
    )

    loose_result = integrate_adaptively(
        cell,
        loose_integrator,
        rtol=1.0e-4,
        atol=1.0e-12,
        max_depth=7,
    )
    tight_result = integrate_adaptively(
        cell,
        tight_integrator,
        rtol=1.0e-7,
        atol=1.0e-12,
        max_depth=7,
    )

    assert tight_result.maximum_level >= loose_result.maximum_level
    assert tight_integrator.calls >= loose_integrator.calls


def test_larger_maximum_depth_never_reduces_work():
    model = SersicModel(
        flux=1000.0,
        re=3.0,
        n=4.0,
        q=0.7,
        theta=0.3,
        x0=0.13,
        y0=-0.21,
    )
    cell = make_pixel_cell(0.0, 0.0)

    shallow_integrator = CountingSersicIntegrator(
        model=model,
        integration_order=3,
    )
    deep_integrator = CountingSersicIntegrator(
        model=model,
        integration_order=3,
    )

    shallow_result = integrate_adaptively(
        cell,
        shallow_integrator,
        rtol=1.0e-8,
        atol=1.0e-14,
        max_depth=3,
    )
    deep_result = integrate_adaptively(
        cell,
        deep_integrator,
        rtol=1.0e-8,
        atol=1.0e-14,
        max_depth=7,
    )

    assert deep_result.maximum_level >= shallow_result.maximum_level
    assert deep_integrator.calls >= shallow_integrator.calls
