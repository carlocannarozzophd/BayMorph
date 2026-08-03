"""Local error estimation for adaptive numerical rendering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

from baymorph.rendering.cell import IntegrationCell

CellIntegrator = Callable[[IntegrationCell], float]


@dataclass(frozen=True, slots=True)
class AdaptiveEstimate:
    """Store one local coarse-versus-refined integration estimate."""

    coarse: float
    fine: float
    error: float
    tolerance: float
    accepted: bool


def estimate_cell(
    cell: IntegrationCell,
    integrate: CellIntegrator,
    *,
    rtol: float,
    atol: float,
) -> AdaptiveEstimate:
    """Estimate the local integration error by isotropic quadrisection.

    The coarse estimate integrates the complete cell once. The refined
    estimate is the sum of the integrals over its four children. Their
    absolute difference is used as a local error indicator.
    """
    if rtol < 0.0:
        raise ValueError("rtol must be non-negative.")

    if atol < 0.0:
        raise ValueError("atol must be non-negative.")

    coarse = float(integrate(cell))
    fine = float(sum(integrate(child) for child in cell.subdivide()))

    if not isfinite(coarse):
        raise ValueError("Coarse integration estimate must be finite.")

    if not isfinite(fine):
        raise ValueError("Refined integration estimate must be finite.")

    error = abs(fine - coarse)
    tolerance = atol + rtol * abs(fine)

    return AdaptiveEstimate(
        coarse=coarse,
        fine=fine,
        error=error,
        tolerance=tolerance,
        accepted=error <= tolerance,
    )


@dataclass(frozen=True, slots=True)
class AdaptiveIntegrationResult:
    """Store the result and diagnostics of adaptive cell integration."""

    value: float
    estimated_error: float
    accepted_cells: int
    refined_cells: int
    maximum_level: int


def integrate_adaptively(
    cell: IntegrationCell,
    integrate: CellIntegrator,
    *,
    rtol: float,
    atol: float,
    max_depth: int,
) -> AdaptiveIntegrationResult:
    """Integrate one cell through recursive isotropic refinement.

    Each cell is compared with the sum of its four children. Accepted cells
    contribute their refined estimate. Rejected cells are recursively
    subdivided until the local tolerance is satisfied or ``max_depth`` is
    reached.
    """
    if max_depth < cell.level:
        raise ValueError(
            "max_depth must be greater than or equal to the cell level."
        )

    estimate = estimate_cell(
        cell,
        integrate,
        rtol=rtol,
        atol=atol,
    )

    if estimate.accepted or cell.level >= max_depth:
        return AdaptiveIntegrationResult(
            value=estimate.fine,
            estimated_error=estimate.error,
            accepted_cells=1,
            refined_cells=0,
            maximum_level=cell.level,
        )

    children = tuple(
        integrate_adaptively(
            child,
            integrate,
            rtol=rtol,
            atol=0.25 * atol,
            max_depth=max_depth,
        )
        for child in cell.subdivide()
    )

    return AdaptiveIntegrationResult(
        value=sum(child.value for child in children),
        estimated_error=sum(
            child.estimated_error for child in children
        ),
        accepted_cells=sum(
            child.accepted_cells for child in children
        ),
        refined_cells=1
        + sum(child.refined_cells for child in children),
        maximum_level=max(
            child.maximum_level for child in children
        ),
    )
