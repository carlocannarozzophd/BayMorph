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


@dataclass(frozen=True, slots=True)
class AdaptiveIntegrationResult:
    """Store the result and diagnostics of adaptive cell integration."""

    value: float
    estimated_error: float
    accepted_cells: int
    refined_cells: int
    maximum_level: int


@dataclass(frozen=True, slots=True)
class _RefinedEstimate:
    """Store a local estimate together with its evaluated children."""

    estimate: AdaptiveEstimate
    children: tuple[IntegrationCell, ...]
    child_values: tuple[float, ...]


def _validate_tolerances(*, rtol: float, atol: float) -> None:
    """Validate relative and absolute integration tolerances."""
    if rtol < 0.0:
        raise ValueError("rtol must be non-negative.")

    if atol < 0.0:
        raise ValueError("atol must be non-negative.")


def _validate_coarse_estimate(value: float) -> None:
    """Require a finite coarse integration estimate."""
    if not isfinite(value):
        raise ValueError("Coarse integration estimate must be finite.")


def _validate_refined_estimate(value: float) -> None:
    """Require a finite refined integration estimate."""
    if not isfinite(value):
        raise ValueError("Refined integration estimate must be finite.")


def _refine_cell(
    cell: IntegrationCell,
    integrate: CellIntegrator,
    *,
    coarse: float,
    rtol: float,
    atol: float,
) -> _RefinedEstimate:
    """Evaluate one cell refinement without recomputing its coarse value."""
    _validate_coarse_estimate(coarse)

    children = cell.subdivide()
    child_values = tuple(float(integrate(child)) for child in children)

    fine = float(sum(child_values))
    _validate_refined_estimate(fine)

    error = abs(fine - coarse)
    tolerance = atol + rtol * abs(fine)

    return _RefinedEstimate(
        estimate=AdaptiveEstimate(
            coarse=coarse,
            fine=fine,
            error=error,
            tolerance=tolerance,
            accepted=error <= tolerance,
        ),
        children=children,
        child_values=child_values,
    )


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
    _validate_tolerances(rtol=rtol, atol=atol)

    coarse = float(integrate(cell))
    refined = _refine_cell(
        cell,
        integrate,
        coarse=coarse,
        rtol=rtol,
        atol=atol,
    )

    return refined.estimate


def _integrate_recursively(
    cell: IntegrationCell,
    integrate: CellIntegrator,
    *,
    coarse: float,
    rtol: float,
    atol: float,
    max_depth: int,
) -> AdaptiveIntegrationResult:
    """Integrate a cell while reusing its previously evaluated coarse value."""
    refined = _refine_cell(
        cell,
        integrate,
        coarse=coarse,
        rtol=rtol,
        atol=atol,
    )
    estimate = refined.estimate

    if estimate.accepted or cell.level >= max_depth:
        return AdaptiveIntegrationResult(
            value=estimate.fine,
            estimated_error=estimate.error,
            accepted_cells=1,
            refined_cells=0,
            maximum_level=cell.level,
        )

    child_results = tuple(
        _integrate_recursively(
            child,
            integrate,
            coarse=child_value,
            rtol=rtol,
            atol=0.25 * atol,
            max_depth=max_depth,
        )
        for child, child_value in zip(
            refined.children,
            refined.child_values,
            strict=True,
        )
    )

    return AdaptiveIntegrationResult(
        value=sum(result.value for result in child_results),
        estimated_error=sum(
            result.estimated_error for result in child_results
        ),
        accepted_cells=sum(
            result.accepted_cells for result in child_results
        ),
        refined_cells=1
        + sum(result.refined_cells for result in child_results),
        maximum_level=max(
            result.maximum_level for result in child_results
        ),
    )


def integrate_adaptively(
    cell: IntegrationCell,
    integrate: CellIntegrator,
    *,
    rtol: float,
    atol: float,
    max_depth: int,
) -> AdaptiveIntegrationResult:
    """Integrate one cell through recursive isotropic refinement.

    Each cell is integrated exactly once. Its value is reused as the coarse
    estimate when that cell is reached recursively. Accepted cells contribute
    their refined estimate. Rejected cells are subdivided until the local
    tolerance is satisfied or ``max_depth`` is reached.
    """
    _validate_tolerances(rtol=rtol, atol=atol)

    if max_depth < cell.level:
        raise ValueError(
            "max_depth must be greater than or equal to the cell level."
        )

    coarse = float(integrate(cell))

    return _integrate_recursively(
        cell,
        integrate,
        coarse=coarse,
        rtol=rtol,
        atol=atol,
        max_depth=max_depth,
    )
