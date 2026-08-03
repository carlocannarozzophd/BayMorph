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
