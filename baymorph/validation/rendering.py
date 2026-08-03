"""Validation utilities for Sérsic rendering accuracy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baymorph.models.sersic import SersicModel


@dataclass(frozen=True)
class RenderingAccuracyResult:
    """Accuracy metrics relative to a high-order reference rendering."""

    integration_order: int
    l1_error: float
    l2_error: float
    linf_error: float
    flux_error: float


def validate_rendering_accuracy(
    model: SersicModel,
    *,
    shape: tuple[int, int],
    reference_order: int = 61,
    integration_orders: tuple[int, ...] = (
        3,
        5,
        7,
        9,
        11,
        13,
    ),
) -> list[RenderingAccuracyResult]:
    """Compare rendering accuracy against a high-order reference."""

    reference = model.render(
        shape=shape,
        integration_order=reference_order,
    )

    results: list[RenderingAccuracyResult] = []

    for order in integration_orders:
        image = model.render(
            shape=shape,
            integration_order=order,
        )

        difference = image - reference

        results.append(
            RenderingAccuracyResult(
                integration_order=order,
                l1_error=float(np.mean(np.abs(difference))),
                l2_error=float(np.sqrt(np.mean(difference**2))),
                linf_error=float(np.max(np.abs(difference))),
                flux_error=float(
                    abs(image.sum() - reference.sum())
                ),
            )
        )

    return results
