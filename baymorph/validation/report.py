"""Tabular summaries of completed recovery experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from baymorph.validation.metrics import bias, coverage_fraction, scatter
from baymorph.validation.recovery import RecoveryResult


def summarize_recovery(
    results: Sequence[RecoveryResult],
) -> dict[str, dict[str, float | int]]:
    """Summarize recovery quality independently for each common parameter.

    Only parameters present in every result are summarized. Coverage is
    included when all results provide an interval for that parameter.

    Raises
    ------
    ValueError
        If no recovery results are provided.
    """
    if not results:
        raise ValueError("At least one recovery result is required.")

    common = set(results[0].parameter_names)
    for result in results[1:]:
        common &= set(result.parameter_names)

    summary: dict[str, dict[str, float | int]] = {}
    for name in results[0].parameter_names:
        if name not in common:
            continue

        recoveries = [result.parameter(name) for result in results]
        estimates = np.array([item.estimate for item in recoveries])
        truths = np.array([item.truth for item in recoveries])

        parameter_summary: dict[str, float | int] = {
            "count": len(recoveries),
            "bias": bias(estimates, truths),
            "scatter": scatter(estimates, truths),
        }

        covered = [item.covered for item in recoveries]
        if all(value is not None for value in covered):
            parameter_summary["coverage"] = coverage_fraction(
                np.array(covered, dtype=bool)
            )

        summary[name] = parameter_summary

    return summary
