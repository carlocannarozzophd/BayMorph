"""Posterior calibration diagnostics for repeated validation experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from baymorph.results import InferenceResult
from baymorph.validation.metrics import (
    interval_contains_truth,
    weighted_credible_interval,
    weighted_posterior_rank,
)


@dataclass(frozen=True)
class ParameterCalibration:
    """Calibration diagnostics for one parameter in one posterior."""

    parameter: str
    truth: float
    posterior_mean: float
    posterior_median: float
    posterior_rank: float
    interval_68: tuple[float, float]
    interval_95: tuple[float, float]

    @property
    def mean_error(self) -> float:
        """Return posterior-mean signed error."""
        return float(self.posterior_mean - self.truth)

    @property
    def median_error(self) -> float:
        """Return posterior-median signed error."""
        return float(self.posterior_median - self.truth)

    @property
    def covered_68(self) -> bool:
        """Return whether the 68% interval contains the truth."""
        return interval_contains_truth(
            *self.interval_68,
            self.truth,
        )

    @property
    def covered_95(self) -> bool:
        """Return whether the 95% interval contains the truth."""
        return interval_contains_truth(
            *self.interval_95,
            self.truth,
        )


@dataclass(frozen=True)
class CalibrationCase:
    """Posterior calibration diagnostics for one validation case."""

    case_id: str
    parameters: Mapping[str, ParameterCalibration]


def calibration_from_inference_result(
    *,
    case_id: str,
    result: InferenceResult,
    truth: Mapping[str, float],
) -> CalibrationCase:
    """Build posterior calibration diagnostics from one inference result."""
    posterior = result.posterior
    names = posterior.parameter_names

    missing = set(truth).difference(names)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(
            "Posterior is missing truth parameters: "
            f"{missing_names}."
        )

    weights = np.asarray(
        posterior.weights,
        dtype=float,
    )

    parameters: dict[str, ParameterCalibration] = {}

    for name, truth_value in truth.items():
        column = names.index(name)
        samples = np.asarray(
            posterior.samples[:, column],
            dtype=float,
        )

        posterior_mean = float(
            np.sum(weights * samples)
        )

        order = np.argsort(samples)
        ordered_samples = samples[order]
        ordered_weights = weights[order]
        cumulative = np.cumsum(ordered_weights)

        posterior_median = float(
            np.interp(
                0.5,
                cumulative,
                ordered_samples,
            )
        )

        parameters[name] = ParameterCalibration(
            parameter=name,
            truth=float(truth_value),
            posterior_mean=posterior_mean,
            posterior_median=posterior_median,
            posterior_rank=weighted_posterior_rank(
                samples,
                weights,
                float(truth_value),
            ),
            interval_68=weighted_credible_interval(
                samples,
                weights,
                probability=0.68,
            ),
            interval_95=weighted_credible_interval(
                samples,
                weights,
                probability=0.95,
            ),
        )

    return CalibrationCase(
        case_id=str(case_id),
        parameters=parameters,
    )


def load_calibration_case(
    *,
    case_id: str,
    result_path: str | Path,
    truth: Mapping[str, float],
) -> CalibrationCase:
    """Load one inference result and compute calibration diagnostics."""
    return calibration_from_inference_result(
        case_id=case_id,
        result=InferenceResult.load(result_path),
        truth=truth,
    )


def summarize_calibration(
    cases: Sequence[CalibrationCase],
) -> dict[str, dict[str, float | int]]:
    """Summarize calibration diagnostics across repeated cases."""
    if not cases:
        raise ValueError(
            "At least one calibration case is required."
        )

    common = set(cases[0].parameters)
    for case in cases[1:]:
        common &= set(case.parameters)

    summary: dict[str, dict[str, float | int]] = {}

    for name in cases[0].parameters:
        if name not in common:
            continue

        values = [
            case.parameters[name]
            for case in cases
        ]

        mean_errors = np.asarray(
            [value.mean_error for value in values],
            dtype=float,
        )
        median_errors = np.asarray(
            [value.median_error for value in values],
            dtype=float,
        )
        ranks = np.asarray(
            [value.posterior_rank for value in values],
            dtype=float,
        )

        summary[name] = {
            "count": len(values),
            "mean_error_mean": float(
                np.mean(mean_errors)
            ),
            "median_error_mean": float(
                np.mean(median_errors)
            ),
            "median_error_scatter": float(
                np.std(
                    median_errors,
                    ddof=1,
                )
                if len(values) > 1
                else 0.0
            ),
            "coverage_68": float(
                np.mean(
                    [
                        value.covered_68
                        for value in values
                    ]
                )
            ),
            "coverage_95": float(
                np.mean(
                    [
                        value.covered_95
                        for value in values
                    ]
                )
            ),
            "rank_mean": float(np.mean(ranks)),
            "rank_min": float(np.min(ranks)),
            "rank_max": float(np.max(ranks)),
        }

    return summary
