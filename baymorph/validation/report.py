"""Tabular summaries of completed recovery experiments."""

from __future__ import annotations

from dataclasses import dataclass

from collections.abc import Sequence

import numpy as np

from baymorph.validation.reference import ReferenceValidationCase
from baymorph.validation.run import (
    ValidationRun,
    posterior_mean,
    required_logz,
)

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


@dataclass(frozen=True)
class ValidationTolerances:
    """Acceptance thresholds for one validation experiment.

    Parameters
    ----------
    logz_absolute : float
        Maximum absolute evidence error.
    mean_sigma_fraction : float
        Maximum posterior-mean error measured in units of the
        reference marginal standard deviation.
    std_sigma_fraction : float
        Maximum posterior-standard-deviation error measured in units
        of the reference marginal standard deviation.
    """

    logz_absolute: float = 0.1
    mean_sigma_fraction: float = 0.1
    std_sigma_fraction: float = 0.1

    def __post_init__(self) -> None:
        for name, value in (
            (
                "logz_absolute",
                self.logz_absolute,
            ),
            (
                "mean_sigma_fraction",
                self.mean_sigma_fraction,
            ),
            (
                "std_sigma_fraction",
                self.std_sigma_fraction,
            ),
        ):
            if (
                not np.isfinite(
                    value
                )
                or value < 0.0
            ):
                raise ValueError(
                    f"{name} must be finite and non-negative."
                )


@dataclass(frozen=True)
class ValidationCheck:
    """Outcome of one scientific acceptance check."""

    name: str
    value: float
    tolerance: float
    passed: bool
    description: str


@dataclass(frozen=True)
class ValidationReport:
    """Scientific validation verdict for one backend result."""

    problem_name: str
    backend_name: str
    checks: tuple[ValidationCheck, ...]

    @property
    def passed(self) -> bool:
        """Return whether every acceptance check passed."""
        return all(
            check.passed
            for check in self.checks
        )

    @property
    def status(self) -> str:
        """Return a compact textual verdict."""
        return (
            "PASS"
            if self.passed
            else "FAIL"
        )

    def summary(self) -> dict:
        """Return a serializable report summary."""
        return {
            "problem_name": self.problem_name,
            "backend_name": self.backend_name,
            "status": self.status,
            "passed": self.passed,
            "checks": [
                {
                    "name": check.name,
                    "value": check.value,
                    "tolerance": check.tolerance,
                    "passed": check.passed,
                    "description": check.description,
                }
                for check in self.checks
            ],
        }


def validate_result(
    problem: ReferenceValidationCase,
    result: ValidationRun,
    *,
    tolerances: ValidationTolerances,
) -> ValidationReport:
    """Evaluate one backend result against trusted references."""
    if result.case_name != problem.name:
        raise ValueError(
            "Result and problem names must match."
        )

    checks: list[
        ValidationCheck
    ] = []

    if problem.reference_logz is not None:
        logz_error = abs(
            _required_logz(result)
            - problem.reference_logz
        )

        checks.append(
            ValidationCheck(
                name="logz",
                value=float(
                    logz_error
                ),
                tolerance=float(
                    tolerances.logz_absolute
                ),
                passed=(
                    logz_error
                    <= tolerances.logz_absolute
                ),
                description=(
                    "Absolute logarithmic-evidence error."
                ),
            )
        )

    if (
        problem.reference_mean is not None
        and problem.reference_covariance is not None
    ):
        reference_std = np.sqrt(
            np.diag(
                problem.reference_covariance
            )
        )

        mean_error = np.abs(
            _posterior_mean(result)
            - problem.reference_mean
        )
        mean_scaled_error = _maximum_scaled_error(
            mean_error,
            reference_std,
        )

        checks.append(
            ValidationCheck(
                name="posterior_mean",
                value=mean_scaled_error,
                tolerance=float(
                    tolerances.mean_sigma_fraction
                ),
                passed=(
                    mean_scaled_error
                    <= tolerances.mean_sigma_fraction
                ),
                description=(
                    "Maximum posterior-mean error in units "
                    "of reference sigma."
                ),
            )
        )

        result_std = np.sqrt(
            np.diag(
                _posterior_covariance(result)
            )
        )
        std_error = np.abs(
            result_std
            - reference_std
        )
        std_scaled_error = _maximum_scaled_error(
            std_error,
            reference_std,
        )

        checks.append(
            ValidationCheck(
                name="posterior_std",
                value=std_scaled_error,
                tolerance=float(
                    tolerances.std_sigma_fraction
                ),
                passed=(
                    std_scaled_error
                    <= tolerances.std_sigma_fraction
                ),
                description=(
                    "Maximum posterior-standard-deviation error "
                    "in units of reference sigma."
                ),
            )
        )

    if not checks:
        raise ValueError(
            "Problem does not provide references required "
            "for scientific validation."
        )

    return ValidationReport(
        problem_name=problem.name,
        backend_name=result.backend_name,
        checks=tuple(
            checks
        ),
    )


_required_logz = required_logz


_posterior_mean = posterior_mean


def _posterior_covariance(
    run: ValidationRun,
) -> np.ndarray:
    """Return the weighted posterior covariance of one validation run."""
    mean = _posterior_mean(
        run
    )
    centered = (
        run.result.samples
        - mean
    )

    return np.einsum(
        "n,ni,nj->ij",
        run.result.weights,
        centered,
        centered,
    )

def _maximum_scaled_error(
    error: np.ndarray,
    scale: np.ndarray,
) -> float:
    """Return the maximum component-wise error relative to a scale."""
    if np.any(
        scale <= 0.0
    ):
        raise ValueError(
            "Reference standard deviations must be positive."
        )

    return float(
        np.max(
            error / scale
        )
    )
