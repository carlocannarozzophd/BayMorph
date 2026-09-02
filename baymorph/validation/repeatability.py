"""Repeated-run analysis for scientific validation experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from baymorph.fitting.sampler_factory import make_sampler
from baymorph.validation.run import (
    ValidationRun,
    execute_validation_case,
    with_validation_seed,
)

from baymorph.validation.reference import ReferenceValidationCase
from baymorph.validation.report import (
    ValidationReport,
    ValidationTolerances,
    validate_result,
)


@dataclass(frozen=True)
class RepeatabilityRecord:
    """Summary of one repeated validation run."""

    problem_name: str
    backend_name: str
    seed: int
    logz: float
    logz_error: float | None
    mean_scaled_error: float | None
    std_scaled_error: float | None
    ncall: int
    runtime_seconds: float
    passed: bool

    def summary(self) -> dict[str, Any]:
        """Return a serializable record."""
        return {
            "problem_name": self.problem_name,
            "backend_name": self.backend_name,
            "seed": self.seed,
            "logz": self.logz,
            "logz_error": self.logz_error,
            "mean_scaled_error": self.mean_scaled_error,
            "std_scaled_error": self.std_scaled_error,
            "ncall": self.ncall,
            "runtime_seconds": self.runtime_seconds,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class RepeatabilitySummary:
    """Aggregate statistics for repeated backend runs."""

    problem_name: str
    backend_name: str
    nruns: int
    pass_fraction: float
    logz_error_median: float | None
    logz_error_p95: float | None
    mean_scaled_error_median: float | None
    mean_scaled_error_p95: float | None
    std_scaled_error_median: float | None
    std_scaled_error_p95: float | None
    ncall_median: float
    runtime_median: float

    def summary(self) -> dict[str, Any]:
        """Return a serializable aggregate summary."""
        return {
            "problem_name": self.problem_name,
            "backend_name": self.backend_name,
            "nruns": self.nruns,
            "pass_fraction": self.pass_fraction,
            "logz_error_median": self.logz_error_median,
            "logz_error_p95": self.logz_error_p95,
            "mean_scaled_error_median": (
                self.mean_scaled_error_median
            ),
            "mean_scaled_error_p95": self.mean_scaled_error_p95,
            "std_scaled_error_median": (
                self.std_scaled_error_median
            ),
            "std_scaled_error_p95": self.std_scaled_error_p95,
            "ncall_median": self.ncall_median,
            "runtime_median": self.runtime_median,
        }


def run_repeatability(
    problem: ReferenceValidationCase,
    backend_name: str,
    *,
    seeds: Iterable[int],
    config: Mapping[str, Any],
    tolerances: ValidationTolerances,
) -> tuple[
    list[ValidationRun],
    list[ValidationReport],
    list[RepeatabilityRecord],
]:
    """Run one problem repeatedly with the same backend."""
    resolved_seeds = [
        int(seed)
        for seed in seeds
    ]

    if not resolved_seeds:
        raise ValueError(
            "At least one seed is required."
        )

    if len(
        set(
            resolved_seeds
        )
    ) != len(
        resolved_seeds
    ):
        raise ValueError(
            "Repeatability seeds must be unique."
        )

    results: list[
        ValidationRun
    ] = []
    reports: list[
        ValidationReport
    ] = []
    records: list[
        RepeatabilityRecord
    ] = []

    for seed in resolved_seeds:
        seeded_config = with_validation_seed(
            backend_name,
            config,
            int(
                seed
            ),
        )

        backend = make_sampler(
            seeded_config,
            problem=problem.problem,
        )

        result = execute_validation_case(
            problem,
            backend,
            seed=int(
                seed
            ),
        )
        report = validate_result(
            problem,
            result,
            tolerances=tolerances,
        )

        results.append(
            result
        )
        reports.append(
            report
        )
        records.append(
            _make_record(
                problem=problem,
                result=result,
                report=report,
                seed=seed,
            )
        )

    return (
        results,
        reports,
        records,
    )


def summarize_repeatability(
    records: Iterable[
        RepeatabilityRecord
    ],
) -> RepeatabilitySummary:
    """Aggregate repeated-run diagnostics."""
    resolved = list(
        records
    )

    if not resolved:
        raise ValueError(
            "At least one repeatability record is required."
        )

    problem_names = {
        record.problem_name
        for record in resolved
    }
    backend_names = {
        record.backend_name
        for record in resolved
    }

    if len(
        problem_names
    ) != 1:
        raise ValueError(
            "All records must refer to the same problem."
        )

    if len(
        backend_names
    ) != 1:
        raise ValueError(
            "All records must refer to the same backend."
        )

    return RepeatabilitySummary(
        problem_name=resolved[0].problem_name,
        backend_name=resolved[0].backend_name,
        nruns=len(
            resolved
        ),
        pass_fraction=float(
            np.mean(
                [
                    record.passed
                    for record in resolved
                ]
            )
        ),
        logz_error_median=_optional_percentile(
            [
                record.logz_error
                for record in resolved
            ],
            50.0,
        ),
        logz_error_p95=_optional_percentile(
            [
                record.logz_error
                for record in resolved
            ],
            95.0,
        ),
        mean_scaled_error_median=_optional_percentile(
            [
                record.mean_scaled_error
                for record in resolved
            ],
            50.0,
        ),
        mean_scaled_error_p95=_optional_percentile(
            [
                record.mean_scaled_error
                for record in resolved
            ],
            95.0,
        ),
        std_scaled_error_median=_optional_percentile(
            [
                record.std_scaled_error
                for record in resolved
            ],
            50.0,
        ),
        std_scaled_error_p95=_optional_percentile(
            [
                record.std_scaled_error
                for record in resolved
            ],
            95.0,
        ),
        ncall_median=float(
            np.median(
                [
                    record.ncall
                    for record in resolved
                ]
            )
        ),
        runtime_median=float(
            np.median(
                [
                    record.runtime_seconds
                    for record in resolved
                ]
            )
        ),
    )



def _required_logz(
    run: ValidationRun,
) -> float:
    """Return evidence required by repeatability analysis."""
    value = run.result.logz

    if value is None:
        raise ValueError(
            "Repeatability analysis requires Bayesian evidence."
        )

    return float(
        value
    )


def _required_likelihood_calls(
    run: ValidationRun,
) -> int:
    """Return likelihood calls required by repeatability analysis."""
    calls = run.likelihood_calls

    if calls is None:
        raise ValueError(
            "Repeatability analysis requires likelihood-call diagnostics."
        )

    return calls

def _make_record(
    *,
    problem: ReferenceValidationCase,
    result: ValidationRun,
    report: ValidationReport,
    seed: int,
) -> RepeatabilityRecord:
    """Convert one result and report into a compact record."""
    checks = {
        check.name: check
        for check in report.checks
    }

    logz_error = None

    if problem.reference_logz is not None:
        logz_error = abs(
            _required_logz(result)
            - problem.reference_logz
        )

    mean_check = checks.get(
        "posterior_mean"
    )
    std_check = checks.get(
        "posterior_std"
    )

    return RepeatabilityRecord(
        problem_name=problem.name,
        backend_name=result.backend_name,
        seed=seed,
        logz=float(
            _required_logz(result)
        ),
        logz_error=(
            None
            if logz_error is None
            else float(
                logz_error
            )
        ),
        mean_scaled_error=(
            None
            if mean_check is None
            else float(
                mean_check.value
            )
        ),
        std_scaled_error=(
            None
            if std_check is None
            else float(
                std_check.value
            )
        ),
        ncall=int(
            _required_likelihood_calls(result)
        ),
        runtime_seconds=float(
            result.runtime_seconds
        ),
        passed=report.passed,
    )


def _optional_percentile(
    values: Iterable[
        float | None
    ],
    percentile: float,
) -> float | None:
    """Calculate a percentile after excluding missing values."""
    present = [
        float(value)
        for value in values
        if value is not None
    ]

    if not present:
        return None

    return float(
        np.percentile(
            present,
            percentile,
        )
    )
