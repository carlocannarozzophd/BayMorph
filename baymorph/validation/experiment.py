"""Execution of individual scientific recovery experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from time import perf_counter
from types import MappingProxyType
from typing import Any, Generic, TypeVar

from baymorph.validation.recovery import RecoveryResult
from baymorph.validation.truth import GroundTruth


ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True)
class RecoveryFit:
    """Normalized output produced by a validation fitting callable.

    Parameters
    ----------
    estimates : mapping of str to float
        Recovered point estimates.
    intervals : mapping, optional
        Posterior intervals represented as ``(lower, upper)`` pairs.
    log_evidence, log_evidence_error : float, optional
        Bayesian evidence and its estimated uncertainty.
    likelihood_evaluations : int, optional
        Number of likelihood evaluations performed by the fitter.
    metadata : mapping, optional
        Additional fitting metadata.
    """

    estimates: Mapping[str, float]
    intervals: Mapping[str, tuple[float, float]] | None = None
    log_evidence: float | None = None
    log_evidence_error: float | None = None
    likelihood_evaluations: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_estimates: dict[str, float] = {}

        for name, value in self.estimates.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Estimate names must be non-empty strings.")

            numeric_value = float(value)
            if not isfinite(numeric_value):
                raise ValueError("Recovered estimates must be finite.")

            normalized_estimates[name] = numeric_value

        if not normalized_estimates:
            raise ValueError("At least one recovered estimate is required.")

        normalized_intervals = None
        if self.intervals is not None:
            normalized_intervals = {}

            for name, bounds in self.intervals.items():
                if len(bounds) != 2:
                    raise ValueError(
                        "Posterior intervals must contain two bounds."
                    )

                lower = float(bounds[0])
                upper = float(bounds[1])

                if not isfinite(lower) or not isfinite(upper):
                    raise ValueError("Posterior interval bounds must be finite.")

                if lower > upper:
                    raise ValueError(
                        "Posterior interval lower bounds cannot exceed "
                        "upper bounds."
                    )

                normalized_intervals[name] = (lower, upper)

        for value, label in (
            (self.log_evidence, "Log evidence"),
            (self.log_evidence_error, "Log-evidence error"),
        ):
            if value is not None and not isfinite(float(value)):
                raise ValueError(f"{label} must be finite.")

        if (
            self.log_evidence_error is not None
            and self.log_evidence_error < 0
        ):
            raise ValueError("Log-evidence error must be non-negative.")

        if (
            self.likelihood_evaluations is not None
            and self.likelihood_evaluations < 0
        ):
            raise ValueError(
                "Likelihood evaluations must be non-negative."
            )

        object.__setattr__(
            self,
            "estimates",
            MappingProxyType(normalized_estimates),
        )
        object.__setattr__(
            self,
            "intervals",
            (
                None
                if normalized_intervals is None
                else MappingProxyType(normalized_intervals)
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True)
class RecoveryExperiment(Generic[ObservationT]):
    """Run one simulation-and-recovery experiment.

    The experiment deliberately depends on callables rather than concrete
    simulation or fitting implementations. This keeps validation independent
    of a particular BayMorph model or inference backend.

    Parameters
    ----------
    case_id : str
        Stable identifier for the validation case.
    truth : GroundTruth
        Complete ground-truth description.
    simulator : callable
        Callable receiving ``truth`` and returning a synthetic observation.
    fitter : callable
        Callable receiving the synthetic observation and returning a
        :class:`RecoveryFit`.
    metadata : mapping, optional
        Additional experiment metadata.
    """

    case_id: str
    truth: GroundTruth
    simulator: Callable[[GroundTruth], ObservationT]
    fitter: Callable[[ObservationT], RecoveryFit]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("Case ID must be a non-empty string.")

        if not callable(self.simulator):
            raise TypeError("Simulator must be callable.")

        if not callable(self.fitter):
            raise TypeError("Fitter must be callable.")

        object.__setattr__(self, "case_id", self.case_id.strip())
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    def run(self) -> RecoveryResult:
        """Execute the simulation and fit, returning normalized diagnostics."""
        observation = self.simulator(self.truth)

        start = perf_counter()
        fit = self.fitter(observation)
        runtime_seconds = perf_counter() - start

        if not isinstance(fit, RecoveryFit):
            raise TypeError("Fitter must return a RecoveryFit instance.")

        missing = set(self.truth.parameters) - set(fit.estimates)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Missing recovered estimates for: {names}.")

        metadata = dict(self.metadata)
        metadata.update(fit.metadata)
        metadata.setdefault("model_name", self.truth.model_name)

        return RecoveryResult(
            case_id=self.case_id,
            truth=dict(self.truth.parameters),
            estimates=dict(fit.estimates),
            intervals=(
                None
                if fit.intervals is None
                else dict(fit.intervals)
            ),
            log_evidence=fit.log_evidence,
            log_evidence_error=fit.log_evidence_error,
            runtime_seconds=runtime_seconds,
            likelihood_evaluations=fit.likelihood_evaluations,
            metadata=metadata,
        )