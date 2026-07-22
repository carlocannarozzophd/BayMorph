"""Containers for individual parameter-recovery experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from baymorph.validation.metrics import (
    fractional_error,
    interval_contains_truth,
    parameter_bias,
)


@dataclass(frozen=True)
class ParameterRecovery:
    """Recovery diagnostics for one fitted parameter."""

    parameter: str
    truth: float
    estimate: float
    lower: float | None = None
    upper: float | None = None

    @property
    def bias(self) -> float:
        """float : Signed difference between estimate and truth."""
        return parameter_bias(self.estimate, self.truth)

    @property
    def fractional_error(self) -> float:
        """float : Signed recovery error divided by truth."""
        return fractional_error(self.estimate, self.truth)

    @property
    def covered(self) -> bool | None:
        """bool or None : Whether the posterior interval contains the truth."""
        if self.lower is None and self.upper is None:
            return None
        if self.lower is None or self.upper is None:
            raise ValueError("Both interval bounds are required for coverage.")
        return interval_contains_truth(self.lower, self.upper, self.truth)


@dataclass(frozen=True)
class RecoveryResult:
    """Result of one simulated-galaxy recovery experiment.

    Parameters
    ----------
    case_id : str
        Stable identifier of the simulation case.
    truth : dict of str to float
        True parameter values used to generate the simulation.
    estimates : dict of str to float
        Point estimates recovered by the fit.
    intervals : dict, optional
        Mapping from parameter name to ``(lower, upper)`` posterior bounds.
    log_evidence, log_evidence_error : float, optional
        Final Bayesian evidence and its estimated uncertainty.
    runtime_seconds : float, optional
        Wall-clock runtime of the fit.
    likelihood_evaluations : int, optional
        Number of likelihood evaluations used by the fit.
    metadata : dict, optional
        Additional immutable-by-convention experiment metadata.
    """

    case_id: str
    truth: dict[str, float]
    estimates: dict[str, float]
    intervals: dict[str, tuple[float, float]] | None = None
    log_evidence: float | None = None
    log_evidence_error: float | None = None
    runtime_seconds: float | None = None
    likelihood_evaluations: int | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        missing = set(self.truth) - set(self.estimates)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Missing recovered estimates for: {names}.")
        if self.runtime_seconds is not None and self.runtime_seconds < 0:
            raise ValueError("Runtime must be non-negative.")
        if self.likelihood_evaluations is not None and self.likelihood_evaluations < 0:
            raise ValueError("Likelihood evaluations must be non-negative.")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """tuple of str : Recovered parameters in truth insertion order."""
        return tuple(self.truth)

    def parameter(self, name: str) -> ParameterRecovery:
        """Build recovery diagnostics for one parameter."""
        if name not in self.truth:
            raise KeyError(name)

        lower = None
        upper = None
        if self.intervals is not None and name in self.intervals:
            lower, upper = self.intervals[name]

        return ParameterRecovery(
            parameter=name,
            truth=float(self.truth[name]),
            estimate=float(self.estimates[name]),
            lower=lower,
            upper=upper,
        )
