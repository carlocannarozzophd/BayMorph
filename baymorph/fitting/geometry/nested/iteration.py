"""Immutable iteration records for native nested sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..proposals import ProposalResult
from .live_points import LivePoint
from .state import DeadPoint


def _freeze_mapping(
    values: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Return an immutable shallow copy of a metadata mapping."""
    if values is None:
        return MappingProxyType({})

    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class IterationPlan:
    """Immutable deterministic plan for one nested-sampling iteration.

    Parameters
    ----------
    iteration : int
        One-based index of the iteration being planned.
    worst_index : int
        Index of the live point selected for removal.
    worst_point : LivePoint
        Live point selected for removal.
    logl_min : float
        Strict likelihood threshold for the replacement proposal.
    logx_previous : float
        Logarithmic prior volume before the iteration.
    logx_current : float
        Planned logarithmic prior volume after the iteration.
    log_weight : float
        Logarithmic prior-volume shell weight.
    """

    iteration: int
    worst_index: int
    worst_point: LivePoint
    logl_min: float
    logx_previous: float
    logx_current: float
    log_weight: float

    def __post_init__(self) -> None:
        """Validate the iteration plan."""
        if self.iteration <= 0:
            raise ValueError(
                "IterationPlan iteration must be positive."
            )

        if self.worst_index < 0:
            raise ValueError(
                "IterationPlan worst_index must be non-negative."
            )

        if not isinstance(self.worst_point, LivePoint):
            raise TypeError(
                "IterationPlan worst_point must be a LivePoint."
            )

        logl_min = float(self.logl_min)
        logx_previous = float(self.logx_previous)
        logx_current = float(self.logx_current)
        log_weight = float(self.log_weight)

        logarithmic_values = (
            logl_min,
            logx_previous,
            logx_current,
            log_weight,
        )

        if any(np.isnan(value) for value in logarithmic_values):
            raise ValueError(
                "IterationPlan logarithmic quantities must not be NaN."
            )

        if logl_min != self.worst_point.logl:
            raise ValueError(
                "IterationPlan likelihood threshold must match the "
                "worst live-point likelihood."
            )

        if not logx_current < logx_previous:
            raise ValueError(
                "IterationPlan prior volume must decrease."
            )

        object.__setattr__(self, "logl_min", logl_min)
        object.__setattr__(self, "logx_previous", logx_previous)
        object.__setattr__(self, "logx_current", logx_current)
        object.__setattr__(self, "log_weight", log_weight)


@dataclass(frozen=True)
class IterationResult:
    """Immutable transaction ready for commit to a nested run.

    Parameters
    ----------
    plan : IterationPlan
        Deterministic iteration plan.
    proposal : ProposalResult
        Accepted replacement proposal.
    dead_point : DeadPoint
        Removed live point and its evidence contribution.
    new_logz : float
        Accumulated logarithmic evidence after the iteration.
    diagnostics : mapping, optional
        Structured iteration-level diagnostics.
    """

    plan: IterationPlan
    proposal: ProposalResult
    dead_point: DeadPoint
    new_logz: float
    diagnostics: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Validate and freeze the iteration result."""
        if not isinstance(self.plan, IterationPlan):
            raise TypeError(
                "IterationResult plan must be an IterationPlan."
            )

        if not isinstance(self.proposal, ProposalResult):
            raise TypeError(
                "IterationResult proposal must be a ProposalResult."
            )

        if not isinstance(self.dead_point, DeadPoint):
            raise TypeError(
                "IterationResult dead_point must be a DeadPoint."
            )

        new_logz = float(self.new_logz)

        if np.isnan(new_logz):
            raise ValueError(
                "IterationResult new_logz must not be NaN."
            )

        if self.proposal.logl <= self.plan.logl_min:
            raise ValueError(
                "IterationResult proposal must satisfy the strict "
                "likelihood constraint."
            )

        if self.dead_point.iteration != self.plan.iteration:
            raise ValueError(
                "IterationResult dead-point iteration must match the plan."
            )

        if not np.array_equal(
            self.dead_point.values,
            self.plan.worst_point.values,
        ):
            raise ValueError(
                "IterationResult dead point must match the planned "
                "worst live point."
            )

        if self.dead_point.logl != self.plan.worst_point.logl:
            raise ValueError(
                "IterationResult dead-point likelihood must match the "
                "planned worst live point."
            )

        if self.dead_point.logx_previous != self.plan.logx_previous:
            raise ValueError(
                "IterationResult dead-point previous prior volume must "
                "match the plan."
            )

        if self.dead_point.logx_current != self.plan.logx_current:
            raise ValueError(
                "IterationResult dead-point current prior volume must "
                "match the plan."
            )

        if self.dead_point.log_weight != self.plan.log_weight:
            raise ValueError(
                "IterationResult dead-point shell weight must match the plan."
            )

        object.__setattr__(self, "new_logz", new_logz)
        object.__setattr__(
            self,
            "diagnostics",
            _freeze_mapping(self.diagnostics),
        )

    @property
    def ncall_increment(self) -> int:
        """Likelihood-call increment contributed by the proposal."""
        return self.proposal.ncall
