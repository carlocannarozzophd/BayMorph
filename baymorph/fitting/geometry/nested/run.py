"""Mutable runtime state for native nested sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import NestedSamplerConfig
from .iteration import IterationPlan, IterationResult
from .live_points import LivePoints
from .shrinkage import ShrinkageModel
from .state import DeadPoint


@dataclass
class NestedRun:
    """Complete mutable state of one nested-sampling execution.

    Parameters
    ----------
    config : NestedSamplerConfig
        Immutable run configuration.
    live_points : LivePoints
        Active live-point population in physical coordinates.
    iteration : int
        Number of committed active iterations.
    ncall : int
        Number of likelihood evaluations performed.
    initialized : bool
        Whether initialization has completed.
    log_prior_volume : float
        Current logarithmic remaining prior volume.
    logz : float
        Current accumulated logarithmic evidence.
    dead_points : list of DeadPoint
        Removed live points in iteration order.
    diagnostics : list of dict
        Committed iteration-level diagnostic records.
    terminated : bool
        Whether active sampling has terminated.
    finalized : bool
        Whether final live-point contributions have been committed.
    """

    config: NestedSamplerConfig
    live_points: LivePoints
    iteration: int = 0
    ncall: int = 0
    initialized: bool = False
    log_prior_volume: float = 0.0
    logz: float = -np.inf
    dead_points: list[DeadPoint] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    terminated: bool = False
    finalized: bool = False

    def __post_init__(self) -> None:
        """Validate the initial runtime state."""
        if not isinstance(self.config, NestedSamplerConfig):
            raise TypeError(
                "config must be a NestedSamplerConfig instance."
            )

        if not isinstance(self.live_points, LivePoints):
            raise TypeError(
                "live_points must be a LivePoints instance."
            )

        if self.iteration < 0:
            raise ValueError(
                "NestedRun iteration must be non-negative."
            )

        if self.ncall < 0:
            raise ValueError(
                "NestedRun ncall must be non-negative."
            )

        self.log_prior_volume = float(self.log_prior_volume)
        self.logz = float(self.logz)

        if np.isnan(self.log_prior_volume):
            raise ValueError(
                "NestedRun log prior volume must not be NaN."
            )

        if np.isnan(self.logz):
            raise ValueError(
                "NestedRun log evidence must not be NaN."
            )

        if self.log_prior_volume > 0.0:
            raise ValueError(
                "NestedRun log prior volume cannot exceed zero."
            )

        if self.finalized and not self.terminated:
            raise ValueError(
                "A finalized NestedRun must also be terminated."
            )

        if self.initialized:
            self._validate_initialized_population()

        self._validate_dead_points()

    @property
    def nlive(self) -> int:
        """Configured number of active live points."""
        return self.config.nlive

    @property
    def ndead(self) -> int:
        """Number of committed dead points."""
        return len(self.dead_points)

    @property
    def active(self) -> bool:
        """Whether active nested-sampling iterations may be performed."""
        return (
            self.initialized
            and not self.terminated
            and not self.finalized
        )

    def mark_initialized(
        self,
        *,
        ncall: int,
    ) -> None:
        """Mark initialization complete.

        Parameters
        ----------
        ncall : int
            Number of likelihood evaluations used during initialization.
        """
        if self.initialized:
            raise RuntimeError(
                "NestedRun is already initialized."
            )

        if ncall < 0:
            raise ValueError(
                "Initialization likelihood-call count must be non-negative."
            )

        self._validate_initialized_population()

        self.ncall = int(ncall)
        self.initialized = True

    def build_iteration_plan(
        self,
        *,
        shrinkage_model: ShrinkageModel,
        rng: np.random.Generator,
    ) -> IterationPlan:
        """Build the deterministic plan for the next active iteration."""
        self._require_active()

        if not isinstance(shrinkage_model, ShrinkageModel):
            raise TypeError(
                "shrinkage_model must be a ShrinkageModel instance."
            )

        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "rng must be a numpy.random.Generator instance."
            )

        next_iteration = self.iteration + 1
        worst_index = self.live_points.worst_index
        worst_point = self.live_points.worst

        next_log_prior_volume = shrinkage_model.log_prior_volume(
            iteration=next_iteration,
            nlive=self.nlive,
            current_log_prior_volume=self.log_prior_volume,
            rng=rng,
        )

        log_weight = shrinkage_model.log_shell_weight(
            iteration=next_iteration,
            nlive=self.nlive,
            previous_log_prior_volume=self.log_prior_volume,
            current_log_prior_volume=next_log_prior_volume,
        )

        return IterationPlan(
            iteration=next_iteration,
            worst_index=worst_index,
            worst_point=worst_point,
            logl_min=worst_point.logl,
            logx_previous=self.log_prior_volume,
            logx_current=next_log_prior_volume,
            log_weight=log_weight,
        )

    def commit(
        self,
        result: IterationResult,
    ) -> None:
        """Atomically commit one completed nested-sampling iteration."""
        self._require_active()

        if not isinstance(result, IterationResult):
            raise TypeError(
                "result must be an IterationResult instance."
            )

        self._validate_commit(result)

        diagnostic_record = dict(result.diagnostics)
        diagnostic_record.setdefault(
            "iteration",
            result.plan.iteration,
        )
        diagnostic_record.setdefault(
            "proposal",
            result.proposal.engine,
        )
        diagnostic_record.setdefault(
            "proposal_attempts",
            result.proposal.attempts,
        )
        diagnostic_record.setdefault(
            "proposal_ncall",
            result.proposal.ncall,
        )

        self.live_points.replace(
            result.plan.worst_index,
            result.proposal.values,
            result.proposal.logl,
        )
        self.dead_points.append(result.dead_point)
        self.log_prior_volume = result.plan.logx_current
        self.logz = result.new_logz
        self.iteration = result.plan.iteration
        self.ncall += result.ncall_increment
        self.diagnostics.append(diagnostic_record)

        self._validate_initialized_population()

    def mark_terminated(self) -> None:
        """Mark active nested sampling as terminated."""
        if not self.initialized:
            raise RuntimeError(
                "Cannot terminate an uninitialized NestedRun."
            )

        if self.finalized:
            raise RuntimeError(
                "Cannot terminate an already finalized NestedRun."
            )

        self.terminated = True

    def mark_finalized(
        self,
        *,
        logz: float,
    ) -> None:
        """Commit the final live-point evidence contribution."""
        if not self.initialized:
            raise RuntimeError(
                "Cannot finalize an uninitialized NestedRun."
            )

        if not self.terminated:
            raise RuntimeError(
                "Cannot finalize an active NestedRun."
            )

        if self.finalized:
            raise RuntimeError(
                "NestedRun has already been finalized."
            )

        logz = float(logz)

        if np.isnan(logz):
            raise ValueError(
                "Final log evidence must not be NaN."
            )

        if logz < self.logz:
            raise ValueError(
                "Final log evidence must not decrease."
            )

        self.logz = logz
        self.finalized = True

    def _validate_commit(
        self,
        result: IterationResult,
    ) -> None:
        plan = result.plan

        if plan.iteration != self.iteration + 1:
            raise ValueError(
                "IterationResult iteration does not follow the current run."
            )

        if plan.logx_previous != self.log_prior_volume:
            raise ValueError(
                "IterationResult previous prior volume does not match "
                "the current run."
            )

        if plan.worst_index != self.live_points.worst_index:
            raise ValueError(
                "IterationResult worst index no longer matches "
                "the live-point population."
            )

        current_worst = self.live_points.worst

        if not np.array_equal(
            plan.worst_point.values,
            current_worst.values,
        ):
            raise ValueError(
                "IterationResult worst point no longer matches "
                "the live-point population."
            )

        if plan.worst_point.logl != current_worst.logl:
            raise ValueError(
                "IterationResult worst likelihood no longer matches "
                "the live-point population."
            )

        if not result.proposal.logl > plan.logl_min:
            raise ValueError(
                "Replacement proposal must satisfy the strict "
                "likelihood constraint."
            )

        if not plan.logx_current < self.log_prior_volume:
            raise ValueError(
                "Committed prior volume must decrease."
            )

        if result.new_logz < self.logz:
            raise ValueError(
                "Committed evidence must not decrease."
            )

        if result.ncall_increment <= 0:
            raise ValueError(
                "Committed likelihood-call increment must be positive."
            )

        if result.dead_point.iteration != plan.iteration:
            raise ValueError(
                "Dead-point iteration does not match the iteration plan."
            )

        if self.dead_points:
            previous_dead_point = self.dead_points[-1]

            if result.dead_point.logl < previous_dead_point.logl:
                raise ValueError(
                    "Dead-point log-likelihoods must be non-decreasing."
                )

    def _validate_initialized_population(self) -> None:
        if len(self.live_points) != self.nlive:
            raise ValueError(
                "Initialized NestedRun must contain exactly "
                f"{self.nlive} live points; "
                f"received {len(self.live_points)}."
            )

    def _validate_dead_points(self) -> None:
        for index, dead_point in enumerate(
            self.dead_points,
            start=1,
        ):
            if not isinstance(dead_point, DeadPoint):
                raise TypeError(
                    "NestedRun dead_points must contain DeadPoint instances."
                )

            if dead_point.iteration != index:
                raise ValueError(
                    "NestedRun dead-point iteration order is invalid."
                )

        for previous, current in zip(
            self.dead_points,
            self.dead_points[1:],
            strict=False,
        ):
            if current.logl < previous.logl:
                raise ValueError(
                    "NestedRun dead-point likelihoods must be "
                    "non-decreasing."
                )

    def _require_active(self) -> None:
        if not self.initialized:
            raise RuntimeError(
                "NestedRun must be initialized before active iterations."
            )

        if self.terminated:
            raise RuntimeError(
                "NestedRun has already terminated."
            )

        if self.finalized:
            raise RuntimeError(
                "NestedRun has already been finalized."
            )
