"""Mutable state and immutable records for nested sampling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DeadPoint:
    """One live point removed during nested sampling.

    Parameters
    ----------
    values : ndarray
        Physical parameter values of the removed live point.
    logl : float
        Log-likelihood of the removed live point.
    logx_previous : float
        Logarithmic prior volume before removal.
    logx_current : float
        Logarithmic prior volume after removal.
    log_weight : float
        Logarithmic prior-volume shell weight.
    log_weighted_likelihood : float
        Logarithmic evidence contribution of the dead point.
    iteration : int
        One-based nested-sampling iteration index.
    """

    values: np.ndarray
    logl: float
    logx_previous: float
    logx_current: float
    log_weight: float
    log_weighted_likelihood: float
    iteration: int

    def __post_init__(self) -> None:
        """Validate and freeze the dead-point record."""
        values = np.asarray(self.values, dtype=float)

        if values.ndim != 1:
            raise ValueError(
                "DeadPoint values must be one-dimensional."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "DeadPoint values must be finite."
            )

        if self.iteration <= 0:
            raise ValueError(
                "DeadPoint iteration must be positive."
            )

        logl = float(self.logl)
        logx_previous = float(self.logx_previous)
        logx_current = float(self.logx_current)
        log_weight = float(self.log_weight)
        log_weighted_likelihood = float(
            self.log_weighted_likelihood
        )

        logarithmic_values = (
            logl,
            logx_previous,
            logx_current,
            log_weight,
            log_weighted_likelihood,
        )

        if any(np.isnan(value) for value in logarithmic_values):
            raise ValueError(
                "DeadPoint logarithmic quantities must not be NaN."
            )

        if not logx_current < logx_previous:
            raise ValueError(
                "DeadPoint prior volume must decrease during removal."
            )

        immutable_values = values.copy()
        immutable_values.setflags(write=False)

        object.__setattr__(self, "values", immutable_values)
        object.__setattr__(self, "logl", logl)
        object.__setattr__(self, "logx_previous", logx_previous)
        object.__setattr__(self, "logx_current", logx_current)
        object.__setattr__(self, "log_weight", log_weight)
        object.__setattr__(
            self,
            "log_weighted_likelihood",
            log_weighted_likelihood,
        )


@dataclass
class NestedState:
    """Mutable execution state of a nested-sampling run.

    Parameters
    ----------
    iteration : int
        Number of completed nested-sampling iterations.
    ncall : int
        Total number of likelihood evaluations.
    initialized : bool
        Whether the initial live-point population has been generated.
    log_prior_volume : float
        Current logarithmic remaining prior volume.
    logz : float
        Current accumulated logarithmic evidence.
    dead_points : list of DeadPoint
        Dead points in removal order.
    terminated : bool
        Whether a termination criterion has been satisfied.
    finalized : bool
        Whether final live-point contributions have been committed.
    """

    iteration: int = 0
    ncall: int = 0
    initialized: bool = False
    log_prior_volume: float = 0.0
    logz: float = -np.inf
    dead_points: list[DeadPoint] = field(default_factory=list)
    terminated: bool = False
    finalized: bool = False

    def __post_init__(self) -> None:
        """Validate the initial state."""
        if self.iteration < 0:
            raise ValueError(
                "NestedState iteration must be non-negative."
            )

        if self.ncall < 0:
            raise ValueError(
                "NestedState ncall must be non-negative."
            )

        self.log_prior_volume = float(self.log_prior_volume)
        self.logz = float(self.logz)

        if np.isnan(self.log_prior_volume):
            raise ValueError(
                "NestedState log prior volume must not be NaN."
            )

        if np.isnan(self.logz):
            raise ValueError(
                "NestedState log evidence must not be NaN."
            )

        if self.log_prior_volume > 0.0:
            raise ValueError(
                "NestedState log prior volume cannot exceed zero."
            )

        if self.finalized and not self.terminated:
            raise ValueError(
                "A finalized NestedState must also be terminated."
            )

    @property
    def ndead(self) -> int:
        """Number of stored dead points."""
        return len(self.dead_points)

    def append_dead_point(
        self,
        dead_point: DeadPoint,
    ) -> None:
        """Append one dead point while preserving iteration order."""
        if not isinstance(dead_point, DeadPoint):
            raise TypeError(
                "dead_point must be a DeadPoint instance."
            )

        expected_iteration = len(self.dead_points) + 1

        if dead_point.iteration != expected_iteration:
            raise ValueError(
                "DeadPoint iteration does not match removal order: "
                f"expected {expected_iteration}, "
                f"received {dead_point.iteration}."
            )

        if self.dead_points:
            previous = self.dead_points[-1]

            if dead_point.logl < previous.logl:
                raise ValueError(
                    "Dead-point log-likelihoods must be "
                    "non-decreasing."
                )

        self.dead_points.append(dead_point)
