"""Base interfaces for geometry-aware constrained proposals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from baymorph.fitting.geometry.nested.live_points import LivePoints
from baymorph.fitting.parameters import ParameterSet


class ProposalFailure(RuntimeError):
    """Raised when a proposal engine cannot satisfy its constraint."""


@dataclass(frozen=True)
class ProposalResult:
    """Successful constrained proposal in physical parameter coordinates.

    Parameters
    ----------
    values : ndarray
        Accepted physical parameter vector.
    logl : float
        Log-likelihood of the accepted vector.
    ncall : int
        Number of likelihood evaluations used by the proposal.
    attempts : int
        Number of generated candidates, including the accepted candidate.
    engine : str
        Stable identifier of the proposal engine.
    diagnostics : mapping, optional
        Engine-specific scalar diagnostic values.
    """

    values: np.ndarray
    logl: float
    ncall: int
    attempts: int
    engine: str
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze proposal-result contents."""
        values = np.asarray(self.values, dtype=float).copy()
        if values.ndim != 1:
            raise ValueError("Proposal values must be a one-dimensional vector.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Proposal values must be finite.")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

        logl = float(self.logl)
        if np.isnan(logl):
            raise ValueError("Proposal log-likelihood must not be NaN.")
        object.__setattr__(self, "logl", logl)

        if int(self.ncall) != self.ncall or int(self.ncall) <= 0:
            raise ValueError("Proposal ncall must be a positive integer.")
        object.__setattr__(self, "ncall", int(self.ncall))

        if int(self.attempts) != self.attempts or int(self.attempts) <= 0:
            raise ValueError("Proposal attempts must be a positive integer.")
        object.__setattr__(self, "attempts", int(self.attempts))

        if self.attempts < self.ncall:
            raise ValueError("Proposal attempts cannot be smaller than ncall.")
        if not self.engine:
            raise ValueError("Proposal engine identifier cannot be empty.")

        diagnostics = {
            str(name): float(value)
            for name, value in self.diagnostics.items()
        }
        if any(not np.isfinite(value) for value in diagnostics.values()):
            raise ValueError("Proposal diagnostics must be finite scalars.")
        object.__setattr__(self, "diagnostics", diagnostics)


class ProposalEngine(ABC):
    """Interface for constrained proposal engines."""

    identifier: str

    @abstractmethod
    def propose(
        self,
        *,
        live_points: LivePoints,
        logl_min: float,
        parameter_set: ParameterSet,
        likelihood: Callable[[np.ndarray], float],
        rng: np.random.Generator,
    ) -> ProposalResult:
        """Generate one physical point satisfying ``logl > logl_min``."""
