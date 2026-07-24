"""Direct constrained-prior proposal engine."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from baymorph.fitting.geometry.nested.live_points import LivePoints
from baymorph.fitting.parameters import ParameterSet

from .base import ProposalEngine, ProposalFailure, ProposalResult


class PriorProposalEngine(ProposalEngine):
    """Generate replacements through direct draws from the full prior.

    This engine is intentionally simple and statistically transparent. It is
    used to validate the nested-sampling core before introducing geometry-aware
    covariance proposals.

    Parameters
    ----------
    max_attempts : int, optional
        Maximum number of prior draws allowed for one accepted proposal.
    """

    identifier = "prior"

    def __init__(self, *, max_attempts: int = 100_000) -> None:
        if int(max_attempts) != max_attempts or int(max_attempts) <= 0:
            raise ValueError("max_attempts must be a positive integer.")
        self.max_attempts = int(max_attempts)

    def propose(
        self,
        *,
        live_points: LivePoints,
        logl_min: float,
        parameter_set: ParameterSet,
        likelihood: Callable[[np.ndarray], float],
        rng: np.random.Generator,
    ) -> ProposalResult:
        """Draw from the prior until the strict likelihood constraint is met."""
        if not callable(likelihood):
            raise TypeError("likelihood must be callable.")

        threshold = float(logl_min)
        if np.isnan(threshold):
            raise ValueError("logl_min must not be NaN.")
        if parameter_set.ndim != live_points.ndim:
            raise ValueError(
                "ParameterSet dimensionality must match the live-point geometry."
            )
        live_points.geometry.validate_parameter_names(parameter_set.names)

        for attempt in range(1, self.max_attempts + 1):
            unit_values = rng.random(parameter_set.ndim)
            physical_values = parameter_set.transform(unit_values)
            live_points.geometry.to_proposal_space(physical_values)

            logl = float(likelihood(physical_values))
            if np.isnan(logl):
                raise ValueError("Likelihood returned NaN during prior proposal.")

            if logl > threshold:
                return ProposalResult(
                    values=physical_values,
                    logl=logl,
                    ncall=attempt,
                    attempts=attempt,
                    engine=self.identifier,
                    diagnostics={
                        "acceptance_fraction": 1.0 / attempt,
                    },
                )

        raise ProposalFailure(
            "PriorProposalEngine failed to satisfy the likelihood constraint "
            f"after {self.max_attempts} attempts."
        )
