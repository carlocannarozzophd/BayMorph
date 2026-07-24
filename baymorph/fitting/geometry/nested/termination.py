"""Termination criteria for native nested sampling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TerminationDecision:
    """Result returned by a termination criterion.

    Parameters
    ----------
    should_stop : bool
        Whether sampling should terminate.
    reason : str
        Human-readable termination reason.
    remaining_logz : float or None
        Estimated remaining evidence.
    delta_logz : float or None
        Estimated remaining evidence contribution.
    """

    should_stop: bool
    reason: str = ""
    remaining_logz: float | None = None
    delta_logz: float | None = None


class TerminationCriterion(ABC):
    """Abstract nested-sampling termination criterion."""

    @abstractmethod
    def evaluate(
        self,
        *,
        iteration: int,
        ncall: int,
        logz: float,
        logx: float,
        live_logl: np.ndarray,
    ) -> TerminationDecision:
        """Evaluate whether sampling should terminate."""


class MaximumIterationTerminationCriterion(
    TerminationCriterion,
):
    """Terminate after a fixed number of iterations."""

    def __init__(
        self,
        max_iterations: int,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError(
                "max_iterations must be positive."
            )

        self.max_iterations = int(max_iterations)

    def evaluate(
        self,
        *,
        iteration: int,
        ncall: int,
        logz: float,
        logx: float,
        live_logl: np.ndarray,
    ) -> TerminationDecision:
        del ncall
        del logz
        del logx
        del live_logl

        if iteration >= self.max_iterations:
            return TerminationDecision(
                should_stop=True,
                reason="maximum iterations reached",
            )

        return TerminationDecision(
            should_stop=False,
        )


class MaximumLikelihoodCallTerminationCriterion(
    TerminationCriterion,
):
    """Terminate after a fixed number of likelihood evaluations."""

    def __init__(
        self,
        max_calls: int,
    ) -> None:
        if max_calls <= 0:
            raise ValueError(
                "max_calls must be positive."
            )

        self.max_calls = int(max_calls)

    def evaluate(
        self,
        *,
        iteration: int,
        ncall: int,
        logz: float,
        logx: float,
        live_logl: np.ndarray,
    ) -> TerminationDecision:
        del iteration
        del logz
        del logx
        del live_logl

        if ncall >= self.max_calls:
            return TerminationDecision(
                should_stop=True,
                reason="maximum likelihood calls reached",
            )

        return TerminationDecision(
            should_stop=False,
        )


class EvidenceTerminationCriterion(
    TerminationCriterion,
):
    """Terminate using the remaining-evidence estimate."""

    def __init__(
        self,
        dlogz: float,
    ) -> None:
        dlogz = float(dlogz)

        if not np.isfinite(dlogz) or dlogz <= 0.0:
            raise ValueError(
                "dlogz must be positive."
            )

        self.dlogz = dlogz

    def evaluate(
        self,
        *,
        iteration: int,
        ncall: int,
        logz: float,
        logx: float,
        live_logl: np.ndarray,
    ) -> TerminationDecision:
        del iteration
        del ncall

        if live_logl.size == 0:
            return TerminationDecision(
                should_stop=True,
                reason="empty live-point population",
            )

        remaining_logz = (
            logx
            + float(np.max(live_logl))
        )

        if np.isneginf(logz):
            return TerminationDecision(
                should_stop=False,
                remaining_logz=remaining_logz,
                delta_logz=np.inf,
            )

        delta = float(
            np.logaddexp(
                logz,
                remaining_logz,
            ) - logz
        )

        return TerminationDecision(
            should_stop=delta < self.dlogz,
            reason="evidence tolerance reached",
            remaining_logz=remaining_logz,
            delta_logz=delta,
        )


class AnyTerminationCriterion(
    TerminationCriterion,
):
    """Terminate when any contained criterion requests termination."""

    def __init__(
        self,
        *criteria: TerminationCriterion,
    ) -> None:
        self.criteria = tuple(criteria)

    def evaluate(
        self,
        **kwargs,
    ) -> TerminationDecision:
        for criterion in self.criteria:
            decision = criterion.evaluate(
                **kwargs,
            )

            if decision.should_stop:
                return decision

        return TerminationDecision(
            should_stop=False,
        )
