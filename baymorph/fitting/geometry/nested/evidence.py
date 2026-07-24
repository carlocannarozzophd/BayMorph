"""Numerically stable evidence utilities for nested sampling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


def logdiffexp(
    first: float,
    second: float,
) -> float:
    """Return ``log(exp(first) - exp(second))``.

    Parameters
    ----------
    first : float
        Larger logarithmic quantity.
    second : float
        Smaller logarithmic quantity.

    Returns
    -------
    float
        Logarithm of the positive difference.

    Raises
    ------
    ValueError
        If ``first <= second`` or either argument is NaN.
    """
    first = float(first)
    second = float(second)

    if np.isnan(first) or np.isnan(second):
        raise ValueError(
            "logdiffexp arguments must not be NaN."
        )

    if not first > second:
        raise ValueError(
            "logdiffexp requires first > second."
        )

    return float(
        first + np.log1p(
            -np.exp(second - first)
        )
    )


def expected_log_prior_volume(
    iteration: int,
    nlive: int,
) -> float:
    """Return the deterministic expected log prior volume.

    The baseline approximation is

    ``log(X_i) = -i / nlive``.
    """
    if isinstance(iteration, bool) or not isinstance(
        iteration,
        int,
    ):
        raise TypeError(
            "Iteration must be an integer."
        )

    if iteration < 0:
        raise ValueError(
            "Iteration must be non-negative."
        )

    if isinstance(nlive, bool) or not isinstance(
        nlive,
        int,
    ):
        raise TypeError(
            "nlive must be an integer."
        )

    if nlive <= 0:
        raise ValueError(
            "nlive must be positive."
        )

    return -float(iteration) / float(nlive)


def log_shell_weight(
    iteration: int,
    nlive: int,
) -> float:
    """Return the logarithmic prior-volume shell weight."""
    if iteration <= 0:
        raise ValueError(
            "Shell weights start at iteration 1."
        )

    previous = expected_log_prior_volume(
        iteration=iteration - 1,
        nlive=nlive,
    )
    current = expected_log_prior_volume(
        iteration=iteration,
        nlive=nlive,
    )

    return logdiffexp(
        previous,
        current,
    )


def update_log_evidence(
    current_logz: float,
    log_weight: float,
    logl: float,
) -> float:
    """Return updated accumulated logarithmic evidence."""
    current_logz = float(current_logz)
    log_weight = float(log_weight)
    logl = float(logl)

    if any(
        np.isnan(value)
        for value in (
            current_logz,
            log_weight,
            logl,
        )
    ):
        raise ValueError(
            "Evidence quantities must not be NaN."
        )

    return float(
        np.logaddexp(
            current_logz,
            log_weight + logl,
        )
    )


@dataclass(frozen=True)
class EvidenceUpdate:
    """Immutable result of one evidence accumulation step.

    Parameters
    ----------
    previous_logz : float
        Evidence before the update.
    contribution : float
        Logarithmic weighted-likelihood contribution.
    new_logz : float
        Evidence after the update.
    """

    previous_logz: float
    contribution: float
    new_logz: float

    def __post_init__(self) -> None:
        """Validate evidence-update values."""
        previous_logz = float(self.previous_logz)
        contribution = float(self.contribution)
        new_logz = float(self.new_logz)

        if any(
            np.isnan(value)
            for value in (
                previous_logz,
                contribution,
                new_logz,
            )
        ):
            raise ValueError(
                "EvidenceUpdate values must not be NaN."
            )

        expected = float(
            np.logaddexp(
                previous_logz,
                contribution,
            )
        )

        if not np.isclose(
            new_logz,
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
            equal_nan=False,
        ):
            raise ValueError(
                "EvidenceUpdate new_logz is inconsistent with its "
                "previous evidence and contribution."
            )

        if new_logz < previous_logz:
            raise ValueError(
                "EvidenceUpdate must not decrease accumulated evidence."
            )

        object.__setattr__(
            self,
            "previous_logz",
            previous_logz,
        )
        object.__setattr__(
            self,
            "contribution",
            contribution,
        )
        object.__setattr__(
            self,
            "new_logz",
            new_logz,
        )


class EvidenceAccumulator(ABC):
    """Abstract evidence-accumulation strategy."""

    @abstractmethod
    def accumulate(
        self,
        *,
        current_logz: float,
        log_weight: float,
        logl: float,
    ) -> EvidenceUpdate:
        """Return one immutable evidence update."""


class LogSpaceEvidenceAccumulator(EvidenceAccumulator):
    """Numerically stable log-space evidence accumulator."""

    def accumulate(
        self,
        *,
        current_logz: float,
        log_weight: float,
        logl: float,
    ) -> EvidenceUpdate:
        contribution = float(
            log_weight + logl
        )

        new_logz = update_log_evidence(
            current_logz=current_logz,
            log_weight=log_weight,
            logl=logl,
        )

        return EvidenceUpdate(
            previous_logz=current_logz,
            contribution=contribution,
            new_logz=new_logz,
        )
