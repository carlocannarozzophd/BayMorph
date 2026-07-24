"""Prior-volume shrinkage models for nested sampling."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .evidence import (
    expected_log_prior_volume,
    log_shell_weight,
)


class ShrinkageModel(ABC):
    """Abstract prior-volume shrinkage model."""

    @abstractmethod
    def log_prior_volume(
        self,
        *,
        iteration: int,
        nlive: int,
        current_log_prior_volume: float,
        rng: np.random.Generator,
    ) -> float:
        """Return the logarithmic remaining prior volume."""

    @abstractmethod
    def log_shell_weight(
        self,
        *,
        iteration: int,
        nlive: int,
        previous_log_prior_volume: float,
        current_log_prior_volume: float,
    ) -> float:
        """Return the logarithmic shell weight."""


class ExpectedShrinkageModel(ShrinkageModel):
    """Deterministic expected shrinkage.

    The expected shrinkage approximation is

        log(X_i) = -i / nlive

    This deterministic model is used for validation and the first
    implementation of the native nested sampler.
    """

    def log_prior_volume(
        self,
        *,
        iteration: int,
        nlive: int,
        current_log_prior_volume: float,
        rng: np.random.Generator,
    ) -> float:
        del current_log_prior_volume
        del rng

        return expected_log_prior_volume(
            iteration=iteration,
            nlive=nlive,
        )

    def log_shell_weight(
        self,
        *,
        iteration: int,
        nlive: int,
        previous_log_prior_volume: float,
        current_log_prior_volume: float,
    ) -> float:
        del previous_log_prior_volume
        del current_log_prior_volume

        return log_shell_weight(
            iteration=iteration,
            nlive=nlive,
        )
