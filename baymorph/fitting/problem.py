"""Backend-neutral Bayesian inference problem."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class InferenceProblem(ABC):
    """Abstract Bayesian inference problem.

    Samplers interact only with this interface.
    """

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of sampled parameters."""

    @property
    @abstractmethod
    def parameter_names(self) -> list[str]:
        """Parameter names in sampling order."""

    @abstractmethod
    def prior_transform(
        self,
        cube: FloatArray,
    ) -> FloatArray:
        """Map unit-cube coordinates to physical parameters."""

    @abstractmethod
    def log_likelihood(
        self,
        theta: FloatArray,
    ) -> float:
        """Evaluate the log-likelihood."""

    @property
    def cyclic_coordinates(self) -> tuple[int, ...]:
        """Sampled coordinates whose unit-cube coordinate is cyclic.

        The prior of such a coordinate is uniform over exactly one period of a
        likelihood periodic in it, so a backend may shift the coordinate
        cyclically without changing the posterior. None by default.
        """
        return ()
