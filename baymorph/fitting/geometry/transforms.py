"""Coordinate transformations used by geometry-aware proposal engines."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Transform(ABC):
    """Abstract invertible transformation into proposal space."""

    @abstractmethod
    def forward(self, value: float) -> float:
        """Transform one physical value into proposal space."""

    @abstractmethod
    def inverse(self, value: float) -> float:
        """Transform one proposal-space value into physical space."""


class IdentityTransform(Transform):
    """Identity coordinate transformation."""

    def forward(self, value: float) -> float:
        return float(value)

    def inverse(self, value: float) -> float:
        return float(value)


class LogTransform(Transform):
    """Natural-log transformation for strictly positive parameters."""

    def forward(self, value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                "LogTransform requires a positive finite physical value."
            )
        return float(np.log(value))

    def inverse(self, value: float) -> float:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(
                "LogTransform requires a finite proposal-space value."
            )
        result = float(np.exp(value))
        if not np.isfinite(result):
            raise ValueError(
                "LogTransform inverse produced a non-finite value."
            )
        return result
