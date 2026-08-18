"""Prior distributions for BayMorph inference parameters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.stats import (
    beta as beta_distribution,
    norm,
)


class Prior(ABC):
    """Abstract prior distribution for one model parameter."""

    @property
    def is_fixed(self) -> bool:
        """Return whether the prior defines a deterministic value."""
        return False

    @abstractmethod
    def transform(self, unit_value: float) -> float:
        """Transform one unit-cube coordinate to physical space."""


@dataclass(frozen=True)
class FixedPrior(Prior):
    """Deterministic parameter value excluded from the sampled state."""

    value: float

    def __post_init__(self) -> None:
        value = float(self.value)

        if not np.isfinite(value):
            raise ValueError("Fixed prior value must be finite.")

        object.__setattr__(self, "value", value)

    @property
    def is_fixed(self) -> bool:
        """Return True because this prior is deterministic."""
        return True

    def transform(self, unit_value: float) -> float:
        """Return the deterministic value.

        The supplied unit-cube value is ignored because fixed parameters do
        not contribute dimensions to the sampled state.
        """
        del unit_value
        return self.value


@dataclass(frozen=True)
class UniformPrior(Prior):
    """Uniform prior over a finite interval."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)

        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("Uniform prior bounds must be finite.")

        if upper <= lower:
            raise ValueError(
                "Uniform prior upper bound must be greater than its "
                "lower bound."
            )

        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def transform(self, unit_value: float) -> float:
        """Map one unit-cube coordinate to the uniform interval."""
        value = _validate_unit_value(unit_value)

        return self.lower + value * (self.upper - self.lower)


@dataclass(frozen=True)
class LogUniformPrior(Prior):
    """Log-uniform prior over a finite strictly positive interval.

    The density is proportional to ``1 / x`` between ``lower`` and
    ``upper``. Equivalently, the natural logarithm of the parameter is
    uniformly distributed between ``log(lower)`` and ``log(upper)``.
    """

    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)

        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(
                "Log-uniform prior bounds must be finite."
            )

        if lower <= 0.0:
            raise ValueError(
                "Log-uniform prior lower bound must be positive."
            )

        if upper <= lower:
            raise ValueError(
                "Log-uniform prior upper bound must be greater than its "
                "lower bound."
            )

        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def transform(self, unit_value: float) -> float:
        """Map one unit-cube coordinate uniformly in logarithmic space."""
        value = _validate_unit_value(
            unit_value
        )

        if value == 0.0:
            return self.lower

        if value == 1.0:
            return self.upper

        log_lower = np.log(
            self.lower
        )
        log_upper = np.log(
            self.upper
        )

        result = np.exp(
            log_lower
            + value
            * (
                log_upper
                - log_lower
            )
        )

        if (
            not np.isfinite(result)
            or result <= 0.0
        ):
            raise ValueError(
                "Log-uniform prior transform produced an invalid value."
            )

        return float(result)


@dataclass(frozen=True)
class NormalPrior(Prior):
    """Unbounded normal prior."""

    mean: float
    sigma: float

    def __post_init__(self) -> None:
        mean = float(self.mean)
        sigma = float(self.sigma)

        if not np.isfinite(mean):
            raise ValueError("Normal prior mean must be finite.")

        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(
                "Normal prior sigma must be finite and positive."
            )

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "sigma", sigma)

    def transform(self, unit_value: float) -> float:
        """Map one unit-cube coordinate through the normal quantile."""
        value = _validate_unit_value(unit_value)

        lower_open = np.nextafter(0.0, 1.0)
        upper_open = np.nextafter(1.0, 0.0)
        clipped = float(
            np.clip(
                value,
                lower_open,
                upper_open,
            )
        )

        return float(
            self.mean
            + self.sigma * norm.ppf(clipped)
        )


@dataclass(frozen=True)
class BetaPrior(Prior):
    """Beta prior scaled to a finite physical interval."""

    alpha: float
    beta: float
    lower: float = 0.0
    upper: float = 1.0

    def __post_init__(self) -> None:
        alpha = float(self.alpha)
        beta = float(self.beta)
        lower = float(self.lower)
        upper = float(self.upper)

        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError(
                "Beta prior alpha must be finite and positive."
            )

        if not np.isfinite(beta) or beta <= 0.0:
            raise ValueError(
                "Beta prior beta must be finite and positive."
            )

        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(
                "Beta prior bounds must be finite."
            )

        if upper <= lower:
            raise ValueError(
                "Beta prior upper bound must be greater than its "
                "lower bound."
            )

        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def transform(self, unit_value: float) -> float:
        """Map one unit-cube coordinate through the scaled beta quantile."""
        value = _validate_unit_value(
            unit_value
        )

        quantile = float(
            beta_distribution.ppf(
                value,
                a=self.alpha,
                b=self.beta,
            )
        )

        result = (
            self.lower
            + quantile
            * (
                self.upper
                - self.lower
            )
        )

        if not np.isfinite(result):
            raise ValueError(
                "Beta prior transform produced a non-finite value."
            )

        return float(result)


def _validate_unit_value(
    unit_value: float,
) -> float:
    """Validate and return one scalar unit-cube coordinate."""
    value = float(unit_value)

    if not np.isfinite(value):
        raise ValueError("Unit-cube value must be finite.")

    if value < 0.0 or value > 1.0:
        raise ValueError(
            "Unit-cube value must lie between zero and one."
        )

    return value
