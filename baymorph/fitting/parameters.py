"""Parameter definitions for fitting."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Parameter:
    """Single fitted parameter."""

    name: str
    lower: float
    upper: float

    def transform(
        self,
        u: float,
    ) -> float:
        """Transform unit-cube value to physical value."""
        return self.lower + u * (
            self.upper - self.lower
        )


class ParameterSet:
    """Ordered collection of parameters."""

    def __init__(
        self,
        parameters: list[Parameter],
    ):
        self.parameters = parameters

    @property
    def names(
        self,
    ) -> list[str]:
        """Parameter names."""
        return [
            parameter.name
            for parameter in self.parameters
        ]

    @property
    def ndim(
        self,
    ) -> int:
        """Number of parameters."""
        return len(
            self.parameters
        )

    @property
    def bounds(
        self,
    ) -> dict[str, tuple[float, float]]:
        """Parameter bounds as dictionary."""
        return {
            parameter.name: (
                parameter.lower,
                parameter.upper,
            )
            for parameter in self.parameters
        }

    def transform(
        self,
        u,
    ) -> np.ndarray:
        """Transform unit-cube vector to physical parameter vector."""
        return np.array(
            [
                parameter.transform(value)
                for parameter, value in zip(
                    self.parameters,
                    u,
                )
            ]
        )