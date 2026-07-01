"""Parameter definitions for fitting."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Parameter:
    """Definition of a fitted model parameter.

    Parameters
    ----------
    name : str
        Parameter name.
    lower : float
        Lower bound of the uniform prior.
    upper : float
        Upper bound of the uniform prior.
    """

    name: str
    lower: float
    upper: float

    def transform(
        self,
        u: float,
    ) -> float:
        """Transform a unit-cube value to the physical parameter space.

        Parameters
        ----------
        u : float
            Value in the interval ``[0, 1]``.

        Returns
        -------
        float
            Physical parameter value obtained by a linear transformation.
        """
        return self.lower + u * (self.upper - self.lower)


class ParameterSet:
    """Ordered collection of fitted parameters.

    Parameters
    ----------
    parameters : list of Parameter
        Sequence of parameter definitions. The order determines the order of
        parameters used by the sampler.
    """

    def __init__(
        self,
        parameters: list[Parameter],
    ):
        self.parameters = parameters

    @property
    def names(
        self,
    ) -> list[str]:
        """list of str : Parameter names in sampling order."""
        return [
            parameter.name
            for parameter in self.parameters
        ]

    @property
    def ndim(
        self,
    ) -> int:
        """int : Number of fitted parameters."""
        return len(
            self.parameters
        )

    @property
    def bounds(
        self,
    ) -> dict[str, tuple[float, float]]:
        """dict : Mapping from parameter names to prior bounds."""
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
        """Transform a unit-cube parameter vector to physical parameters.

        Parameters
        ----------
        u : array-like
            Unit-cube parameter vector with one value per parameter.

        Returns
        -------
        ndarray
            Physical parameter vector in the same order as ``parameters``.
        """
        return np.array(
            [
                parameter.transform(value)
                for parameter, value in zip(
                    self.parameters,
                    u,
                )
            ]
        )