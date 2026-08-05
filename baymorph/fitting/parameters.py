"""Parameter definitions for Bayesian inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baymorph.fitting.priors import (
    FixedPrior,
    Prior,
    UniformPrior,
)


@dataclass(frozen=True, init=False)
class Parameter:
    """Definition of one model parameter and its prior.

    The legacy construction

    ``Parameter(name, lower, upper)``

    remains supported and creates a :class:`UniformPrior`.
    """

    name: str
    prior: Prior

    def __init__(
        self,
        name: str,
        lower: float | None = None,
        upper: float | None = None,
        *,
        prior: Prior | None = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(
                "Parameter name must be a non-empty string."
            )

        if prior is not None:
            if lower is not None or upper is not None:
                raise ValueError(
                    "Explicit prior cannot be combined with lower or "
                    "upper bounds."
                )

            if not isinstance(prior, Prior):
                raise TypeError(
                    "prior must be a Prior instance."
                )

            resolved_prior = prior

        else:
            if lower is None or upper is None:
                raise ValueError(
                    "Parameter requires either an explicit prior or both "
                    "lower and upper bounds."
                )

            resolved_prior = UniformPrior(
                lower=lower,
                upper=upper,
            )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "prior", resolved_prior)

    @property
    def is_fixed(self) -> bool:
        """Return whether the parameter is deterministic."""
        return self.prior.is_fixed

    @property
    def fixed_value(self) -> float:
        """Return the deterministic value of a fixed parameter."""
        if not isinstance(self.prior, FixedPrior):
            raise AttributeError(
                f"Parameter '{self.name}' is not fixed."
            )

        return self.prior.value

    @property
    def lower(self) -> float:
        """Return the lower bound of a uniform prior."""
        if not isinstance(self.prior, UniformPrior):
            raise AttributeError(
                f"Parameter '{self.name}' does not have a uniform prior."
            )

        return self.prior.lower

    @property
    def upper(self) -> float:
        """Return the upper bound of a uniform prior."""
        if not isinstance(self.prior, UniformPrior):
            raise AttributeError(
                f"Parameter '{self.name}' does not have a uniform prior."
            )

        return self.prior.upper

    def transform(
        self,
        unit_value: float,
    ) -> float:
        """Transform one unit-cube coordinate through the parameter prior."""
        return self.prior.transform(unit_value)


class ParameterSet:
    """Ordered collection of model parameters.

    Fixed parameters remain part of the complete model definition but are
    excluded from the sampled state.
    """

    def __init__(
        self,
        parameters: list[Parameter],
    ) -> None:
        if not all(
            isinstance(parameter, Parameter)
            for parameter in parameters
        ):
            raise TypeError(
                "parameters must contain only Parameter instances."
            )

        names = [
            parameter.name
            for parameter in parameters
        ]

        if len(set(names)) != len(names):
            raise ValueError(
                "Parameter names must be unique."
            )

        self.all_parameters = list(parameters)
        self.parameters = [
            parameter
            for parameter in parameters
            if not parameter.is_fixed
        ]

    @property
    def names(self) -> list[str]:
        """Return sampled parameter names in sampling order."""
        return [
            parameter.name
            for parameter in self.parameters
        ]

    @property
    def all_names(self) -> list[str]:
        """Return all model parameter names in model order."""
        return [
            parameter.name
            for parameter in self.all_parameters
        ]

    @property
    def ndim(self) -> int:
        """Return the number of sampled parameters."""
        return len(self.parameters)

    @property
    def fixed_values(self) -> dict[str, float]:
        """Return deterministic parameter values."""
        return {
            parameter.name: parameter.fixed_value
            for parameter in self.all_parameters
            if parameter.is_fixed
        }

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        """Return bounds for sampled uniform-prior parameters.

        Raises
        ------
        TypeError
            If a sampled parameter does not have a uniform prior.
        """
        bounds: dict[str, tuple[float, float]] = {}

        for parameter in self.parameters:
            if not isinstance(
                parameter.prior,
                UniformPrior,
            ):
                raise TypeError(
                    "Finite bounds are unavailable for parameter "
                    f"'{parameter.name}' with prior "
                    f"'{type(parameter.prior).__name__}'."
                )

            bounds[parameter.name] = (
                parameter.prior.lower,
                parameter.prior.upper,
            )

        return bounds

    def transform(
        self,
        unit_values,
    ) -> np.ndarray:
        """Transform one sampled unit-cube vector to physical space."""
        values = np.asarray(
            unit_values,
            dtype=float,
        )

        if values.shape != (self.ndim,):
            raise ValueError(
                "Unit-cube vector must contain one value per sampled "
                "parameter."
            )

        return np.asarray(
            [
                parameter.transform(value)
                for parameter, value in zip(
                    self.parameters,
                    values,
                    strict=True,
                )
            ],
            dtype=float,
        )

    def complete_values(
        self,
        sampled_values,
    ) -> dict[str, float]:
        """Reconstruct all model parameters from one sampled vector."""
        values = np.asarray(
            sampled_values,
            dtype=float,
        )

        if values.shape != (self.ndim,):
            raise ValueError(
                "Sampled vector must contain one value per sampled "
                "parameter."
            )

        sampled = {
            parameter.name: float(value)
            for parameter, value in zip(
                self.parameters,
                values,
                strict=True,
            )
        }

        return {
            parameter.name: (
                parameter.fixed_value
                if parameter.is_fixed
                else sampled[parameter.name]
            )
            for parameter in self.all_parameters
        }
