"""Reference problems and trusted targets for scientific validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from baymorph.fitting.parameters import ParameterSet
from baymorph.fitting.problem import InferenceProblem


class CallableInferenceProblem(InferenceProblem):
    """Inference problem defined by parameters and a likelihood callable.

    This concrete implementation is intended for controlled mathematical
    validation problems whose likelihood does not require an astronomical
    observation model.
    """

    def __init__(
        self,
        *,
        parameters: ParameterSet,
        log_likelihood: Callable[[np.ndarray], float],
    ) -> None:
        if not isinstance(
            parameters,
            ParameterSet,
        ):
            raise TypeError(
                "parameters must be a ParameterSet instance."
            )

        if not callable(
            log_likelihood
        ):
            raise TypeError(
                "log_likelihood must be callable."
            )

        self.parameters = parameters
        self._log_likelihood = log_likelihood

    @property
    def ndim(self) -> int:
        """Return the number of sampled parameters."""
        return self.parameters.ndim

    @property
    def parameter_names(self) -> list[str]:
        """Return sampled parameter names in sampling order."""
        return self.parameters.names

    def prior_transform(
        self,
        cube: np.ndarray,
    ) -> np.ndarray:
        """Transform one unit-cube vector to physical parameters."""
        return self.parameters.transform(
            cube
        )

    def log_likelihood(
        self,
        theta: np.ndarray,
    ) -> float:
        """Evaluate the controlled validation likelihood."""
        return float(
            self._log_likelihood(
                np.asarray(
                    theta,
                    dtype=float,
                )
            )
        )


@dataclass(frozen=True)
class ReferenceValidationCase:
    """Pair one inference problem with trusted validation references.

    The inference problem is the exact object executed by production
    inference backends. Reference quantities are external scientific targets
    used only to evaluate posterior and evidence accuracy.
    """

    name: str
    problem: InferenceProblem
    reference_logz: float | None = None
    reference_mean: np.ndarray | None = None
    reference_covariance: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        name = str(
            self.name
        ).strip()

        if not name:
            raise ValueError(
                "Validation case name must not be empty."
            )

        if not isinstance(
            self.problem,
            InferenceProblem,
        ):
            raise TypeError(
                "problem must be an InferenceProblem instance."
            )

        reference_logz = self.reference_logz

        if (
            reference_logz is not None
            and not np.isfinite(
                reference_logz
            )
        ):
            raise ValueError(
                "reference_logz must be finite when provided."
            )

        reference_mean = (
            None
            if self.reference_mean is None
            else np.asarray(
                self.reference_mean,
                dtype=float,
            )
        )

        reference_covariance = (
            None
            if self.reference_covariance is None
            else np.asarray(
                self.reference_covariance,
                dtype=float,
            )
        )

        if (
            reference_mean is None
        ) != (
            reference_covariance is None
        ):
            raise ValueError(
                "reference_mean and reference_covariance must be "
                "provided together."
            )

        if reference_mean is not None:
            expected_mean_shape = (
                self.problem.ndim,
            )
            expected_covariance_shape = (
                self.problem.ndim,
                self.problem.ndim,
            )

            if reference_mean.shape != expected_mean_shape:
                raise ValueError(
                    "reference_mean shape must match problem dimensionality."
                )

            if (
                reference_covariance.shape
                != expected_covariance_shape
            ):
                raise ValueError(
                    "reference_covariance shape must match problem "
                    "dimensionality."
                )

            if not np.all(
                np.isfinite(
                    reference_mean
                )
            ):
                raise ValueError(
                    "reference_mean must contain only finite values."
                )

            if not np.all(
                np.isfinite(
                    reference_covariance
                )
            ):
                raise ValueError(
                    "reference_covariance must contain only finite values."
                )

            if not np.allclose(
                reference_covariance,
                reference_covariance.T,
            ):
                raise ValueError(
                    "reference_covariance must be symmetric."
                )

        object.__setattr__(
            self,
            "name",
            name,
        )
        object.__setattr__(
            self,
            "reference_mean",
            (
                None
                if reference_mean is None
                else reference_mean.copy()
            ),
        )
        object.__setattr__(
            self,
            "reference_covariance",
            (
                None
                if reference_covariance is None
                else reference_covariance.copy()
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(
                    self.metadata
                )
            ),
        )
