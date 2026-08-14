"""Parameter routing for multi-observation inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from baymorph.fitting.hierarchical import (
    physical_prior_copula_values,
)
from baymorph.fitting.parameters import (
    Parameter,
    ParameterSet,
)
from baymorph.fitting.priors import (
    FixedPrior,
    NormalPrior,
    Prior,
    UniformPrior,
)
from baymorph.fitting.routing import (
    ProjectionParameterRouter,
)


CouplingMode = Literal[
    "shared",
    "independent",
    "hierarchical",
]


@dataclass(frozen=True)
class MultiObservationParameterSpec:
    """Define the cross-observation coupling of one physical parameter."""

    name: str
    coupling: CouplingMode
    shared: Parameter | None = None
    by_observation: Mapping[str, Parameter] | None = None
    physical_prior: Prior | None = None
    correlation_prior: Prior | None = None

    @classmethod
    def shared_parameter(
        cls,
        parameter: Parameter,
    ) -> MultiObservationParameterSpec:
        """Create an exactly shared parameter specification."""
        return cls(
            name=parameter.name,
            coupling="shared",
            shared=parameter,
        )

    @classmethod
    def independent(
        cls,
        name: str,
        parameters: Mapping[str, Parameter],
    ) -> MultiObservationParameterSpec:
        """Create an observation-specific parameter specification."""
        return cls(
            name=name,
            coupling="independent",
            by_observation=dict(parameters),
        )

    @classmethod
    def observation_specific(
        cls,
        name: str,
        parameters: Mapping[str, Parameter],
    ) -> MultiObservationParameterSpec:
        """Create an observation-specific parameter specification."""
        return cls.independent(
            name=name,
            parameters=parameters,
        )

    @classmethod
    def hierarchical(
        cls,
        name: str,
        *,
        prior: Prior,
        correlation_prior: Prior | None = None,
    ) -> MultiObservationParameterSpec:
        """Create physical-prior-preserving hierarchical coupling.

        The physical marginal prior is supplied directly by ``prior``.

        Cross-observation dependence is represented by an equicorrelated
        Gaussian copula. Unless overridden internally or through an advanced
        interface, the latent correlation uses a Uniform(0, 1) reference
        prior.
        """
        if correlation_prior is None:
            correlation_prior = UniformPrior(
                lower=0.0,
                upper=1.0,
            )

        return cls(
            name=name,
            coupling="hierarchical",
            physical_prior=prior,
            correlation_prior=correlation_prior,
        )


class MultiObservationParameterLayout:
    """Map one sampled state to physical parameters for every observation."""

    def __init__(
        self,
        observations: tuple[str, ...] | list[str],
        specifications: tuple[MultiObservationParameterSpec, ...]
        | list[MultiObservationParameterSpec] = (),
    ) -> None:
        self._observations = tuple(
            observations
        )
        self._specifications = tuple(
            specifications
        )

        self._validate()

        global_parameters: list[Parameter] = []

        for specification in self._specifications:
            if specification.coupling == "shared":
                assert specification.shared is not None

                global_parameters.append(
                    specification.shared
                )
                continue

            if specification.coupling == "independent":
                assert specification.by_observation is not None

                for observation in self._observations:
                    parameter = (
                        specification.by_observation[
                            observation
                        ]
                    )

                    global_parameters.append(
                        Parameter(
                            name=self.global_name(
                                specification.name,
                                observation,
                            ),
                            prior=parameter.prior,
                        )
                    )

                continue

            assert specification.coupling == "hierarchical"
            assert specification.physical_prior is not None
            assert specification.correlation_prior is not None

            global_parameters.append(
                Parameter(
                    name=self.correlation_name(
                        specification.name
                    ),
                    prior=specification.correlation_prior,
                )
            )

            for observation in self._observations:
                global_parameters.append(
                    Parameter(
                        name=self.latent_name(
                            specification.name,
                            observation,
                        ),
                        prior=NormalPrior(
                            mean=0.0,
                            sigma=1.0,
                        ),
                    )
                )

        derived_names = {
            self.global_name(
                specification.name,
                observation,
            )
            for specification in self._specifications
            if specification.coupling == "hierarchical"
            for observation in self._observations
        }

        sampled_or_fixed_names = {
            parameter.name
            for parameter in global_parameters
        }

        collisions = (
            derived_names
            & sampled_or_fixed_names
        )

        if collisions:
            names = ", ".join(
                sorted(collisions)
            )
            raise ValueError(
                "Hierarchical derived parameter names collide with "
                f"sampled global parameters: {names}."
            )

        self._parameter_set = ParameterSet(
            global_parameters
        )

        routes = {
            observation: {}
            for observation in self._observations
        }

        for specification in self._specifications:
            if specification.coupling == "shared":
                for observation in self._observations:
                    routes[observation][
                        specification.name
                    ] = specification.name

                continue

            for observation in self._observations:
                routes[observation][
                    specification.name
                ] = self.global_name(
                    specification.name,
                    observation,
                )

        self._router = ProjectionParameterRouter(
            observations=self._observations,
            routes=routes,
        )

    @property
    def observations(self) -> tuple[str, ...]:
        """Return observation identifiers in deterministic routing order."""
        return self._observations

    @property
    def specifications(
        self,
    ) -> tuple[MultiObservationParameterSpec, ...]:
        """Return parameter specifications in model order."""
        return self._specifications

    @property
    def parameter_set(self) -> ParameterSet:
        """Return the global parameter set consumed by the sampler."""
        return self._parameter_set

    @property
    def ndim(self) -> int:
        """Return the number of sampled global parameters."""
        return self._parameter_set.ndim

    @staticmethod
    def global_name(
        name: str,
        observation: str,
    ) -> str:
        """Return an observation-specific global physical parameter name."""
        return f"{name}__{observation}"

    @staticmethod
    def latent_name(
        name: str,
        observation: str,
    ) -> str:
        """Return an independent standard-normal latent-coordinate name."""
        return f"{name}__z__{observation}"

    @staticmethod
    def correlation_name(
        name: str,
    ) -> str:
        """Return the latent Gaussian correlation parameter name."""
        return f"{name}__rho"

    def complete_global_values(
        self,
        sampled_values: Mapping[str, float],
    ) -> dict[str, float]:
        """Combine sampled, fixed, and deterministic global values."""
        complete = dict(
            self._parameter_set.fixed_values
        )
        complete.update(
            {
                name: float(value)
                for name, value in sampled_values.items()
            }
        )

        missing = set(
            self._parameter_set.all_names
        ).difference(
            complete
        )

        if missing:
            missing_names = ", ".join(
                sorted(missing)
            )
            raise KeyError(
                "Missing global parameter values: "
                f"{missing_names}."
            )

        for specification in self._specifications:
            if specification.coupling != "hierarchical":
                continue

            assert specification.physical_prior is not None

            correlation = float(
                complete[
                    self.correlation_name(
                        specification.name
                    )
                ]
            )

            latent_values = np.asarray(
                [
                    complete[
                        self.latent_name(
                            specification.name,
                            observation,
                        )
                    ]
                    for observation in self._observations
                ],
                dtype=float,
            )

            physical_values = (
                physical_prior_copula_values(
                    specification.physical_prior,
                    latent_values,
                    correlation,
                )
            )

            for observation, value in zip(
                self._observations,
                physical_values,
                strict=True,
            ):
                complete[
                    self.global_name(
                        specification.name,
                        observation,
                    )
                ] = float(value)

        return complete

    def route(
        self,
        values: Mapping[str, float],
    ) -> dict[str, dict[str, float]]:
        """Route global values to physical parameters for every observation."""
        complete = self.complete_global_values(
            values
        )

        return self._router.route(
            complete
        )

    def _validate(self) -> None:
        if not self._observations:
            raise ValueError(
                "Multi-observation parameter layout requires "
                "at least one observation."
            )

        if (
            len(set(self._observations))
            != len(self._observations)
        ):
            raise ValueError(
                "Observation names must be unique."
            )

        if any(
            not isinstance(observation, str)
            or not observation
            for observation in self._observations
        ):
            raise ValueError(
                "Observation names must be non-empty strings."
            )

        names = [
            specification.name
            for specification in self._specifications
        ]

        if len(set(names)) != len(names):
            raise ValueError(
                "Parameter names must be unique."
            )

        for specification in self._specifications:
            if specification.coupling == "shared":
                self._validate_shared(
                    specification
                )
                continue

            if specification.coupling == "independent":
                self._validate_independent(
                    specification
                )
                continue

            if specification.coupling == "hierarchical":
                self._validate_hierarchical(
                    specification
                )
                continue

            raise ValueError(
                "Unsupported coupling mode "
                f"'{specification.coupling}'."
            )

    def _validate_shared(
        self,
        specification: MultiObservationParameterSpec,
    ) -> None:
        """Validate one exactly shared parameter."""
        if specification.shared is None:
            raise ValueError(
                f"Shared parameter '{specification.name}' "
                "requires one Parameter definition."
            )

        if specification.shared.name != specification.name:
            raise ValueError(
                "Shared parameter name does not match its specification."
            )

        if specification.by_observation is not None:
            raise ValueError(
                f"Shared parameter '{specification.name}' "
                "cannot define independent parameters."
            )

        if (
            specification.physical_prior is not None
            or specification.correlation_prior is not None
        ):
            raise ValueError(
                f"Shared parameter '{specification.name}' "
                "cannot define hierarchical priors."
            )

    def _validate_independent(
        self,
        specification: MultiObservationParameterSpec,
    ) -> None:
        """Validate one observation-specific parameter."""
        if specification.shared is not None:
            raise ValueError(
                f"Independent parameter '{specification.name}' "
                "cannot define a shared parameter."
            )

        if specification.by_observation is None:
            raise ValueError(
                f"Independent parameter '{specification.name}' "
                "requires one definition per observation."
            )

        if (
            specification.physical_prior is not None
            or specification.correlation_prior is not None
        ):
            raise ValueError(
                f"Independent parameter '{specification.name}' "
                "cannot define hierarchical priors."
            )

        supplied = set(
            specification.by_observation
        )
        expected = set(
            self._observations
        )

        if supplied != expected:
            raise ValueError(
                f"Independent parameter '{specification.name}' "
                "must define exactly the configured observations."
            )

        for observation, parameter in (
            specification.by_observation.items()
        ):
            if parameter.name != specification.name:
                raise ValueError(
                    f"Parameter name for observation '{observation}' "
                    f"does not match specification "
                    f"'{specification.name}'."
                )

    def _validate_hierarchical(
        self,
        specification: MultiObservationParameterSpec,
    ) -> None:
        """Validate physical-prior-preserving hierarchical coupling."""
        if len(self._observations) < 2:
            raise ValueError(
                f"Hierarchical parameter '{specification.name}' "
                "requires at least two observations."
            )

        if specification.shared is not None:
            raise ValueError(
                f"Hierarchical parameter '{specification.name}' "
                "cannot define a shared parameter."
            )

        if specification.by_observation is not None:
            raise ValueError(
                f"Hierarchical parameter '{specification.name}' "
                "cannot define independent parameters."
            )

        if specification.physical_prior is None:
            raise ValueError(
                f"Hierarchical parameter '{specification.name}' "
                "requires a physical prior."
            )

        if not isinstance(
            specification.physical_prior,
            Prior,
        ):
            raise TypeError(
                f"Physical prior for hierarchical parameter "
                f"'{specification.name}' must be a Prior."
            )

        if specification.physical_prior.is_fixed:
            raise ValueError(
                f"Hierarchical parameter '{specification.name}' "
                "cannot use a fixed physical prior."
            )

        correlation_prior = (
            specification.correlation_prior
        )

        if correlation_prior is None:
            raise ValueError(
                f"Hierarchical parameter '{specification.name}' "
                "requires a latent correlation prior."
            )

        if isinstance(
            correlation_prior,
            FixedPrior,
        ):
            if (
                correlation_prior.value < 0.0
                or correlation_prior.value > 1.0
            ):
                raise ValueError(
                    f"Hierarchical correlation prior for "
                    f"'{specification.name}' must lie in [0, 1]."
                )

            return

        if isinstance(
            correlation_prior,
            UniformPrior,
        ):
            if (
                correlation_prior.lower < 0.0
                or correlation_prior.upper > 1.0
            ):
                raise ValueError(
                    f"Hierarchical correlation prior for "
                    f"'{specification.name}' must have support "
                    "inside [0, 1]."
                )

            return

        raise TypeError(
            f"Hierarchical correlation prior for "
            f"'{specification.name}' must have known support "
            "inside [0, 1]. FixedPrior and UniformPrior are "
            "currently supported."
        )
