"""Parameter routing for multi-observation inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from baymorph.fitting.parameters import Parameter, ParameterSet
from baymorph.fitting.priors import (
    FixedPrior,
    NormalPrior,
    UniformPrior,
)
from baymorph.fitting.routing import ProjectionParameterRouter

CouplingMode = Literal[
    "shared",
    "independent",
    "hierarchical_log_normal",
]


@dataclass(frozen=True)
class MultiObservationParameterSpec:
    """Define the coupling and prior structure of one model parameter."""

    name: str
    coupling: CouplingMode
    shared: Parameter | None = None
    by_observation: Mapping[str, Parameter] | None = None
    location: Parameter | None = None
    scale: Parameter | None = None

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
    def hierarchical_log_normal(
        cls,
        name: str,
        *,
        location: Parameter,
        scale: Parameter,
    ) -> MultiObservationParameterSpec:
        """Create a non-centred log-normal hierarchical specification.

        The sampled state contains one location coordinate, one non-negative
        scale coordinate, and one standard-normal latent coordinate per
        observation. Observation-level physical values are derived as

        ``exp(location + scale * latent)``.
        """
        return cls(
            name=name,
            coupling="hierarchical_log_normal",
            location=location,
            scale=scale,
        )


class MultiObservationParameterLayout:
    """Map one sampled state to complete parameters for every observation."""

    def __init__(
        self,
        observations: tuple[str, ...] | list[str],
        specifications: tuple[MultiObservationParameterSpec, ...]
        | list[MultiObservationParameterSpec] = (),
    ) -> None:
        self._observations = tuple(observations)
        self._specifications = tuple(specifications)

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
                    parameter = specification.by_observation[
                        observation
                    ]

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

            assert specification.location is not None
            assert specification.scale is not None

            global_parameters.extend(
                [
                    specification.location,
                    specification.scale,
                ]
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
            if specification.coupling
            == "hierarchical_log_normal"
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

            if specification.coupling == "independent":
                for observation in self._observations:
                    routes[observation][
                        specification.name
                    ] = self.global_name(
                        specification.name,
                        observation,
                    )
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
        """Return the standardized hierarchical latent-coordinate name."""
        return f"{name}__z__{observation}"

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
            if (
                specification.coupling
                != "hierarchical_log_normal"
            ):
                continue

            assert specification.location is not None
            assert specification.scale is not None

            location = float(
                complete[
                    specification.location.name
                ]
            )
            scale = float(
                complete[
                    specification.scale.name
                ]
            )

            if not np.isfinite(location):
                raise ValueError(
                    f"Hierarchical location for "
                    f"'{specification.name}' must be finite."
                )

            if (
                not np.isfinite(scale)
                or scale < 0.0
            ):
                raise ValueError(
                    f"Hierarchical scale for "
                    f"'{specification.name}' must be finite "
                    "and non-negative."
                )

            for observation in self._observations:
                latent = float(
                    complete[
                        self.latent_name(
                            specification.name,
                            observation,
                        )
                    ]
                )

                if not np.isfinite(latent):
                    raise ValueError(
                        f"Hierarchical latent coordinate for "
                        f"'{specification.name}' in observation "
                        f"'{observation}' must be finite."
                    )

                log_value = (
                    location
                    + scale * latent
                )

                with np.errstate(
                    over="ignore",
                    invalid="ignore",
                ):
                    value = float(
                        np.exp(log_value)
                    )

                if (
                    not np.isfinite(value)
                    or value <= 0.0
                ):
                    raise ValueError(
                        f"Derived hierarchical value for "
                        f"'{specification.name}' in observation "
                        f"'{observation}' must be finite "
                        "and positive."
                    )

                complete[
                    self.global_name(
                        specification.name,
                        observation,
                    )
                ] = value

        return complete

    def route(
        self,
        values: Mapping[str, float],
    ) -> dict[str, dict[str, float]]:
        """Route global values to complete parameters for every observation."""
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

            if (
                specification.coupling
                == "hierarchical_log_normal"
            ):
                self._validate_hierarchical_log_normal(
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
        """Validate one shared parameter specification."""
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
            specification.location is not None
            or specification.scale is not None
        ):
            raise ValueError(
                f"Shared parameter '{specification.name}' "
                "cannot define hierarchical hyperparameters."
            )

    def _validate_independent(
        self,
        specification: MultiObservationParameterSpec,
    ) -> None:
        """Validate one observation-specific parameter specification."""
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
            specification.location is not None
            or specification.scale is not None
        ):
            raise ValueError(
                f"Independent parameter '{specification.name}' "
                "cannot define hierarchical hyperparameters."
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

    def _validate_hierarchical_log_normal(
        self,
        specification: MultiObservationParameterSpec,
    ) -> None:
        """Validate one non-centred log-normal hierarchy."""
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

        if specification.location is None:
            raise ValueError(
                f"Hierarchical parameter '{specification.name}' "
                "requires a location hyperparameter."
            )

        if specification.scale is None:
            raise ValueError(
                f"Hierarchical parameter '{specification.name}' "
                "requires a scale hyperparameter."
            )

        if (
            specification.location.name
            == specification.scale.name
        ):
            raise ValueError(
                f"Hierarchical parameter '{specification.name}' "
                "requires distinct location and scale names."
            )

        scale_prior = specification.scale.prior

        if isinstance(
            scale_prior,
            FixedPrior,
        ):
            if scale_prior.value < 0.0:
                raise ValueError(
                    f"Hierarchical scale prior for "
                    f"'{specification.name}' must have "
                    "non-negative support."
                )

            return

        if isinstance(
            scale_prior,
            UniformPrior,
        ):
            if scale_prior.lower < 0.0:
                raise ValueError(
                    f"Hierarchical scale prior for "
                    f"'{specification.name}' must have "
                    "non-negative support."
                )

            return

        raise TypeError(
            f"Hierarchical scale prior for "
            f"'{specification.name}' must have known "
            "non-negative support. FixedPrior and "
            "UniformPrior are currently supported."
        )
