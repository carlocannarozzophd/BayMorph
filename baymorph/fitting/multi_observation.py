"""Parameter routing for multi-observation inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from baymorph.fitting.parameters import Parameter, ParameterSet
from baymorph.fitting.routing import ProjectionParameterRouter

CouplingMode = Literal["shared", "independent"]


@dataclass(frozen=True)
class MultiObservationParameterSpec:
    """Define the coupling and prior of one model parameter."""

    name: str
    coupling: CouplingMode
    shared: Parameter | None = None
    by_observation: Mapping[str, Parameter] | None = None

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
    def band_specific(
        cls,
        name: str,
        parameters: Mapping[str, Parameter],
    ) -> MultiObservationParameterSpec:
        """Create an observation-specific specification through a legacy alias."""
        return cls.independent(
            name=name,
            parameters=parameters,
        )

    @property
    def by_band(self) -> Mapping[str, Parameter] | None:
        """Return observation-specific parameters through the legacy name."""
        return self.by_observation


class MultiObservationParameterLayout:
    """Map one sampled state to complete parameters for every observation."""

    def __init__(
        self,
        observations: tuple[str, ...] | list[str] | None = None,
        specifications: tuple[MultiObservationParameterSpec, ...]
        | list[MultiObservationParameterSpec] = (),
        *,
        bands: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        if observations is None:
            observations = bands
        elif bands is not None:
            raise ValueError(
                "Specify observations or the legacy bands argument, not both."
            )

        if observations is None:
            observations = ()

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
    def bands(self) -> tuple[str, ...]:
        """Return observation identifiers through the legacy name."""
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
        """Return the unambiguous global name of an independent parameter."""
        return f"{name}__{observation}"

    def complete_global_values(
        self,
        sampled_values: Mapping[str, float],
    ) -> dict[str, float]:
        """Combine sampled and deterministic global parameter values."""
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

        supplied = set(
            specification.by_observation
        )
        expected = set(
            self._observations
        )

        if supplied != expected:
            raise ValueError(
                f"Independent parameter '{specification.name}' "
                "must define exactly the configured bands/observations."
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
