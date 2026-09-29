"""Parameter routing for multi-observation inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from baymorph.fitting.hierarchical import (
    physical_prior_copula_values,
    physical_prior_copula_values_from_matrix,
)
from baymorph.fitting.observation_covariates import (
    wavelength_reference_correlation_matrix,
)
from baymorph.fitting.parameters import (
    Parameter,
    ParameterSet,
)
from baymorph.fitting.priors import (
    BetaPrior,
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
    "wavelength",
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
        interface, the latent correlation uses a Beta(2, 2) reference prior.
        """
        if correlation_prior is None:
            correlation_prior = BetaPrior(
                alpha=2.0,
                beta=2.0,
            )

        return cls(
            name=name,
            coupling="hierarchical",
            physical_prior=prior,
            correlation_prior=correlation_prior,
        )

    @classmethod
    def wavelength(
        cls,
        name: str,
        *,
        prior: Prior,
        reference_correlation_prior: Prior | None = None,
    ) -> MultiObservationParameterSpec:
        """Create physical-prior-preserving wavelength coupling.

        The physical marginal prior is supplied directly by ``prior``.
        Dependence is represented by a Gaussian copula whose correlation
        decreases smoothly with logarithmic wavelength separation.

        The latent correlation assigned to a factor-two wavelength
        separation uses a Beta(2, 2) reference prior unless explicitly
        overridden through the advanced interface.
        """
        if reference_correlation_prior is None:
            reference_correlation_prior = BetaPrior(
                alpha=2.0,
                beta=2.0,
            )

        return cls(
            name=name,
            coupling="wavelength",
            physical_prior=prior,
            correlation_prior=reference_correlation_prior,
        )


class MultiObservationParameterLayout:
    """Map one sampled state to physical parameters for every observation."""

    def __init__(
        self,
        observations: tuple[str, ...] | list[str],
        specifications: tuple[MultiObservationParameterSpec, ...]
        | list[MultiObservationParameterSpec] = (),
        wavelength_coordinates: Mapping[str, float] | None = None,
    ) -> None:
        self._observations = tuple(
            observations
        )
        self._specifications = tuple(
            specifications
        )
        self._wavelength_coordinates = (
            None
            if wavelength_coordinates is None
            else {
                str(name): float(value)
                for name, value in wavelength_coordinates.items()
            }
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

            assert specification.coupling in {
                "hierarchical",
                "wavelength",
            }
            assert specification.physical_prior is not None
            assert specification.correlation_prior is not None

            dependence_name = (
                self.correlation_name(
                    specification.name
                )
                if specification.coupling == "hierarchical"
                else self.reference_correlation_name(
                    specification.name
                )
            )

            global_parameters.append(
                Parameter(
                    name=dependence_name,
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
            if specification.coupling in {
                "hierarchical",
                "wavelength",
            }
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
                    routes[
                        observation
                    ][
                        specification.name
                    ] = specification.name

                continue

            for observation in self._observations:
                routes[
                    observation
                ][
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
    def reference_correlation_name(
        name: str,
    ) -> str:
        """Return the wavelength reference-correlation parameter name."""
        return f"{name}__rstar"

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
            if specification.coupling not in {
                "hierarchical",
                "wavelength",
            }:
                continue

            assert specification.physical_prior is not None

            latent_values = np.asarray(
                [
                    float(
                        complete[
                            self.latent_name(
                                specification.name,
                                observation,
                            )
                        ]
                    )
                    for observation in self._observations
                ],
                dtype=float,
            )

            if not np.all(
                np.isfinite(
                    latent_values
                )
            ):
                raise ValueError(
                    f"Latent coordinates for dependent parameter "
                    f"'{specification.name}' must be finite."
                )

            if specification.coupling == "hierarchical":
                correlation = float(
                    complete[
                        self.correlation_name(
                            specification.name
                        )
                    ]
                )

                physical_values = (
                    physical_prior_copula_values(
                        specification.physical_prior,
                        latent_values,
                        correlation,
                    )
                )
            else:
                assert self._wavelength_coordinates is not None

                reference_correlation = float(
                    complete[
                        self.reference_correlation_name(
                            specification.name
                        )
                    ]
                )

                observation_entries = {
                    observation: {
                        "covariates": {
                            "wavelength": {
                                "value": float(
                                    np.exp(
                                        self._wavelength_coordinates[
                                            observation
                                        ]
                                    )
                                ),
                                "unit": "m",
                            }
                        }
                    }
                    for observation in self._observations
                }

                _, correlation_matrix = (
                    wavelength_reference_correlation_matrix(
                        observation_entries,
                        reference_correlation=reference_correlation,
                    )
                )

                physical_values = (
                    physical_prior_copula_values_from_matrix(
                        specification.physical_prior,
                        latent_values,
                        correlation_matrix,
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

            if specification.coupling == "wavelength":
                self._validate_wavelength(
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
        """Validate physical-prior-preserving exchangeable coupling."""
        self._validate_copula_specification(
            specification,
            label="Hierarchical",
        )

    def _validate_wavelength(
        self,
        specification: MultiObservationParameterSpec,
    ) -> None:
        """Validate physical-prior-preserving wavelength coupling."""
        self._validate_copula_specification(
            specification,
            label="Wavelength",
        )

        if self._wavelength_coordinates is None:
            raise ValueError(
                f"Wavelength parameter '{specification.name}' "
                "requires wavelength coordinates."
            )

        supplied = set(
            self._wavelength_coordinates
        )
        expected = set(
            self._observations
        )

        if supplied != expected:
            raise ValueError(
                f"Wavelength parameter '{specification.name}' "
                "requires coordinates for exactly the configured "
                "observations."
            )

        values = np.asarray(
            [
                self._wavelength_coordinates[
                    observation
                ]
                for observation in self._observations
            ],
            dtype=float,
        )

        if not np.all(
            np.isfinite(
                values
            )
        ):
            raise ValueError(
                "Wavelength coordinates must be finite."
            )

    def _validate_copula_specification(
        self,
        specification: MultiObservationParameterSpec,
        *,
        label: str,
    ) -> None:
        """Validate one physical-prior-preserving Gaussian-copula coupling."""
        if len(self._observations) < 2:
            raise ValueError(
                f"{label} parameter '{specification.name}' "
                "requires at least two observations."
            )

        if specification.shared is not None:
            raise ValueError(
                f"{label} parameter '{specification.name}' "
                "cannot define a shared parameter."
            )

        if specification.by_observation is not None:
            raise ValueError(
                f"{label} parameter '{specification.name}' "
                "cannot define independent parameters."
            )

        if specification.physical_prior is None:
            raise ValueError(
                f"{label} parameter '{specification.name}' "
                "requires a physical prior."
            )

        if not isinstance(
            specification.physical_prior,
            Prior,
        ):
            raise TypeError(
                f"Physical prior for {label.lower()} parameter "
                f"'{specification.name}' must be a Prior."
            )

        if specification.physical_prior.is_fixed:
            raise ValueError(
                f"{label} parameter '{specification.name}' "
                "cannot use a fixed physical prior."
            )

        correlation_prior = (
            specification.correlation_prior
        )

        if correlation_prior is None:
            raise ValueError(
                f"{label} parameter '{specification.name}' "
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
                    f"{label} correlation prior for "
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
                    f"{label} correlation prior for "
                    f"'{specification.name}' must have support "
                    "inside [0, 1]."
                )

            return

        if isinstance(
            correlation_prior,
            BetaPrior,
        ):
            if (
                correlation_prior.lower < 0.0
                or correlation_prior.upper > 1.0
            ):
                raise ValueError(
                    f"{label} correlation prior for "
                    f"'{specification.name}' must have support "
                    "inside [0, 1]."
                )

            return

        raise TypeError(
            f"{label} correlation prior for "
            f"'{specification.name}' must have known support "
            "inside [0, 1]. FixedPrior, UniformPrior, and "
            "BetaPrior are currently supported."
        )
