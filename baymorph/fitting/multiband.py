"""Parameter routing for restricted multi-band inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from baymorph.fitting.parameters import Parameter, ParameterSet
from baymorph.fitting.routing import ProjectionParameterRouter

CouplingMode = Literal["shared", "independent"]


@dataclass(frozen=True)
class MultiBandParameterSpec:
    """Define the coupling and prior of one model parameter."""

    name: str
    coupling: CouplingMode
    shared: Parameter | None = None
    by_band: Mapping[str, Parameter] | None = None

    @classmethod
    def shared_parameter(
        cls,
        parameter: Parameter,
    ) -> MultiBandParameterSpec:
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
    ) -> MultiBandParameterSpec:
        """Create an independently parameterized specification."""
        return cls(
            name=name,
            coupling="independent",
            by_band=dict(parameters),
        )

    @classmethod
    def band_specific(
        cls,
        name: str,
        parameters: Mapping[str, Parameter],
    ) -> MultiBandParameterSpec:
        """Create an independent specification through the legacy alias."""
        return cls.independent(
            name=name,
            parameters=parameters,
        )


class MultiBandParameterLayout:
    """Map one sampled state to complete parameters for every band."""

    def __init__(
        self,
        bands: tuple[str, ...] | list[str],
        specifications: tuple[MultiBandParameterSpec, ...]
        | list[MultiBandParameterSpec],
    ) -> None:
        self._bands = tuple(bands)
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

            assert specification.by_band is not None

            for band in self._bands:
                parameter = specification.by_band[band]

                global_parameters.append(
                    Parameter(
                        name=self.global_name(
                            specification.name,
                            band,
                        ),
                        prior=parameter.prior,
                    )
                )

        self._parameter_set = ParameterSet(
            global_parameters
        )

        routes = {
            band: {}
            for band in self._bands
        }

        for specification in self._specifications:
            if specification.coupling == "shared":
                for band in self._bands:
                    routes[band][
                        specification.name
                    ] = specification.name
                continue

            for band in self._bands:
                routes[band][
                    specification.name
                ] = self.global_name(
                    specification.name,
                    band,
                )

        self._router = ProjectionParameterRouter(
            observations=self._bands,
            routes=routes,
        )

    @property
    def bands(self) -> tuple[str, ...]:
        """Return bands in deterministic routing order."""
        return self._bands

    @property
    def specifications(
        self,
    ) -> tuple[MultiBandParameterSpec, ...]:
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
        band: str,
    ) -> str:
        """Return the unambiguous global name of an independent parameter."""
        return f"{name}__{band}"

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
        ).difference(complete)

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
        """Route global values to complete parameters for every band."""
        complete = self.complete_global_values(
            values
        )

        return self._router.route(
            complete
        )

    def _validate(self) -> None:
        if not self._bands:
            raise ValueError(
                "Multi-band parameter layout requires at least one band."
            )

        if len(set(self._bands)) != len(self._bands):
            raise ValueError(
                "Band names must be unique."
            )

        if any(
            not isinstance(band, str) or not band
            for band in self._bands
        ):
            raise ValueError(
                "Band names must be non-empty strings."
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
        specification: MultiBandParameterSpec,
    ) -> None:
        """Validate one shared parameter specification."""
        if specification.shared is None:
            raise ValueError(
                f"Shared parameter '{specification.name}' "
                "requires one Parameter definition."
            )

        if specification.shared.name != specification.name:
            raise ValueError(
                "Shared parameter name does not match its "
                "specification."
            )

        if specification.by_band is not None:
            raise ValueError(
                f"Shared parameter '{specification.name}' "
                "cannot define independent parameters."
            )

    def _validate_independent(
        self,
        specification: MultiBandParameterSpec,
    ) -> None:
        """Validate one independent parameter specification."""
        if specification.shared is not None:
            raise ValueError(
                f"Independent parameter '{specification.name}' "
                "cannot define a shared parameter."
            )

        if specification.by_band is None:
            raise ValueError(
                f"Independent parameter '{specification.name}' "
                "requires one definition per band."
            )

        supplied_bands = set(
            specification.by_band
        )
        expected_bands = set(
            self._bands
        )

        if supplied_bands != expected_bands:
            raise ValueError(
                f"Independent parameter '{specification.name}' "
                "must define exactly the configured bands."
            )

        for band, parameter in specification.by_band.items():
            if parameter.name != specification.name:
                raise ValueError(
                    f"Parameter name for band '{band}' "
                    f"does not match specification "
                    f"'{specification.name}'."
                )
