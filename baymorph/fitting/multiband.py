"""Parameter routing for restricted multi-band inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from baymorph.fitting.parameters import Parameter, ParameterSet

CouplingMode = Literal["shared", "band_specific"]


@dataclass(frozen=True)
class MultiBandParameterSpec:
    """Define the coupling and prior bounds of one model parameter."""

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
    def band_specific(
        cls,
        name: str,
        parameters: Mapping[str, Parameter],
    ) -> MultiBandParameterSpec:
        """Create a band-specific parameter specification."""
        return cls(
            name=name,
            coupling="band_specific",
            by_band=dict(parameters),
        )


class MultiBandParameterLayout:
    """Map one sampled parameter vector to per-band model parameters."""

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
                global_parameters.append(specification.shared)
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
                        lower=parameter.lower,
                        upper=parameter.upper,
                    )
                )

        self._parameter_set = ParameterSet(global_parameters)

    @property
    def bands(self) -> tuple[str, ...]:
        """Return the bands in deterministic routing order."""
        return self._bands

    @property
    def parameter_set(self) -> ParameterSet:
        """Return the global parameter set consumed by the sampler."""
        return self._parameter_set

    @property
    def ndim(self) -> int:
        """Return the global sampled dimension."""
        return self._parameter_set.ndim

    @staticmethod
    def global_name(
        name: str,
        band: str,
    ) -> str:
        """Return the unambiguous global name of a local parameter."""
        return f"{name}__{band}"

    def route(
        self,
        values: Mapping[str, float],
    ) -> dict[str, dict[str, float]]:
        """Route global named values to model parameters for every band."""
        missing = set(self._parameter_set.names).difference(values)

        if missing:
            missing_names = ", ".join(sorted(missing))
            raise KeyError(
                f"Missing global parameter values: {missing_names}."
            )

        routed = {
            band: {}
            for band in self._bands
        }

        for specification in self._specifications:
            if specification.coupling == "shared":
                value = values[specification.name]

                for band in self._bands:
                    routed[band][specification.name] = value

                continue

            for band in self._bands:
                routed[band][specification.name] = values[
                    self.global_name(
                        specification.name,
                        band,
                    )
                ]

        return routed

    def _validate(self) -> None:
        if not self._bands:
            raise ValueError(
                "Multi-band parameter layout requires at least one band."
            )

        if len(set(self._bands)) != len(self._bands):
            raise ValueError("Band names must be unique.")

        if any(not band for band in self._bands):
            raise ValueError("Band names must be non-empty strings.")

        names = [
            specification.name
            for specification in self._specifications
        ]

        if len(set(names)) != len(names):
            raise ValueError("Parameter names must be unique.")

        for specification in self._specifications:
            if specification.coupling == "shared":
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
                        f"Shared parameter '{specification.name}' cannot "
                        "define band-specific parameters."
                    )

                continue

            if specification.coupling != "band_specific":
                raise ValueError(
                    f"Unsupported coupling mode "
                    f"'{specification.coupling}'."
                )

            if specification.shared is not None:
                raise ValueError(
                    f"Band-specific parameter '{specification.name}' "
                    "cannot define a shared parameter."
                )

            if specification.by_band is None:
                raise ValueError(
                    f"Band-specific parameter '{specification.name}' "
                    "requires one definition per band."
                )

            supplied_bands = set(specification.by_band)
            expected_bands = set(self._bands)

            if supplied_bands != expected_bands:
                raise ValueError(
                    f"Band-specific parameter '{specification.name}' "
                    "must define exactly the configured bands."
                )

            for band, parameter in specification.by_band.items():
                if parameter.name != specification.name:
                    raise ValueError(
                        f"Parameter name for band '{band}' does not match "
                        f"specification '{specification.name}'."
                    )
