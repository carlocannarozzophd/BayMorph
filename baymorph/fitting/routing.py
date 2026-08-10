"""Deterministic routing of global inference parameters to observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class ProjectionParameterRouter:
    """Route global parameter values through observation-specific projections.

    Each observation defines a mapping from the local parameter name required
    by its representation to the corresponding global inferential-state name.

    This router performs only deterministic projection and renaming. It does
    not define priors, sampled dimensionality, or probabilistic coupling.
    """

    def __init__(
        self,
        *,
        observations: Sequence[str],
        routes: Mapping[str, Mapping[str, str]],
    ) -> None:
        resolved_observations = tuple(observations)

        if not resolved_observations:
            raise ValueError(
                "Parameter routing requires at least one observation."
            )

        if len(set(resolved_observations)) != len(
            resolved_observations
        ):
            raise ValueError(
                "Observation names must be unique."
            )

        if any(
            not isinstance(observation, str)
            or not observation
            for observation in resolved_observations
        ):
            raise TypeError(
                "Observation names must be non-empty strings."
            )

        supplied = set(routes)
        expected = set(resolved_observations)

        if supplied != expected:
            raise ValueError(
                "Parameter routes must define exactly the configured "
                "observations."
            )

        normalized: dict[str, dict[str, str]] = {}

        for observation in resolved_observations:
            observation_routes = routes[observation]

            if not isinstance(
                observation_routes,
                Mapping,
            ):
                raise TypeError(
                    "Observation parameter routes must be mappings."
                )

            normalized_routes: dict[str, str] = {}

            for local_name, global_name in observation_routes.items():
                if (
                    not isinstance(local_name, str)
                    or not local_name
                ):
                    raise TypeError(
                        "Local parameter names must be non-empty strings."
                    )

                if (
                    not isinstance(global_name, str)
                    or not global_name
                ):
                    raise TypeError(
                        "Global parameter names must be non-empty strings."
                    )

                normalized_routes[
                    local_name
                ] = global_name

            normalized[
                observation
            ] = normalized_routes

        self._observations = resolved_observations
        self._routes = normalized

    @property
    def observations(self) -> tuple[str, ...]:
        """Return observations in deterministic routing order."""
        return self._observations

    def route(
        self,
        global_values: Mapping[str, float],
    ) -> dict[str, dict[str, float]]:
        """Project one complete global state to all observations."""
        routed: dict[str, dict[str, float]] = {}

        for observation in self._observations:
            observation_values: dict[str, float] = {}

            for (
                local_name,
                global_name,
            ) in self._routes[observation].items():
                if global_name not in global_values:
                    raise KeyError(
                        "Missing global parameter value "
                        f"'{global_name}'."
                    )

                observation_values[
                    local_name
                ] = float(
                    global_values[
                        global_name
                    ]
                )

            routed[
                observation
            ] = observation_values

        return routed
