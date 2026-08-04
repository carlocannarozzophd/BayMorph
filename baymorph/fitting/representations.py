"""Construction of observation-indexed intrinsic representations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


ModelFactory = Callable[..., object]


class RepresentationFactory:
    """Build observation-indexed representations from one arbitrary model.

    Parameters
    ----------
    factory : callable
        Arbitrary model factory receiving keyword arguments and returning an
        object exposing a ``render(shape=...)`` method.
    fixed_kwargs : mapping
        Observation-indexed keyword arguments fixed outside the sampled state.
        Typical examples include the PSF kernel or numerical rendering
        configuration associated with each observation.

    Notes
    -----
    This class does not know the model family, parameter coupling, likelihood,
    observation operator, or sampling backend. It only combines fixed
    observation-specific configuration with routed sampled parameters.
    """

    def __init__(
        self,
        *,
        factory: ModelFactory,
        fixed_kwargs: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if not callable(factory):
            raise TypeError("factory must be callable.")

        if not fixed_kwargs:
            raise ValueError(
                "RepresentationFactory requires at least one observation."
            )

        normalized: dict[str, dict[str, Any]] = {}

        for observation, kwargs in fixed_kwargs.items():
            if not isinstance(observation, str) or not observation:
                raise TypeError(
                    "Observation names must be non-empty strings."
                )

            if not isinstance(kwargs, Mapping):
                raise TypeError(
                    "Fixed representation arguments must be mappings."
                )

            normalized[observation] = dict(kwargs)

        self.factory = factory
        self.fixed_kwargs = normalized

    @property
    def observations(self) -> tuple[str, ...]:
        """Return observation names in deterministic construction order."""
        return tuple(self.fixed_kwargs)

    def build(
        self,
        observation: str,
        sampled_kwargs: Mapping[str, float],
    ) -> object:
        """Build one intrinsic representation.

        Raises
        ------
        KeyError
            If the observation is not configured.
        ValueError
            If a sampled parameter attempts to overwrite fixed configuration.
        """
        if observation not in self.fixed_kwargs:
            raise KeyError(
                f"Unknown representation observation: {observation!r}."
            )

        if not isinstance(sampled_kwargs, Mapping):
            raise TypeError("sampled_kwargs must be a mapping.")

        fixed = self.fixed_kwargs[observation]
        overlap = set(fixed).intersection(sampled_kwargs)

        if overlap:
            conflicting = ", ".join(sorted(overlap))
            raise ValueError(
                "Sampled parameters cannot overwrite fixed representation "
                f"arguments: {conflicting}."
            )

        return self.factory(
            **fixed,
            **sampled_kwargs,
        )

    def build_all(
        self,
        sampled_kwargs: Mapping[str, Mapping[str, float]],
    ) -> dict[str, object]:
        """Build representations for every configured observation."""
        supplied = set(sampled_kwargs)
        expected = set(self.observations)

        if supplied != expected:
            raise ValueError(
                "Sampled representation parameters must define exactly the "
                "configured observations."
            )

        return {
            observation: self.build(
                observation,
                sampled_kwargs[observation],
            )
            for observation in self.observations
        }
