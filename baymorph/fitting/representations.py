"""Construction of observation-indexed intrinsic representations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


ModelFactory = Callable[..., object]
ParameterTransform = Callable[
    [Mapping[str, float]],
    Mapping[str, float],
]


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
    parameter_transforms : mapping, optional
        Observation-indexed deterministic transformations applied to routed
        inferential parameters before representation construction.

    Notes
    -----
    This class does not define the model family, parameter coupling,
    likelihood, observation operator, or sampling backend.

    Parameter coupling and projection occur before representation
    construction. Optional parameter transformations may convert those
    scientific parameters into the coordinate system or units required by
    an individual observation.
    """

    def __init__(
        self,
        *,
        factory: ModelFactory,
        fixed_kwargs: Mapping[str, Mapping[str, Any]],
        parameter_transforms: Mapping[
            str,
            ParameterTransform,
        ]
        | None = None,
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

        transforms = dict(
            parameter_transforms
            or {}
        )

        unknown = set(transforms).difference(
            normalized
        )

        if unknown:
            names = ", ".join(
                sorted(unknown)
            )
            raise ValueError(
                "Parameter transforms reference unknown observations: "
                f"{names}."
            )

        for observation, transform in transforms.items():
            if not callable(transform):
                raise TypeError(
                    "Parameter transform for observation "
                    f"'{observation}' must be callable."
                )

        self.factory = factory
        self.fixed_kwargs = normalized
        self.parameter_transforms = transforms

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
            If a transformed parameter attempts to overwrite fixed
            configuration.
        """
        if observation not in self.fixed_kwargs:
            raise KeyError(
                f"Unknown representation observation: {observation!r}."
            )

        if not isinstance(sampled_kwargs, Mapping):
            raise TypeError("sampled_kwargs must be a mapping.")

        parameters: Mapping[str, float] = sampled_kwargs

        transform = self.parameter_transforms.get(
            observation
        )

        if transform is not None:
            transformed = transform(
                dict(sampled_kwargs)
            )

            if not isinstance(
                transformed,
                Mapping,
            ):
                raise TypeError(
                    "Parameter transforms must return mappings."
                )

            parameters = transformed

        fixed = self.fixed_kwargs[observation]
        overlap = set(fixed).intersection(
            parameters
        )

        if overlap:
            conflicting = ", ".join(
                sorted(overlap)
            )
            raise ValueError(
                "Representation parameters cannot overwrite fixed "
                f"arguments: {conflicting}."
            )

        return self.factory(
            **fixed,
            **parameters,
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
