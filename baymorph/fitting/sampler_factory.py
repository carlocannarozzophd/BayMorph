"""Factory functions for constructing BayMorph sampling backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from baymorph.fitting.dynesty import DynestyFitter
from baymorph.fitting.native import NativeFitter
from baymorph.fitting.parameters import ParameterSet
from baymorph.fitting.pocomc import PocoMCFitter
from baymorph.fitting.problem import InferenceProblem
from baymorph.likelihood import ImageLikelihood


def make_sampler(
    config: Mapping[str, Any] | str | None,
    *,
    problem: InferenceProblem | None = None,
    model_factory=None,
    parameters: ParameterSet | None = None,
    likelihood: ImageLikelihood | None = None,
):
    """Construct a configured sampling backend.

    Parameters
    ----------
    config : mapping, str, or None
        Sampler configuration. A string is interpreted as the sampler name.
        If ``None``, Dynesty is selected.
    problem : InferenceProblem, optional
        Backend-neutral inference problem.
    model_factory : callable, optional
        Legacy single-image model factory.
    parameters : ParameterSet, optional
        Legacy single-image fitted-parameter definitions.
    likelihood : ImageLikelihood, optional
        Legacy single-image likelihood.

    Returns
    -------
    DynestyFitter, PocoMCFitter, or NativeFitter
        Configured sampling backend.
    """
    sampler_config = _normalize_config(
        config
    )

    sampler_name = str(
        sampler_config.get(
            "name",
            "dynesty",
        )
    ).strip().lower()

    if problem is not None:
        if any(
            value is not None
            for value in (
                model_factory,
                parameters,
                likelihood,
            )
        ):
            raise ValueError(
                "problem cannot be combined with model_factory, "
                "parameters, or likelihood."
            )

        if not isinstance(
            problem,
            InferenceProblem,
        ):
            raise TypeError(
                "problem must be an InferenceProblem instance."
            )

        if sampler_name == "dynesty":
            return DynestyFitter(
                problem=problem
            )

        if sampler_name == "pocomc":
            return PocoMCFitter(
                problem=problem,
                config=sampler_config,
            )

        if sampler_name == "native":
            raise ValueError(
                "The archived native backend does not support "
                "InferenceProblem construction."
            )

    if sampler_name == "dynesty":
        return DynestyFitter(
            model_factory=model_factory,
            parameters=parameters,
            likelihood=likelihood,
        )

    if sampler_name == "pocomc":
        return PocoMCFitter(
            model_factory=model_factory,
            parameters=parameters,
            likelihood=likelihood,
            config=sampler_config,
        )

    if sampler_name == "native":
        return NativeFitter(
            model_factory=model_factory,
            parameters=parameters,
            likelihood=likelihood,
            config=sampler_config,
        )

    raise ValueError(
        f"Unknown sampler {sampler_name!r}. "
        "Available samplers are 'dynesty', 'pocomc', and 'native'."
    )


def _normalize_config(
    config: Mapping[str, Any] | str | None,
) -> dict[str, Any]:
    """Return a mutable sampler configuration dictionary."""
    if config is None:
        return {
            "name": "dynesty",
        }

    if isinstance(
        config,
        str,
    ):
        return {
            "name": config,
        }

    if not isinstance(
        config,
        Mapping,
    ):
        raise TypeError(
            "Sampler configuration must be a mapping, "
            "string, or None."
        )

    return dict(
        config
    )
