"""Factory functions for constructing BayMorph inference backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from baymorph.fitting.backend import (
    InferenceBackend,
    backend_registry,
)
from baymorph.fitting.backends import register_builtin_backends
from baymorph.fitting.dynesty import DynestyFitter
from baymorph.fitting.native import NativeFitter
from baymorph.fitting.parameters import ParameterSet
from baymorph.fitting.pocomc import PocoMCFitter
from baymorph.fitting.problem import InferenceProblem
from baymorph.likelihood import ImageLikelihood


register_builtin_backends()


def make_sampler(
    config: Mapping[str, Any] | str | None,
    *,
    problem: InferenceProblem | None = None,
    model_factory=None,
    parameters: ParameterSet | None = None,
    likelihood: ImageLikelihood | None = None,
):
    """Construct a configured BayMorph inference backend.

    Parameters
    ----------
    config : mapping, str, or None
        Sampler configuration. A string is interpreted as the backend name.
        If ``None``, Nautilus is selected for the canonical
        ``InferenceProblem`` path, while the legacy single-image path
        retains pocoMC temporarily for backward compatibility.
    problem : InferenceProblem, optional
        Backend-neutral inference problem. When supplied, construction is
        delegated entirely to the backend registry.
    model_factory : callable, optional
        Legacy single-image model factory.
    parameters : ParameterSet, optional
        Legacy single-image fitted-parameter definitions.
    likelihood : ImageLikelihood, optional
        Legacy single-image likelihood.

    Returns
    -------
    InferenceBackend, DynestyFitter, PocoMCFitter, or NativeFitter
        Configured inference implementation.

    Notes
    -----
    The ``InferenceProblem`` path is the canonical BayMorph architecture.

    Legacy construction from ``model_factory``, ``parameters``, and
    ``likelihood`` is retained temporarily for backward compatibility and
    remains explicitly coupled to the historical fitter classes.
    """
    sampler_config = _normalize_config(
        config
    )

    if (
        config is None
        and problem is not None
    ):
        sampler_config["name"] = "nautilus"

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

        if sampler_name == "native":
            raise ValueError(
                "The archived native backend does not support "
                "InferenceProblem construction."
            )

        return backend_registry.create(
            sampler_name,
            problem=problem,
            config=sampler_config,
        )

    return _make_legacy_sampler(
        sampler_name=sampler_name,
        sampler_config=sampler_config,
        model_factory=model_factory,
        parameters=parameters,
        likelihood=likelihood,
    )


def _make_legacy_sampler(
    *,
    sampler_name: str,
    sampler_config: Mapping[str, Any],
    model_factory,
    parameters: ParameterSet | None,
    likelihood: ImageLikelihood | None,
):
    """Construct a backend through the legacy single-image interface."""
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
        "Available legacy samplers are "
        "'dynesty', 'pocomc', and 'native'."
    )


def _normalize_config(
    config: Mapping[str, Any] | str | None,
) -> dict[str, Any]:
    """Return a mutable sampler configuration dictionary."""
    if config is None:
        return {
            "name": "pocomc",
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


__all__ = [
    "InferenceBackend",
    "make_sampler",
]
