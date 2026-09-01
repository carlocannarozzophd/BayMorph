"""Factory for constructing BayMorph inference backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from baymorph.fitting.backend import (
    InferenceBackend,
    backend_registry,
)
from baymorph.fitting.backends import register_builtin_backends
from baymorph.fitting.problem import InferenceProblem


register_builtin_backends()


def make_sampler(
    config: Mapping[str, Any] | str | None,
    *,
    problem: InferenceProblem,
) -> InferenceBackend:
    """Construct a backend for one backend-neutral inference problem.

    Parameters
    ----------
    config : mapping, str, or None
        Backend configuration. A string is interpreted as the backend name.
        If ``None`` or if a mapping omits ``name``, Nautilus is selected.
    problem : InferenceProblem
        Complete backend-neutral probabilistic problem.

    Returns
    -------
    InferenceBackend
        Configured inference backend.

    Notes
    -----
    Scientific model construction, priors, likelihoods, and observation
    semantics belong to ``InferenceProblem``. The backend factory owns only
    numerical backend selection and configuration.
    """
    if not isinstance(
        problem,
        InferenceProblem,
    ):
        raise TypeError(
            "problem must be an InferenceProblem instance."
        )

    sampler_config = _normalize_config(
        config
    )

    sampler_name = str(
        sampler_config.get(
            "name",
            "nautilus",
        )
    ).strip().lower()

    return backend_registry.create(
        sampler_name,
        problem=problem,
        config=sampler_config,
    )


def _normalize_config(
    config: Mapping[str, Any] | str | None,
) -> dict[str, Any]:
    """Return a mutable backend configuration dictionary."""
    if config is None:
        return {
            "name": "nautilus",
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
