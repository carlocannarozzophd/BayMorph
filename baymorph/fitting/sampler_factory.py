"""Factory functions for constructing BayMorph sampling backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from baymorph.fitting.dynesty import DynestyFitter
from baymorph.fitting.parameters import ParameterSet
from baymorph.likelihood import ImageLikelihood


def make_sampler(
    config: Mapping[str, Any] | str | None,
    *,
    model_factory,
    parameters: ParameterSet,
    likelihood: ImageLikelihood,
):
    """Construct a configured sampling backend.

    Parameters
    ----------
    config : mapping, str, or None
        Sampler configuration. A string is interpreted as the sampler name.
        If ``None``, Dynesty is selected.
    model_factory : callable
        Callable receiving model parameters as keyword arguments and returning
        an object exposing a ``render(shape=...)`` method.
    parameters : ParameterSet
        Ordered fitted-parameter definitions.
    likelihood : ImageLikelihood
        Fully configured image likelihood.

    Returns
    -------
    DynestyFitter
        Configured sampling backend.

    Raises
    ------
    TypeError
        If ``config`` is not a mapping, string, or ``None``.
    ValueError
        If the requested sampler is unsupported.
    """
    sampler_config = _normalize_config(config)

    sampler_name = str(
        sampler_config.get(
            "name",
            "dynesty",
        )
    ).strip().lower()

    if sampler_name != "dynesty":
        raise ValueError(
            f"Unknown sampler {sampler_name!r}. "
            "The only currently available sampler is 'dynesty'."
        )

    return DynestyFitter(
        model_factory=model_factory,
        parameters=parameters,
        likelihood=likelihood,
    )


def _normalize_config(
    config: Mapping[str, Any] | str | None,
) -> dict[str, Any]:
    """Return a mutable sampler configuration dictionary."""
    if config is None:
        return {
            "name": "dynesty",
        }

    if isinstance(config, str):
        return {
            "name": config,
        }

    if not isinstance(config, Mapping):
        raise TypeError(
            "Sampler configuration must be a mapping, string, or None."
        )

    return dict(config)
