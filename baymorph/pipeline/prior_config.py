"""Shared configuration parsing for BayMorph prior distributions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from baymorph.fitting.priors import (
    BetaPrior,
    FixedPrior,
    LogUniformPrior,
    NormalPrior,
    Prior,
    UniformPrior,
)


def _parse_numeric_value(
    value: Any,
) -> float:
    """Parse one numeric prior value, including supported symbolic constants."""
    if isinstance(
        value,
        str,
    ):
        symbolic_values = {
            "pi": math.pi,
            "pi/2": math.pi / 2.0,
            "-pi/2": -math.pi / 2.0,
            "-pi": -math.pi,
        }

        normalized = value.strip().lower()

        if normalized in symbolic_values:
            return float(
                symbolic_values[
                    normalized
                ]
            )

    return float(
        value
    )


def parse_prior_config(
    config: Mapping[str, Any],
    *,
    references: Mapping[str, float] | None = None,
) -> Prior:
    """Build one typed prior from a YAML-compatible mapping.

    Uniform priors may optionally be expressed relative to an explicit
    reference value using ``relative_to``. The resulting object is still an
    ordinary absolute :class:`UniformPrior`.

    Relative priors are intended for deterministic coordinate references,
    such as a user-supplied target seed transformed into cutout coordinates.
    """
    if not isinstance(
        config,
        Mapping,
    ):
        raise TypeError(
            "Prior configuration must be a mapping."
        )

    prior_type = str(
        config.get(
            "type",
            "",
        )
    ).strip().lower()

    if prior_type == "fixed":
        _require_exact_keys(
            config,
            {
                "type",
                "value",
            },
            context="Fixed prior",
        )

        return FixedPrior(
            value=_parse_numeric_value(
                config["value"]
            ),
        )

    if prior_type == "uniform":
        has_reference = "relative_to" in config

        expected = {
            "type",
            "lower",
            "upper",
        }

        if has_reference:
            expected.add(
                "relative_to"
            )

        _require_exact_keys(
            config,
            expected,
            context="Uniform prior",
        )

        lower = _parse_numeric_value(
            config["lower"]
        )
        upper = _parse_numeric_value(
            config["upper"]
        )

        if has_reference:
            reference_name = str(
                config["relative_to"]
            ).strip()

            if not reference_name:
                raise ValueError(
                    "Uniform prior relative_to must be a non-empty string."
                )

            if references is None:
                raise ValueError(
                    "Relative uniform prior requires reference values."
                )

            if reference_name not in references:
                raise ValueError(
                    f"Unknown prior reference {reference_name!r}."
                )

            reference_value = float(
                references[
                    reference_name
                ]
            )

            lower += reference_value
            upper += reference_value

        return UniformPrior(
            lower=lower,
            upper=upper,
        )

    if prior_type == "log_uniform":
        _require_exact_keys(
            config,
            {
                "type",
                "lower",
                "upper",
            },
            context="Log-uniform prior",
        )

        return LogUniformPrior(
            lower=_parse_numeric_value(
                config["lower"]
            ),
            upper=_parse_numeric_value(
                config["upper"]
            ),
        )

    if prior_type == "beta":
        has_lower = "lower" in config
        has_upper = "upper" in config

        if has_lower or has_upper:
            expected = {
                "type",
                "alpha",
                "beta",
                "lower",
                "upper",
            }
        else:
            expected = {
                "type",
                "alpha",
                "beta",
            }

        _require_exact_keys(
            config,
            expected,
            context="Beta prior",
        )

        return BetaPrior(
            alpha=_parse_numeric_value(
                config["alpha"]
            ),
            beta=_parse_numeric_value(
                config["beta"]
            ),
            lower=_parse_numeric_value(
                config.get(
                    "lower",
                    0.0,
                )
            ),
            upper=_parse_numeric_value(
                config.get(
                    "upper",
                    1.0,
                )
            ),
        )

    if prior_type == "normal":
        _require_exact_keys(
            config,
            {
                "type",
                "mean",
                "sigma",
            },
            context="Normal prior",
        )

        return NormalPrior(
            mean=_parse_numeric_value(
                config["mean"]
            ),
            sigma=_parse_numeric_value(
                config["sigma"]
            ),
        )

    raise ValueError(
        "Prior type must be one of "
        "'fixed', 'uniform', 'log_uniform', 'beta', or 'normal'."
    )


def _require_exact_keys(
    config: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    """Require an exact configuration-key set."""
    supplied = set(
        config
    )

    if supplied != expected:
        missing = sorted(
            expected.difference(
                supplied
            )
        )

        extra = sorted(
            supplied.difference(
                expected
            )
        )

        raise ValueError(
            f"{context} configuration keys are invalid. "
            f"Missing: {missing}. Extra: {extra}."
        )


_PARAMETER_PRIOR_ALIASES = {
    "q_bulge": "q",
    "q_disk": "q",
    "theta_bulge": "theta",
    "theta_disk": "theta",
}


def prior_config_for_parameter(
    name: str,
    priors: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the configured prior for one model parameter."""
    if name in priors:
        config = priors[name]

        if not isinstance(config, Mapping):
            raise TypeError(
                f"Prior configuration for parameter {name!r} "
                "must be a mapping."
            )

        return config

    alias = _PARAMETER_PRIOR_ALIASES.get(
        name
    )

    if (
        alias is not None
        and alias in priors
    ):
        config = priors[alias]

        if not isinstance(config, Mapping):
            raise TypeError(
                f"Prior configuration for parameter {alias!r} "
                "must be a mapping."
            )

        return config

    raise ValueError(
        f"Missing prior configuration for parameter: {name}"
    )


def ineffective_shared_priors(
    parameter_names: Sequence[str],
    priors: Mapping[str, Any],
    default_priors: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Return the declared shared geometry priors that would have no effect.

    A prior under a shared name, ``q`` or ``theta``, applies to a component
    parameter such as ``q_bulge`` only when that parameter has no prior of its
    own (see :func:`prior_config_for_parameter`). When the default priors of a
    model hold a prior for every component parameter that the shared name
    serves, and none under the shared name, every resolved configuration holds
    a prior for each of these parameters, so a prior under the shared name has
    no effect.

    Parameters
    ----------
    parameter_names : sequence of str
        Parameter names of the model, in model order.
    priors : mapping
        Priors configured for the model, with or without the defaults merged in.
    default_priors : mapping
        Default priors of the model, empty when it has none.

    Returns
    -------
    dict of str to tuple of str
        Each ineffective shared name, ``q`` before ``theta``, mapped to the
        component parameters that it serves, in model order. Empty when no
        shared prior is ineffective.
    """
    ineffective: dict[str, tuple[str, ...]] = {}

    for name in sorted(set(_PARAMETER_PRIOR_ALIASES.values())):
        served = tuple(
            parameter
            for parameter in parameter_names
            if _PARAMETER_PRIOR_ALIASES.get(parameter) == name
        )

        if (
            name in priors
            and name not in parameter_names
            and name not in default_priors
            and served
            and all(parameter in default_priors for parameter in served)
        ):
            ineffective[name] = served

    return ineffective


def shared_prior_names(parameter_names: Sequence[str]) -> tuple[str, ...]:
    """Return the shared geometry names that serve parameters of a model.

    A shared name, ``q`` or ``theta``, serves the component parameters, such
    as ``q_bulge`` and ``q_disk``, that take its prior when they have none of
    their own (see :func:`prior_config_for_parameter`).

    Parameters
    ----------
    parameter_names : sequence of str
        Parameter names of the model, in model order.

    Returns
    -------
    tuple of str
        The shared names that serve at least one parameter of the model and
        are not parameters themselves, ``q`` before ``theta``.
    """
    return tuple(
        name
        for name in sorted(set(_PARAMETER_PRIOR_ALIASES.values()))
        if name not in parameter_names
        and any(
            _PARAMETER_PRIOR_ALIASES.get(parameter) == name
            for parameter in parameter_names
        )
    )


def unknown_prior_names(
    parameter_names: Sequence[str],
    priors: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return the declared prior names that no parameter of a model reads.

    A prior is read under the name of a parameter of the model or under a
    shared name that serves one (see :func:`shared_prior_names`). A prior
    under any other name would have no effect.

    Parameters
    ----------
    parameter_names : sequence of str
        Parameter names of the model, in model order.
    priors : mapping
        Priors configured for the model, with or without the defaults merged in.

    Returns
    -------
    tuple of str
        The names of ``priors`` that are neither parameters of the model nor
        shared names that serve them, sorted. Empty when every name is read.
    """
    known = set(parameter_names) | set(shared_prior_names(parameter_names))

    return tuple(sorted(str(name) for name in priors if str(name) not in known))


__all__ = [
    "ineffective_shared_priors",
    "parse_prior_config",
    "prior_config_for_parameter",
    "shared_prior_names",
    "unknown_prior_names",
]
