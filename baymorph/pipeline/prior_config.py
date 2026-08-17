"""Shared configuration parsing for BayMorph prior distributions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from baymorph.fitting.priors import (
    BetaPrior,
    FixedPrior,
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
        "'fixed', 'uniform', 'beta', or 'normal'."
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


__all__ = [
    "parse_prior_config",
]
