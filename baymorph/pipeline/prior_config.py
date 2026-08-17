"""Shared configuration parsing for BayMorph prior distributions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from baymorph.fitting.priors import (
    BetaPrior,
    FixedPrior,
    NormalPrior,
    Prior,
    UniformPrior,
)


def parse_prior_config(
    config: Mapping[str, Any],
) -> Prior:
    """Build one typed prior from a YAML-compatible mapping."""
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
            value=config["value"],
        )

    if prior_type == "uniform":
        _require_exact_keys(
            config,
            {
                "type",
                "lower",
                "upper",
            },
            context="Uniform prior",
        )

        return UniformPrior(
            lower=config["lower"],
            upper=config["upper"],
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
            alpha=config["alpha"],
            beta=config["beta"],
            lower=config.get(
                "lower",
                0.0,
            ),
            upper=config.get(
                "upper",
                1.0,
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
            mean=config["mean"],
            sigma=config["sigma"],
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
    supplied = set(config)

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
