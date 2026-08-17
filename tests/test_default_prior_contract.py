"""Tests for the canonical BayMorph default-prior configuration contract."""

from __future__ import annotations

from collections.abc import Mapping

from baymorph.pipeline.config import DEFAULT_CONFIG
from baymorph.pipeline.prior_config import parse_prior_config


_LEGACY_DATA_SCALED_KEYS = {
    "min_factor",
    "max_factor",
}


def test_all_default_model_priors_are_typed() -> None:
    """Every package model prior must use the shared typed-prior language."""
    model_priors = DEFAULT_CONFIG["fit"]["priors"]

    for model_name, priors in model_priors.items():
        for parameter_name, config in priors.items():
            assert isinstance(
                config,
                Mapping,
            )

            assert "type" in config, (
                f"{model_name}.{parameter_name} does not use "
                "typed prior syntax."
            )

            references = None

            if config.get("relative_to") == "target_x":
                references = {
                    "target_x": 0.0,
                }

            elif config.get("relative_to") == "target_y":
                references = {
                    "target_y": 0.0,
                }

            parse_prior_config(
                config,
                references=references,
            )


def test_background_default_priors_are_typed() -> None:
    """Residual-background defaults must use the shared prior language."""
    priors = DEFAULT_CONFIG[
        "fit"
    ][
        "background"
    ][
        "priors"
    ]

    for config in priors.values():
        parse_prior_config(
            config
        )


def test_no_data_scaled_default_priors_remain() -> None:
    """Package defaults must not derive scientific prior support from data."""
    model_priors = DEFAULT_CONFIG["fit"]["priors"]

    for model_name, priors in model_priors.items():
        for parameter_name, config in priors.items():
            legacy_keys = _LEGACY_DATA_SCALED_KEYS.intersection(
                config
            )

            assert not legacy_keys, (
                f"{model_name}.{parameter_name} still uses legacy "
                f"data-scaled prior keys: {sorted(legacy_keys)}."
            )


def test_default_centroid_priors_reference_explicit_target_seed() -> None:
    """Default centre priors must reference target seeds, not image centroids."""
    for priors in DEFAULT_CONFIG["fit"]["priors"].values():
        if "x0" in priors:
            assert priors[
                "x0"
            ][
                "relative_to"
            ] == "target_x"

        if "y0" in priors:
            assert priors[
                "y0"
            ][
                "relative_to"
            ] == "target_y"
