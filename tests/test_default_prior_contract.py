"""Tests for the canonical BayMorph default-prior configuration contract."""

from __future__ import annotations

from collections.abc import Mapping

from baymorph.pipeline.config import DEFAULT_CONFIG
from baymorph.pipeline.prior_config import parse_prior_config


_DATA_SCALED_KEYS = {
    "min_factor",
    "max_factor",
}


def test_non_flux_default_priors_are_typed() -> None:
    """Non-data-scaled production defaults must use typed prior syntax."""
    model_priors = DEFAULT_CONFIG[
        "fit"
    ][
        "priors"
    ]

    for model_name, priors in model_priors.items():
        for parameter_name, config in priors.items():
            assert isinstance(
                config,
                Mapping,
            )

            if _DATA_SCALED_KEYS.intersection(
                config
            ):
                assert parameter_name.startswith(
                    "flux"
                )
                continue

            assert "type" in config, (
                f"{model_name}.{parameter_name} does not use "
                "typed prior syntax."
            )

            references = None

            if config.get(
                "relative_to"
            ) == "target_x":
                references = {
                    "target_x": 0.0,
                }

            elif config.get(
                "relative_to"
            ) == "target_y":
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


def test_only_flux_defaults_remain_data_scaled() -> None:
    """Legacy data scaling must be restricted to unresolved flux priors."""
    model_priors = DEFAULT_CONFIG[
        "fit"
    ][
        "priors"
    ]

    data_scaled = []

    for model_name, priors in model_priors.items():
        for parameter_name, config in priors.items():
            if _DATA_SCALED_KEYS.intersection(
                config
            ):
                data_scaled.append(
                    (
                        model_name,
                        parameter_name,
                    )
                )

    assert data_scaled

    assert all(
        parameter_name.startswith(
            "flux"
        )
        for _, parameter_name in data_scaled
    )


def test_default_centroid_priors_reference_explicit_target_seed() -> None:
    """Default centre priors must reference target seeds, not image centroids."""
    for priors in DEFAULT_CONFIG[
        "fit"
    ][
        "priors"
    ].values():
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
