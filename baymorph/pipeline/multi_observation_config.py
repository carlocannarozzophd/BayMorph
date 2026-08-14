"""Configuration parsing for production multi-observation inference."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from baymorph.fitting.multi_observation import (
    MultiObservationParameterLayout,
    MultiObservationParameterSpec,
)
from baymorph.fitting.parameters import Parameter
from baymorph.fitting.priors import (
    FixedPrior,
    NormalPrior,
    Prior,
    UniformPrior,
)
from baymorph.models import get_model
from baymorph.models.geometry import sky_parameter_names
from baymorph.pipeline.config import load_resolved_config

SINGLE_SERSIC_PARAMETER_NAMES = (
    "flux",
    "re",
    "n",
    "q",
    "theta",
    "x0",
    "y0",
    "sky",
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
            value=config[
                "value"
            ],
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
            lower=config[
                "lower"
            ],
            upper=config[
                "upper"
            ],
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
            mean=config[
                "mean"
            ],
            sigma=config[
                "sigma"
            ],
        )

    raise ValueError(
        "Prior type must be one of "
        "'fixed', 'uniform', or 'normal'."
    )


def resolve_multi_observation_observation_configs(
    config: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load resolved single-observation configurations."""
    observations = config.get(
        "observations"
    )

    if (
        not isinstance(
            observations,
            Mapping,
        )
        or not observations
    ):
        raise ValueError(
            "Multi-observation configuration requires a non-empty "
            "'observations' mapping."
        )

    root = (
        Path(
            base_dir
        )
        if base_dir is not None
        else Path.cwd()
    )

    resolved: dict[
        str,
        dict[str, Any],
    ] = {}

    for observation, observation_config in observations.items():
        if (
            not isinstance(
                observation,
                str,
            )
            or not observation
        ):
            raise ValueError(
                "Observation names must be non-empty strings."
            )

        if not isinstance(
            observation_config,
            Mapping,
        ):
            raise TypeError(
                f"Observation '{observation}' configuration "
                "must be a mapping."
            )

        _require_exact_keys(
            observation_config,
            {
                "config_path",
            },
            context=f"Observation '{observation}'",
        )

        config_path = Path(
            observation_config[
                "config_path"
            ]
        )

        if not config_path.is_absolute():
            config_path = (
                root
                / config_path
            )

        resolved[
            observation
        ] = load_resolved_config(
            config_path
        )

    return resolved


def build_multi_observation_parameter_layout(
    config: Mapping[str, Any],
) -> MultiObservationParameterLayout:
    """Build the production multi-observation parameter layout."""
    observations = _observation_names(
        config
    )

    fit_config = config.get(
        "fit"
    )

    if not isinstance(
        fit_config,
        Mapping,
    ):
        raise ValueError(
            "Multi-observation configuration requires a 'fit' mapping."
        )

    models = fit_config.get(
        "models"
    )

    if (
        not isinstance(
            models,
            list,
        )
        or len(
            models
        )
        != 1
        or not isinstance(
            models[
                0
            ],
            str,
        )
        or not models[
            0
        ]
    ):
        raise ValueError(
            "The production multi-observation workflow currently "
            "requires exactly one registered model."
        )

    model_name = models[
        0
    ]

    model_definition = get_model(
        model_name
    )

    multi_observation_config = (
        _multi_observation_fit_section(
            fit_config
        )
    )

    parameters = multi_observation_config.get(
        "parameters"
    )

    if not isinstance(
        parameters,
        Mapping,
    ):
        raise ValueError(
            "Fit multi-observation configuration requires "
            "a 'parameters' mapping."
        )

    parameter_frame = (
        multi_observation_parameter_frame(
            config
        )
    )

    if parameter_frame == "sky":
        parameter_names = sky_parameter_names(
            model_name
        )
    else:
        parameter_names = tuple(
            model_definition.parameter_names
        )

    supplied = set(
        parameters
    )
    expected = set(
        parameter_names
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
            "Multi-observation model parameters must be "
            "defined exactly. "
            f"Missing: {missing}. Extra: {extra}."
        )

    specifications = [
        _parse_parameter_specification(
            name=name,
            config=parameters[
                name
            ],
            observations=observations,
        )
        for name in parameter_names
    ]

    return MultiObservationParameterLayout(
        observations=observations,
        specifications=specifications,
    )


def resolve_multi_observation_config(
    config: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> tuple[
    dict[str, dict[str, Any]],
    MultiObservationParameterLayout,
]:
    """Resolve observation configurations and parameter coupling."""
    observation_configs = (
        resolve_multi_observation_observation_configs(
            config,
            base_dir=base_dir,
        )
    )

    layout = (
        build_multi_observation_parameter_layout(
            config
        )
    )

    return (
        observation_configs,
        layout,
    )


def multi_observation_parameter_frame(
    config: Mapping[str, Any],
) -> str:
    """Return the declared multi-observation inferential parameter frame."""
    fit_config = config.get(
        "fit"
    )

    if not isinstance(
        fit_config,
        Mapping,
    ):
        raise ValueError(
            "Multi-observation configuration requires a 'fit' mapping."
        )

    section = (
        _multi_observation_fit_section(
            fit_config
        )
    )

    frame = str(
        section.get(
            "parameter_frame",
            "detector",
        )
    ).strip().lower()

    if frame not in {
        "detector",
        "sky",
    }:
        raise ValueError(
            "fit.multi_observation.parameter_frame must be "
            "'detector' or 'sky'."
        )

    return frame


def _multi_observation_fit_section(
    fit_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the canonical multi-observation fit configuration."""
    section = fit_config.get(
        "multi_observation"
    )

    if not isinstance(
        section,
        Mapping,
    ):
        raise TypeError(
            "Fit configuration requires a "
            "'multi_observation' mapping."
        )

    return section

def _observation_names(
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return validated observation names in configuration order."""
    observations = config.get(
        "observations"
    )

    if (
        not isinstance(
            observations,
            Mapping,
        )
        or not observations
    ):
        raise ValueError(
            "Multi-observation configuration requires a non-empty "
            "'observations' mapping."
        )

    names = tuple(
        observations
    )

    if any(
        not isinstance(
            name,
            str,
        )
        or not name
        for name in names
    ):
        raise ValueError(
            "Observation names must be non-empty strings."
        )

    return names


def _parse_parameter_specification(
    *,
    name: str,
    config: Any,
    observations: tuple[str, ...],
) -> MultiObservationParameterSpec:
    """Parse one explicit parameter-coupling specification."""
    if not isinstance(
        config,
        Mapping,
    ):
        raise TypeError(
            f"Parameter '{name}' configuration must be a mapping."
        )

    coupling = str(
        config.get(
            "coupling",
            "",
        )
    ).strip().lower()

    if coupling == "shared":
        _require_exact_keys(
            config,
            {
                "coupling",
                "prior",
            },
            context=(
                f"Shared parameter "
                f"'{name}'"
            ),
        )

        return (
            MultiObservationParameterSpec
            .shared_parameter(
                Parameter(
                    name,
                    prior=parse_prior_config(
                        config[
                            "prior"
                        ]
                    ),
                )
            )
        )

    if coupling == "independent":
        _require_exact_keys(
            config,
            {
                "coupling",
                "priors",
            },
            context=(
                f"Independent parameter "
                f"'{name}'"
            ),
        )

        priors = config[
            "priors"
        ]

        if not isinstance(
            priors,
            Mapping,
        ):
            raise TypeError(
                f"Independent parameter '{name}' priors "
                "must be a mapping."
            )

        supplied = set(
            priors
        )
        expected = set(
            observations
        )

        if supplied != expected:
            raise ValueError(
                f"Independent parameter '{name}' must define "
                "exactly the configured observations."
            )

        return (
            MultiObservationParameterSpec
            .independent(
                name,
                {
                    observation: Parameter(
                        name,
                        prior=parse_prior_config(
                            priors[
                                observation
                            ]
                        ),
                    )
                    for observation in observations
                },
            )
        )

    if coupling == "hierarchical_log_normal":
        _require_exact_keys(
            config,
            {
                "coupling",
                "location_prior",
                "scale_prior",
            },
            context=(
                f"Hierarchical parameter "
                f"'{name}'"
            ),
        )

        return (
            MultiObservationParameterSpec
            .hierarchical_log_normal(
                name,
                location=Parameter(
                    f"{name}__mu_log",
                    prior=parse_prior_config(
                        config[
                            "location_prior"
                        ]
                    ),
                ),
                scale=Parameter(
                    f"{name}__tau_log",
                    prior=parse_prior_config(
                        config[
                            "scale_prior"
                        ]
                    ),
                ),
            )
        )

    raise ValueError(
        f"Parameter '{name}' coupling must be "
        "'shared', 'independent', or "
        "'hierarchical_log_normal'."
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
    "SINGLE_SERSIC_PARAMETER_NAMES",
    "build_multi_observation_parameter_layout",
    "multi_observation_parameter_frame",
    "parse_prior_config",
    "resolve_multi_observation_config",
    "resolve_multi_observation_observation_configs",
]
