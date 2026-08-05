"""Configuration parsing for production multi-band inference."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from baymorph.fitting.multiband import (
    MultiBandParameterLayout,
    MultiBandParameterSpec,
)
from baymorph.fitting.parameters import Parameter
from baymorph.fitting.priors import (
    FixedPrior,
    NormalPrior,
    Prior,
    UniformPrior,
)
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
    """Build one typed prior from a YAML-compatible mapping.

    Parameters
    ----------
    config : mapping
        Prior configuration with an explicit ``type`` entry.

    Returns
    -------
    Prior
        Parsed fixed, uniform, or normal prior.

    Raises
    ------
    TypeError
        If the configuration is not a mapping.
    ValueError
        If the prior type or configuration fields are invalid.
    """
    if not isinstance(config, Mapping):
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
        "'fixed', 'uniform', or 'normal'."
    )


def resolve_multiband_observation_configs(
    config: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load resolved single-observation configurations for every band.

    Parameters
    ----------
    config : mapping
        Joint multi-band configuration.
    base_dir : str or pathlib.Path, optional
        Directory relative to which observation configuration paths are
        resolved. The current working directory is used by default.

    Returns
    -------
    dict
        Mapping from observation name to resolved single-observation
        configuration.

    Raises
    ------
    TypeError
        If an observation configuration is not a mapping.
    ValueError
        If observations or their configuration fields are invalid.
    """
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
            "Multi-band configuration requires a non-empty "
            "'observations' mapping."
        )

    root = (
        Path(base_dir)
        if base_dir is not None
        else Path.cwd()
    )

    resolved: dict[
        str,
        dict[str, Any],
    ] = {}

    for band, observation_config in observations.items():
        if (
            not isinstance(
                band,
                str,
            )
            or not band
        ):
            raise ValueError(
                "Observation names must be non-empty strings."
            )

        if not isinstance(
            observation_config,
            Mapping,
        ):
            raise TypeError(
                f"Observation '{band}' configuration "
                "must be a mapping."
            )

        _require_exact_keys(
            observation_config,
            {
                "config_path",
            },
            context=f"Observation '{band}'",
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

        resolved[band] = load_resolved_config(
            config_path
        )

    return resolved


def build_multiband_parameter_layout(
    config: Mapping[str, Any],
) -> MultiBandParameterLayout:
    """Build the production single-Sérsic multi-band parameter layout.

    Parameters
    ----------
    config : mapping
        Joint multi-band configuration.

    Returns
    -------
    MultiBandParameterLayout
        Parameter layout preserving explicit coupling and typed priors.

    Raises
    ------
    TypeError
        If a parameter or prior configuration has an invalid type.
    ValueError
        If the model, coupling, parameter set, or prior definitions are
        invalid.
    """
    bands = _observation_names(
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
            "Multi-band configuration requires a 'fit' mapping."
        )

    models = fit_config.get(
        "models"
    )

    if models != [
        "single_sersic"
    ]:
        raise ValueError(
            "The production multi-band workflow currently supports "
            "exactly fit.models: ['single_sersic']."
        )

    multiband_config = fit_config.get(
        "multiband"
    )

    if not isinstance(
        multiband_config,
        Mapping,
    ):
        raise ValueError(
            "Fit configuration requires a 'multiband' mapping."
        )

    parameters = multiband_config.get(
        "parameters"
    )

    if not isinstance(
        parameters,
        Mapping,
    ):
        raise ValueError(
            "Fit multi-band configuration requires "
            "a 'parameters' mapping."
        )

    supplied = set(
        parameters
    )
    expected = set(
        SINGLE_SERSIC_PARAMETER_NAMES
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
            "Multi-band single-Sersic parameters must be "
            "defined exactly. "
            f"Missing: {missing}. Extra: {extra}."
        )

    specifications = [
        _parse_parameter_specification(
            name=name,
            config=parameters[name],
            bands=bands,
        )
        for name in SINGLE_SERSIC_PARAMETER_NAMES
    ]

    return MultiBandParameterLayout(
        bands=bands,
        specifications=specifications,
    )


def resolve_multiband_config(
    config: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> tuple[
    dict[str, dict[str, Any]],
    MultiBandParameterLayout,
]:
    """Resolve observation configurations and the parameter layout.

    Parameters
    ----------
    config : mapping
        Joint multi-band configuration.
    base_dir : str or pathlib.Path, optional
        Directory used to resolve relative observation configuration paths.

    Returns
    -------
    tuple
        Resolved observation configurations and multi-band parameter layout.
    """
    observation_configs = (
        resolve_multiband_observation_configs(
            config,
            base_dir=base_dir,
        )
    )

    layout = (
        build_multiband_parameter_layout(
            config
        )
    )

    return (
        observation_configs,
        layout,
    )


def _observation_names(
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return validated observation names in YAML order."""
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
            "Multi-band configuration requires a non-empty "
            "'observations' mapping."
        )

    bands = tuple(
        observations
    )

    if any(
        not isinstance(
            band,
            str,
        )
        or not band
        for band in bands
    ):
        raise ValueError(
            "Observation names must be non-empty strings."
        )

    return bands


def _parse_parameter_specification(
    *,
    name: str,
    config: Any,
    bands: tuple[str, ...],
) -> MultiBandParameterSpec:
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
            MultiBandParameterSpec
            .shared_parameter(
                Parameter(
                    name,
                    prior=parse_prior_config(
                        config["prior"]
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

        supplied_bands = set(
            priors
        )
        expected_bands = set(
            bands
        )

        if supplied_bands != expected_bands:
            raise ValueError(
                f"Independent parameter '{name}' must define "
                "exactly the configured observations."
            )

        return (
            MultiBandParameterSpec
            .independent(
                name,
                {
                    band: Parameter(
                        name,
                        prior=parse_prior_config(
                            priors[band]
                        ),
                    )
                    for band in bands
                },
            )
        )

    raise ValueError(
        f"Parameter '{name}' coupling must be "
        "'shared' or 'independent'."
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
