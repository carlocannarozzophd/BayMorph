"""Configuration parsing for production multi-observation inference."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from baymorph.fitting.multi_observation import (
    MultiObservationParameterLayout,
    MultiObservationParameterSpec,
)
from baymorph.fitting.observation_covariates import (
    wavelength_covariate,
)
from baymorph.fitting.parameters import Parameter
from baymorph.models import get_model
from baymorph.models.geometry import sky_parameter_names
from baymorph.observation.background import background_parameter_names
from baymorph.pipeline.config import (
    load_config,
    resolve_config,
)
from baymorph.pipeline.prior_config import (
    _require_exact_keys,
    parse_prior_config,
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

    fit_config = config.get(
        "fit",
        {},
    )

    if not isinstance(
        fit_config,
        Mapping,
    ):
        raise TypeError(
            "Multi-observation fit configuration must be a mapping."
        )

    global_likelihood = fit_config.get(
        "likelihood"
    )

    if "likelihood" in fit_config and global_likelihood is None:
        raise ValueError("Multi-observation fit.likelihood cannot be null.")

    if (
        global_likelihood is not None
        and not isinstance(
            global_likelihood,
            Mapping,
        )
    ):
        raise TypeError(
            "Multi-observation fit.likelihood must be a mapping."
        )

    if global_likelihood is not None:
        _joint_likelihood(
            global_likelihood,
            "Multi-observation fit.likelihood",
        )

    background_model = multi_observation_background_model(
        config
    )

    resolved: dict[
        str,
        dict[str, Any],
    ] = {}
    config_owners: dict[Path, str] = {}

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

        supplied_keys = set(
            observation_config
        )
        allowed_keys = {
            "config_path",
            "covariates",
        }

        if (
            "config_path" not in supplied_keys
            or not supplied_keys.issubset(
                allowed_keys
            )
        ):
            missing = (
                []
                if "config_path" in supplied_keys
                else ["config_path"]
            )
            extra = sorted(
                supplied_keys.difference(
                    allowed_keys
                )
            )

            raise ValueError(
                f"Observation '{observation}' configuration keys "
                f"are invalid. Missing: {missing}. Extra: {extra}."
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

        config_path = config_path.expanduser().resolve(strict=False)
        _claim_path(
            config_path,
            observation,
            config_owners,
            "child configuration paths",
        )

        raw_observation_config = load_config(
            config_path
        )
        observation_fit = raw_observation_config.get(
            "fit",
            {},
        )

        if not isinstance(
            observation_fit,
            Mapping,
        ):
            raise TypeError(
                f"Observation '{observation}' fit configuration "
                "must be a mapping."
            )

        if "likelihood" in observation_fit:
            child_likelihood = observation_fit["likelihood"]
            if child_likelihood is None:
                raise ValueError(
                    f"Observation '{observation}' fit.likelihood "
                    "cannot be null."
                )
            if not isinstance(child_likelihood, Mapping):
                raise TypeError(
                    f"Observation '{observation}' fit.likelihood "
                    "must be a mapping."
                )
            _joint_likelihood(
                child_likelihood,
                f"Observation '{observation}' fit.likelihood",
            )
        elif global_likelihood is not None:
            raw_observation_config = deepcopy(
                raw_observation_config
            )
            observation_fit = dict(
                observation_fit
            )
            observation_fit[
                "likelihood"
            ] = deepcopy(
                global_likelihood
            )
            raw_observation_config[
                "fit"
            ] = observation_fit

        _require_consistent_child_background(
            observation_fit,
            observation=observation,
            background_model=background_model,
        )

        item = resolve_config(raw_observation_config)
        _joint_likelihood(
            item["fit"]["likelihood"],
            f"Observation '{observation}' resolved fit.likelihood",
        )
        resolved[observation] = item

    _unique_resources(resolved)
    return resolved


def _joint_likelihood(value, context):
    kind = value.get("type")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError(
            f"{context} must explicitly define a non-empty 'type'."
        )
    if "noise" in value:
        raise ValueError(
            f"{context} cannot define 'noise'."
        )


def multi_observation_background_model(
    config: Mapping[str, Any],
) -> str:
    """Return the residual-background family of one joint configuration.

    The joint ``fit.background`` section declares only the background
    model family shared by every observation. The per-observation
    background priors are declared, like every other sampled parameter,
    under ``fit.multi_observation.parameters`` with independent coupling
    (AD-031, AD-044).
    """
    fit_config = config.get(
        "fit",
        {},
    )

    if not isinstance(
        fit_config,
        Mapping,
    ):
        raise TypeError(
            "Multi-observation fit configuration must be a mapping."
        )

    background = fit_config.get(
        "background"
    )

    if background is None:
        if "background" in fit_config:
            raise TypeError(
                "Multi-observation fit.background must be a mapping."
            )

        return "constant"

    if not isinstance(
        background,
        Mapping,
    ):
        raise TypeError(
            "Multi-observation fit.background must be a mapping."
        )

    unsupported = sorted(
        set(
            background
        ).difference(
            {
                "model",
            }
        )
    )

    if unsupported:
        raise ValueError(
            "Multi-observation fit.background accepts only 'model'. "
            "Per-observation background priors belong under "
            "fit.multi_observation.parameters with independent coupling. "
            f"Unsupported entries: {unsupported}."
        )

    model = str(
        background.get(
            "model",
            "constant",
        )
    ).strip().lower()

    background_parameter_names(
        model
    )

    return model


def _require_consistent_child_background(
    observation_fit: Mapping[str, Any],
    *,
    observation: str,
    background_model: str,
) -> None:
    """Reject a child configuration declaring a different background family."""
    background = observation_fit.get(
        "background"
    )

    if background is None:
        return

    if not isinstance(
        background,
        Mapping,
    ):
        raise TypeError(
            f"Observation '{observation}' fit.background must be a mapping."
        )

    child_model = background.get(
        "model"
    )

    if child_model is None:
        return

    child_model = str(
        child_model
    ).strip().lower()

    if child_model != background_model:
        raise ValueError(
            f"Observation '{observation}' declares fit.background.model "
            f"{child_model!r} but the joint configuration uses "
            f"{background_model!r}. One joint problem has one background "
            "family; declare it in the joint fit.background section."
        )


def _path(value, context):
    if (
        not isinstance(value, (str, Path))
        or not str(value).strip()
    ):
        raise ValueError(
            f"{context} must be a non-empty path."
        )
    return Path(value).expanduser().resolve(strict=False)


def _claim_path(path, observation, owners, resource):
    previous = owners.get(path)
    if previous is not None:
        raise ValueError(
            f"Observations must use distinct {resource}; "
            f"'{previous}' and '{observation}' both resolve to {path}."
        )
    owners[path] = observation


def _unique_resources(configs):
    images = {}
    outputs = {}

    for observation, config in configs.items():
        image = _path(
            config["data"]["image_path"],
            f"Observation '{observation}' data.image_path",
        )
        _claim_path(
            image,
            observation,
            images,
            "FITS image paths",
        )

        root = _path(
            config["outputs"]["root"],
            f"Observation '{observation}' outputs.root",
        )
        name = str(config["object"]["name"]).strip()
        if not name:
            raise ValueError(
                f"Observation '{observation}' object.name "
                "must be non-empty."
            )

        namespace = (root / name).resolve(strict=False)
        _claim_path(
            namespace,
            observation,
            outputs,
            "preparation output namespaces",
        )


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

    if model_definition.geometry is None:
        raise ValueError(
            f"Model {model_name!r} has no geometry contract, so the "
            "multi-observation workflow does not support it yet; fit it to "
            "one observation at a time (AD-048)."
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
        morphology_names = sky_parameter_names(
            model_name
        )
    else:
        morphology_names = tuple(
            model_definition.parameter_names
        )

    background_names = background_parameter_names(
        multi_observation_background_model(
            config
        )
    )

    parameter_names = (
        tuple(
            morphology_names
        )
        + background_names
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

        hint = ""

        if "sky" in supplied:
            hint = (
                " The legacy 'sky' parameter is no longer supported: "
                "declare the residual background as 'background_level' "
                "with independent coupling (AD-044)."
            )

        raise ValueError(
            "Multi-observation model parameters must be "
            "defined exactly. "
            f"Missing: {missing}. Extra: {extra}."
            + hint
        )

    specifications = []

    for name in parameter_names:
        specification = _parse_parameter_specification(
            name=name,
            config=parameters[
                name
            ],
            observations=observations,
        )

        if (
            name in background_names
            and specification.coupling != "independent"
        ):
            raise ValueError(
                f"Residual-background parameter '{name}' must use "
                "'independent' coupling: each observation retains its own "
                "residual background (AD-044)."
            )

        specifications.append(
            specification
        )

    uses_wavelength_coupling = any(
        specification.coupling == "wavelength"
        for specification in specifications
    )

    wavelength_coordinates = None

    if uses_wavelength_coupling:
        observation_entries = config.get(
            "observations"
        )

        assert isinstance(
            observation_entries,
            Mapping,
        )

        wavelength_coordinates = wavelength_covariate(
            observation_entries
        )

    return MultiObservationParameterLayout(
        observations=observations,
        specifications=specifications,
        wavelength_coordinates=wavelength_coordinates,
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

    if coupling == "hierarchical":
        _require_exact_keys(
            config,
            {
                "coupling",
                "prior",
            },
            context=(
                f"Hierarchical parameter "
                f"'{name}'"
            ),
        )

        return MultiObservationParameterSpec.hierarchical(
            name,
            prior=parse_prior_config(
                config["prior"]
            ),
        )

    if coupling == "wavelength":
        _require_exact_keys(
            config,
            {
                "coupling",
                "prior",
            },
            context=(
                f"Wavelength parameter "
                f"'{name}'"
            ),
        )

        return MultiObservationParameterSpec.wavelength(
            name,
            prior=parse_prior_config(
                config["prior"]
            ),
        )

    raise ValueError(
        f"Parameter '{name}' coupling must be "
        "'shared', 'independent', 'hierarchical', or "
        "'wavelength'."
    )


__all__ = [
    "build_multi_observation_parameter_layout",
    "multi_observation_background_model",
    "multi_observation_parameter_frame",
    "parse_prior_config",
    "resolve_multi_observation_config",
    "resolve_multi_observation_observation_configs",
]
