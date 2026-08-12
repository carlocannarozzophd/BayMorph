"""Output-path management for multi-observation analyses."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def multi_observation_output_dir(
    config: Mapping[str, Any],
) -> Path:
    """Return the base output directory for a multi-observation analysis."""
    outputs = config.get(
        "outputs"
    )

    if not isinstance(
        outputs,
        Mapping,
    ):
        raise ValueError(
            "Multi-observation configuration requires an 'outputs' mapping."
        )

    root = outputs.get(
        "root"
    )
    analysis_name = outputs.get(
        "analysis_name"
    )

    if not isinstance(
        root,
        (str, Path),
    ) or not str(root):
        raise ValueError(
            "Multi-observation outputs.root must be a non-empty path."
        )

    _validate_path_component(
        analysis_name,
        context="outputs.analysis_name",
    )

    return (
        Path(root)
        / str(analysis_name)
    )


def multi_observation_diagnostic_format(
    config: Mapping[str, Any],
) -> str:
    """Return the diagnostic-plot format for a multi-observation analysis."""
    diagnostics = config.get(
        "diagnostics",
        {},
    )

    if not isinstance(
        diagnostics,
        Mapping,
    ):
        raise TypeError(
            "Multi-observation diagnostics must be a mapping."
        )

    plot_format = str(
        diagnostics.get(
            "format",
            "pdf",
        )
    ).strip().lower()

    if not plot_format:
        raise ValueError(
            "Diagnostic format must be non-empty."
        )

    return plot_format


def multi_observation_paths(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build paths for joint and observation-specific fit products."""
    base = multi_observation_output_dir(
        config
    )

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

    observation_names = tuple(
        observations
    )

    for name in observation_names:
        _validate_path_component(
            name,
            context="Observation identifier",
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
        or not models
    ):
        raise ValueError(
            "Multi-observation fit configuration requires a non-empty "
            "'models' list."
        )

    plot_format = (
        multi_observation_diagnostic_format(
            config
        )
    )

    fit_paths = {}

    for model_name in models:
        _validate_path_component(
            model_name,
            context="Model name",
        )

        fit_dir = (
            base
            / "fits"
            / model_name
        )

        observation_paths = {}

        for observation_name in observation_names:
            observation_dir = (
                fit_dir
                / "observations"
                / observation_name
            )

            observation_paths[
                observation_name
            ] = {
                "dir": observation_dir,
                "best_model_image": (
                    observation_dir
                    / "best_model_image.npy"
                ),
                "residual": (
                    observation_dir
                    / "residual.npy"
                ),
                "norm_residual": (
                    observation_dir
                    / "norm_residual.npy"
                ),
                "summary": (
                    observation_dir
                    / "fit_summary.json"
                ),
                "residual_plot": (
                    observation_dir
                    / f"data_model_residual.{plot_format}"
                ),
                "radial_profile": (
                    observation_dir
                    / f"radial_profile.{plot_format}"
                ),
            }

        fit_paths[
            model_name
        ] = {
            "dir": fit_dir,
            "posterior": (
                fit_dir
                / "posterior.npz"
            ),
            "best_parameters": (
                fit_dir
                / "best_parameters.json"
            ),
            "fit_summary": (
                fit_dir
                / "fit_summary.json"
            ),
            "corner": (
                fit_dir
                / f"corner.{plot_format}"
            ),
            "observations_dir": (
                fit_dir
                / "observations"
            ),
            "observations": observation_paths,
        }

    return {
        "base": base,
        "fits": {
            "dir": (
                base
                / "fits"
            ),
            **fit_paths,
        },
    }


def make_multi_observation_output_tree(
    config: Mapping[str, Any],
) -> Path:
    """Create the standard multi-observation analysis output tree."""
    paths = multi_observation_paths(
        config
    )

    directories = [
        paths["base"],
        paths["fits"]["dir"],
    ]

    fit_config = config[
        "fit"
    ]

    for model_name in fit_config[
        "models"
    ]:
        fit_paths = paths[
            "fits"
        ][
            model_name
        ]

        directories.extend(
            [
                fit_paths["dir"],
                fit_paths[
                    "observations_dir"
                ],
            ]
        )

        directories.extend(
            observation_paths[
                "dir"
            ]
            for observation_paths in fit_paths[
                "observations"
            ].values()
        )

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return paths[
        "base"
    ]


def _validate_path_component(
    value: Any,
    *,
    context: str,
) -> None:
    """Require a non-empty value that is safe as one filesystem component."""
    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise ValueError(
            f"{context} must be a non-empty string."
        )

    value = value.strip()

    if (
        value in {
            ".",
            "..",
        }
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(
            f"{context} must be a single safe filesystem component."
        )
