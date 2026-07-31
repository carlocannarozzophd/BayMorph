"""Runtime-report infrastructure for reproducible BayMorph analyses."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from baymorph.pipeline.config import resolve_config
from baymorph.pipeline.outputs import (
    make_output_tree,
    pipeline_paths,
)


def build_runtime_report(
    config: dict[str, Any],
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a serializable report of resolved and runtime information.

    Parameters
    ----------
    config : dict
        Resolved or resolvable BayMorph configuration.
    runtime : dict, optional
        Runtime-derived values produced during pipeline execution.

    Returns
    -------
    dict
        Independent report containing the resolved configuration and runtime
        values.
    """
    return {
        "config": deepcopy(resolve_config(config)),
        "runtime": deepcopy(runtime) if runtime is not None else {},
    }


def save_runtime_report(
    config: dict[str, Any],
    report: dict[str, Any],
) -> Path:
    """Save a runtime report as YAML.

    Parameters
    ----------
    config : dict
        BayMorph pipeline configuration used to resolve the output path.
    report : dict
        Runtime report to serialize.

    Returns
    -------
    pathlib.Path
        Path to the saved runtime report.
    """
    make_output_tree(config)
    path = pipeline_paths(config)["runtime_report"]

    with path.open("w") as stream:
        yaml.safe_dump(
            report,
            stream,
            sort_keys=False,
        )

    return path


def load_runtime_report(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Load a previously saved runtime report.

    Parameters
    ----------
    config : dict
        BayMorph pipeline configuration used to resolve the output path.

    Returns
    -------
    dict
        Loaded runtime report.

    Raises
    ------
    FileNotFoundError
        If the runtime report does not exist.
    ValueError
        If the runtime report is empty.
    TypeError
        If the YAML top level is not a mapping.
    """
    path = pipeline_paths(config)["runtime_report"]

    with path.open("r") as stream:
        report = yaml.safe_load(stream)

    if report is None:
        raise ValueError(f"Runtime report is empty: {path}")

    if not isinstance(report, dict):
        raise TypeError("Runtime report must define a YAML mapping at top level.")

    return report
