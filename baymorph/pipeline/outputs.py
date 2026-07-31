"""Output-path management for BayMorph pipeline runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from baymorph.pipeline.config import resolve_config


def object_output_dir(config: dict[str, Any]) -> Path:
    """Return the base output directory for one target object.

    Parameters
    ----------
    config : dict
        Resolved or resolvable pipeline configuration.

    Returns
    -------
    pathlib.Path
        Output directory for the configured object.
    """
    root = Path(config["outputs"]["root"])
    object_name = config["object"]["name"]
    return root / object_name


def make_output_tree(config: dict[str, Any]) -> Path:
    """Create the standard BayMorph output directory tree.

    Parameters
    ----------
    config : dict
        Pipeline configuration.

    Returns
    -------
    pathlib.Path
        Base output directory for the configured object.
    """
    config = resolve_config(config)

    base = object_output_dir(config)

    subdirs = [
        base,
        base / "input",
        base / "image_preparation",
        base / "psf",
        base / "fits",
        base / "model_comparison",
    ]

    subdirs.extend(
        base / "fits" / model_name
        for model_name in config["fit"]["models"]
    )

    for path in subdirs:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    return base


def save_resolved_config(config: dict[str, Any]) -> Path:
    """Save the resolved pipeline configuration to disk.

    Parameters
    ----------
    config : dict
        Resolved or resolvable pipeline configuration.

    Returns
    -------
    pathlib.Path
        Path to the saved resolved configuration file.
    """
    make_output_tree(config)
    path = pipeline_paths(config)["config_resolved"]

    with path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return path


def diagnostic_format(config: dict[str, Any]) -> str:
    """Return the configured diagnostic-plot file format.

    Parameters
    ----------
    config : dict
        Pipeline configuration.

    Returns
    -------
    str
        Lowercase diagnostic file extension.
    """
    return str(
        config.get(
            "diagnostics",
            {},
        ).get(
            "format",
            "pdf",
        )
    ).lower()


def pipeline_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Build the standard output-path dictionary for a pipeline run.

    Parameters
    ----------
    config : dict
        Pipeline configuration.

    Returns
    -------
    dict
        Nested dictionary containing output directories and file paths for
        image preparation, PSF construction, fitting, diagnostics, and model
        comparison products.
    """
    config = resolve_config(config)

    base = object_output_dir(config)

    plot_format = diagnostic_format(config)

    input_image = base / "input" / Path(
        config["data"]["image_path"]
    ).name

    fit_paths = {}

    for model_name in config["fit"]["models"]:
        fit_dir = base / "fits" / model_name

        fit_paths[model_name] = {
            "dir": fit_dir,
            "fit_results": fit_dir / "fit_results.json",
            "corner": fit_dir / f"corner.{plot_format}",
            "residual": fit_dir / f"data_model_residual.{plot_format}",
            "radial_profile": fit_dir / f"radial_profile.{plot_format}",
        }

    return {
        "base": base,
        "config_resolved": base / "config_resolved.yaml",
        "runtime_report": base / "runtime_report.yaml",
        "input": {
            "dir": base / "input",
            "image": input_image,
        },
        "image_preparation": {
            "dir": base / "image_preparation",
            "prepared_image": base / "image_preparation" / "prepared_image.fits",
            "noise_map": base / "image_preparation" / "noise_map.fits",
            "mask": base / "image_preparation" / "mask.fits",
        },
        "psf": {
            "dir": base / "psf",
            "psf_fits": base / "psf" / "psf.fits",
            "metadata": base / "psf" / "psf_metadata.json",
            "candidates": base / "psf" / "psf_candidates.json",
            "candidates_plot": base / "psf" / f"psf_candidates.{plot_format}",
            "diagnostics_plot": base / "psf" / f"psf_diagnostics.{plot_format}",
        },
        "fits": {
            "dir": base / "fits",
            **fit_paths,
        },
        "model_comparison": {
            "dir": base / "model_comparison",
            "bayes_factor": base / "model_comparison" / "bayes_factor.json",
            "summary": base / "model_comparison" / "summary.txt",
        },
    }