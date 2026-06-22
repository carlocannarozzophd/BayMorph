from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def object_output_dir(config: dict[str, Any]) -> Path:
    root = Path(config["outputs"]["root"])
    object_name = config["object"]["name"]
    return root / object_name


def make_output_tree(config: dict[str, Any]) -> Path:
    base = object_output_dir(config)

    subdirs = [
        base,
        base / "image_preparation",
        base / "psf",
        base / "fits",
        base / "fits" / "single_sersic",
        base / "fits" / "double_sersic",
        base / "model_comparison",
    ]

    for path in subdirs:
        path.mkdir(parents=True, exist_ok=True)

    return base


def save_resolved_config(config: dict[str, Any]) -> Path:
    make_output_tree(config)
    path = pipeline_paths(config)["config_resolved"]

    with path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return path


def diagnostic_format(config: dict[str, Any]) -> str:
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
    base = object_output_dir(config)

    plot_format = diagnostic_format(config)

    return {
        "base": base,
        "config_resolved": base / "config_resolved.yaml",
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
            "single_sersic": {
                "dir": base / "fits" / "single_sersic",
                "fit_results": base / "fits" / "single_sersic" / "fit_results.json",
                "corner": base / "fits" / "single_sersic" / f"corner.{plot_format}",
                "residual": base / "fits" / "single_sersic" / f"data_model_residual.{plot_format}",
                "radial_profile": base / "fits" / "single_sersic" / f"radial_profile.{plot_format}",
            },
            "double_sersic": {
                "dir": base / "fits" / "double_sersic",
                "fit_results": base / "fits" / "double_sersic" / "fit_results.json",
                "corner": base / "fits" / "double_sersic" / f"corner.{plot_format}",
                "residual": base / "fits" / "double_sersic" / f"data_model_residual.{plot_format}",
                "radial_profile": base / "fits" / "double_sersic" / f"radial_profile.{plot_format}",
            },
        },
        "model_comparison": {
            "dir": base / "model_comparison",
            "bayes_factor": base / "model_comparison" / "bayes_factor.json",
            "summary": base / "model_comparison" / "summary.txt",
        },
    }