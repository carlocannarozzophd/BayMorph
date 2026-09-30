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


def _masked_fraction(mask) -> float | None:
    """Return the fraction of excluded pixels in an optional mask."""
    if mask is None:
        return None

    import numpy as np

    array = np.asarray(mask, dtype=bool)

    if array.ndim != 2:
        raise ValueError("Runtime-report masks must be two-dimensional.")

    if array.size == 0:
        return 0.0

    return float(np.mean(array))


def build_image_preparation_runtime(
    products,
    *,
    observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build runtime metadata from prepared-observation products.

    Parameters
    ----------
    products : object
        Prepared-observation object exposing BayMorph image-preparation
        products.
    observation : dict, optional
        Normalized observational metadata.

    Returns
    -------
    dict
        Serializable runtime information for image preparation.
    """
    import numpy as np

    noise = np.asarray(
        products.noise,
        dtype=float,
    )
    finite_noise = noise[np.isfinite(noise)]

    noise_summary = {
        "median": (float(np.median(finite_noise)) if finite_noise.size else None),
        "minimum": (float(np.min(finite_noise)) if finite_noise.size else None),
        "maximum": (float(np.max(finite_noise)) if finite_noise.size else None),
    }

    geometry = (
        {
            "x0": float(products.geometry.x0),
            "y0": float(products.geometry.y0),
            "q": float(products.geometry.q),
            "theta": float(products.geometry.theta),
        }
        if products.geometry is not None
        else None
    )

    plateau = (
        {
            "radius": float(products.plateau.radius),
            "index": int(products.plateau.index),
            "background_level": float(products.plateau.background_level),
            "noise_sigma": float(products.plateau.noise_sigma),
            "window_start": int(products.plateau.window_start),
            "window_stop": int(products.plateau.window_stop),
        }
        if products.plateau is not None
        else None
    )

    sky_region = (
        {
            "inner_radius": float(products.sky_region.inner_radius),
            "outer_radius": float(products.sky_region.outer_radius),
            "n_pixels": int(products.sky_region.n_pixels),
            "coverage": float(products.sky_region.coverage),
        }
        if products.sky_region is not None
        else None
    )

    return {
        "observation": deepcopy(observation),
        "image_preparation": {
            "cutout": {
                "mode": str(products.cutout_mode),
                "shape": [
                    int(products.prepared_data.shape[0]),
                    int(products.prepared_data.shape[1]),
                ],
                "bounds": deepcopy(products.cutout_bounds),
            },
            "background": {
                "level": float(products.background),
                # Diagnostic only. A gradient significant at several sigma
                # indicates that fit.background.model should be "plane"; the
                # choice stays with the user.
                "sky_gradient": deepcopy(products.sky_gradient),
            },
            "noise": noise_summary,
            "target": {
                "x": int(products.target_x),
                "y": int(products.target_y),
                "label": int(products.target_label),
                "geometry": geometry,
            },
            "plateau": plateau,
            "sky_region": sky_region,
            "masking": {
                "convention": "true_is_excluded",
                "automatic_fraction": _masked_fraction(products.automatic_mask),
                "external_fraction": _masked_fraction(products.external_mask),
                "contaminant_fraction": _masked_fraction(products.contaminant_mask),
                "fitting_fraction": _masked_fraction(products.fitting_mask),
            },
        },
    }
