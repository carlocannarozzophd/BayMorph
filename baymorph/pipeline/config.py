from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL_KEYS = ("object", "data", "outputs")

DEFAULT_CONFIG = {
    "data": {
        "mask_path": None,
        "noise_map_path": None,
        "psf_path": None,
        "pixel_scale": None,
        "target_position": None,
        "cutout_size": None,
    },
    "outputs": {
        "overwrite": False,
    },
    "preparation": {
        "background": {
            "method": "empirical",
            "box_size": 30,
            "filter_size": 3,
        },
        "noise": {
            "method": "empirical",
        },
        "detection": {
            "n_sigma": 3.0,
            "n_levels": 32,
            "npixels": 20,
        },
    },
    "masking": {
        "connected_source": {
            "enabled": True,
        },
        "star_halo": {
            "enabled": True,
        },
        "interactive_refinement": {
            "enabled": True,
        },
    },
    "psf": {
        "type": "empirical",
        "empirical": {
            "candidate_selection": {
                "auto": True,
                "target_n_candidates": 15,
                "min_area": 20,
                "max_area": 300,
                "min_peak": 50,
                "max_peak": 50000,
                "max_eccentricity": 0.50,
                "max_elongation": 1.50,
                "edge_margin": 80,
                "galaxy_exclusion_box": None,
                "sigma_factor": 4.0,
                "size_margin": 3,
                "min_half_size": 12,
                "max_half_size": 25,
                "isolation_margin": 5,
            },
            "construction": {
                "auto_size": True,
                "normalize": True,
            },
        },
    },
    "fit": {
        "models": ["single_sersic"],
        "sampler": {
            "name": "dynesty",
            "nlive": 500,
        },
        "comparison": {
            "enabled": False,
            "metric": "bayes_factor",
        },
    },
    "diagnostics": {
        "corner": True,
        "residual": True,
        "radial_profile": True,
        "posterior_profile_bands": True,
    },
}

def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    if not isinstance(config, dict):
        raise TypeError("Config file must define a YAML mapping at top level.")

    return config


def validate_config(config: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    if "name" not in config["object"]:
        raise ValueError("Config section 'object' must contain 'name'.")

    if "image_path" not in config["data"]:
        raise ValueError("Config section 'data' must contain 'image_path'.")

    if "root" not in config["outputs"]:
        raise ValueError("Config section 'outputs' must contain 'root'.")


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)

    resolved = deep_update(DEFAULT_CONFIG, config)

    resolved["object"] = dict(resolved["object"])
    resolved["data"] = dict(resolved["data"])
    resolved["outputs"] = dict(resolved["outputs"])

    resolved["object"]["name"] = str(resolved["object"]["name"])
    resolved["data"]["image_path"] = str(Path(resolved["data"]["image_path"]))
    resolved["outputs"]["root"] = str(Path(resolved["outputs"]["root"]))

    return resolved


def load_resolved_config(path: str | Path) -> dict[str, Any]:
    return resolve_config(load_config(path))


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)

    for key, value in updates.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value

    return result