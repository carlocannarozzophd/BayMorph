"""Configuration loading, validation, and resolution for BayMorph pipelines."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from baymorph.core.observation import Observation

import yaml


REQUIRED_TOP_LEVEL_KEYS = ("object", "data", "outputs")

DEFAULT_CONFIG = {
    "data": {
        "mask_path": None,
        "noise_map_path": None,
        "psf_path": None,
        "pixel_scale": None,
        "target_position": None,
        "cutout": {
            "mode": "fixed",
            "size": None,
        },
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
                "target_n_candidates": 30,
                "min_area": 10,
                "max_area": 800,
                "min_peak": 20,
                "max_peak": 80000,
                "max_eccentricity": 0.8,
                "max_elongation": 2.5,
                "edge_margin": 30,
                "galaxy_exclusion_box": None,
                "sigma_factor": 4.0,
                "size_margin": 4,
                "min_half_size": 12,
                "max_half_size": 30,
                "isolation_margin": 2,
            },
            "construction": {
                "auto_size": True,
                "normalize": True,
            },
        },
    },
    "fit": {
        "models": ["double_sersic"],
        "sampler": {
            "name": "dynesty",
            "nlive": 500,
            "sample": "auto",
            "seed": None,
            "n_processes": 1,
            "queue_size": None,
        },
        "comparison": {
            "enabled": False,
            "metric": "bayes_factor",
        },
        "priors": {
            "single_sersic": {
                "flux": {
                    "min_factor": 0.0,
                    "max_factor": 2.0,
                },
                "re": {
                    "min": 1.0,
                    "max": 120.0,
                },
                "n": {
                    "min": 0.5,
                    "max": 6.0,
                },
                "q": {
                    "min": 0.4,
                    "max": 1.0,
                },
                "theta": {
                    "min": "-pi/2",
                    "max": "pi/2",
                },
                "x0": {
                    "offset": 8.0,
                },
                "y0": {
                    "offset": 8.0,
                },
                "sky": {
                    "min": -50.0,
                    "max": 50.0,
                },
            },
            "double_sersic": {
                "flux_bulge": {
                    "min_factor": 0.0,
                    "max_factor": 1.0,
                },
                "re_bulge": {
                    "min": 1.0,
                    "max": 30.0,
                },
                "n_bulge": {
                    "min": 1.5,
                    "max": 6.0,
                },
                "flux_disk": {
                    "min_factor": 0.0,
                    "max_factor": 2.0,
                },
                "re_disk": {
                    "min": 10.0,
                    "max": 120.0,
                },
                "n_disk": {
                    "min": 0.5,
                    "max": 3.0,
                },
                "q": {
                    "min": 0.4,
                    "max": 1.0,
                },
                "theta": {
                    "min": "-pi/2",
                    "max": "pi/2",
                },
                "x0": {
                    "offset": 8.0,
                },
                "y0": {
                    "offset": 8.0,
                },
                "sky": {
                    "min": -50.0,
                    "max": 50.0,
                },
            },
        },
    },
    "diagnostics": {
        "format": "pdf",
        "corner": True,
        "residual": True,
        "radial_profile": True,
        "posterior_profile_bands": True,
    },
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a BayMorph pipeline configuration from a YAML file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Raw configuration dictionary loaded from disk.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the YAML file is empty.
    TypeError
        If the YAML top level is not a mapping.
    """
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
    """Validate required top-level BayMorph configuration entries.

    Parameters
    ----------
    config : dict
        Raw or partially resolved configuration dictionary.

    Raises
    ------
    ValueError
        If a required top-level section or required entry is missing.
    """
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
    """Merge a user configuration with BayMorph defaults.

    Parameters
    ----------
    config : dict
        User configuration dictionary.

    Returns
    -------
    dict
        Fully resolved configuration dictionary containing default values for
        omitted optional sections.

    Raises
    ------
    ValueError
        If the input configuration is missing required entries.
    """
    validate_config(config)

    resolved = deep_update(
        deepcopy(DEFAULT_CONFIG),
        config,
    )

    resolved["object"] = dict(resolved["object"])
    resolved["data"] = dict(resolved["data"])
    resolved["outputs"] = dict(resolved["outputs"])

    resolved["object"]["name"] = str(resolved["object"]["name"])
    resolved["data"]["image_path"] = str(Path(resolved["data"]["image_path"]))
    resolved["outputs"]["root"] = str(Path(resolved["outputs"]["root"]))

    data_cfg = resolved["data"]

    if (
        "cutout" not in data_cfg
        and data_cfg.get("cutout_size") is not None
    ):
        data_cfg["cutout"] = {
            "mode": "fixed",
            "size": data_cfg["cutout_size"],
        }

    return resolved


def resolve_runtime_config(
    config: dict[str, Any],
    observation: Observation,
) -> dict[str, Any]:
    """Resolve configuration entries that require image metadata.

    Parameters
    ----------
    config : dict
        Previously resolved configuration dictionary.
    observation : Observation
        Loaded observation providing runtime metadata.

    Returns
    -------
    dict
        Runtime-resolved configuration dictionary.

    Notes
    -----
    This function resolves configuration options that cannot be determined
    from the YAML file alone because they require information contained in the
    loaded observation.
    """
    resolved = deepcopy(config)

    target = resolved["data"].get("target_position")

    if target == "center":
        resolved["data"]["target_position"] = {
            "x": observation.metadata.center_x,
            "y": observation.metadata.center_y,
        }

    return resolved


def load_resolved_config(path: str | Path) -> dict[str, Any]:
    """Load, validate, and resolve a BayMorph YAML configuration.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Resolved configuration dictionary.
    """
    return resolve_config(load_config(path))


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a nested dictionary.

    Parameters
    ----------
    base : dict
        Base dictionary.
    updates : dict
        Dictionary containing replacement or nested update values.

    Returns
    -------
    dict
        Updated dictionary. Nested dictionaries are recursively merged, while
        non-dictionary values replace the corresponding base value.
    """
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