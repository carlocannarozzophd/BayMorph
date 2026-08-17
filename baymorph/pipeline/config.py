"""Configuration loading, validation, and resolution for BayMorph pipelines."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from baymorph.core.observation import Observation
from baymorph.fitting.execution import resolve_sampler_execution

REQUIRED_TOP_LEVEL_KEYS = ("object", "data", "outputs")

SAMPLER_DEFAULTS = {
    "pocomc": {
        "name": "pocomc",
        "execution": {
            "processes": "auto",
            "progress": True,
        },
        "init": {},
        "run": {},
    },
    "dynesty": {
        "name": "dynesty",
        "mode": "static",
        "execution": {
            "processes": "auto",
            "progress": True,
        },
        "init": {},
        "run": {},
    },
    "nautilus": {
        "name": "nautilus",
        "execution": {
            "processes": "auto",
            "progress": True,
        },
        "init": {},
        "run": {},
    },
}


DEFAULT_CONFIG = {
    "data": {
        "mask_path": None,
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
            "box_size": 30,
            "filter_size": 3,
        },
        "noise": {},
        "detection": {
            "n_sigma": 3.0,
            "n_levels": 32,
            "n_pixels": 20,
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
        },
    },
    "fit": {
        "models": ["double_sersic"],
        "likelihood": {
            "type": "gaussian",
            "include_normalization": True,
        },
        "sampler": {
            "name": "pocomc",
        },
        "comparison": {
            "enabled": False,
            "metric": "bayes_factor",
        },
        "background": {
            "model": "constant",
            "priors": {
                "background_level": {
                    "type": "uniform",
                    "lower": -50.0,
                    "upper": 50.0,
                },
                "background_gradient_x": {
                    "type": "uniform",
                    "lower": -0.1,
                    "upper": 0.1,
                },
                "background_gradient_y": {
                    "type": "uniform",
                    "lower": -0.1,
                    "upper": 0.1,
                },
            },
        },
        "priors": {
            "single_sersic": {
                "re": {
                    "type": "uniform",
                    "lower": 1.0,
                    "upper": 120.0,
                },
                "n": {
                    "type": "uniform",
                    "lower": 0.5,
                    "upper": 6.0,
                },
                "q": {
                    "type": "uniform",
                    "lower": 0.4,
                    "upper": 1.0,
                },
                "theta": {
                    "type": "uniform",
                    "lower": "-pi/2",
                    "upper": "pi/2",
                },
                "x0": {
                    "type": "uniform",
                    "lower": -8.0,
                    "upper": 8.0,
                    "relative_to": "target_x",
                },
                "y0": {
                    "type": "uniform",
                    "lower": -8.0,
                    "upper": 8.0,
                    "relative_to": "target_y",
                },
                "sky": {
                    "type": "uniform",
                    "lower": -50.0,
                    "upper": 50.0,
                },
            },
            "double_sersic": {
                "re_bulge": {
                    "type": "uniform",
                    "lower": 1.0,
                    "upper": 30.0,
                },
                "n_bulge": {
                    "type": "uniform",
                    "lower": 1.5,
                    "upper": 6.0,
                },
                "re_disk": {
                    "type": "uniform",
                    "lower": 10.0,
                    "upper": 120.0,
                },
                "n_disk": {
                    "type": "uniform",
                    "lower": 0.5,
                    "upper": 3.0,
                },
                "q": {
                    "type": "uniform",
                    "lower": 0.4,
                    "upper": 1.0,
                },
                "theta": {
                    "type": "uniform",
                    "lower": "-pi/2",
                    "upper": "pi/2",
                },
                "x0": {
                    "type": "uniform",
                    "lower": -8.0,
                    "upper": 8.0,
                    "relative_to": "target_x",
                },
                "y0": {
                    "type": "uniform",
                    "lower": -8.0,
                    "upper": 8.0,
                    "relative_to": "target_y",
                },
                "sky": {
                    "type": "uniform",
                    "lower": -50.0,
                    "upper": 50.0,
                },
            },
            "sersic_point_source": {
                "re": {
                    "type": "uniform",
                    "lower": 1.0,
                    "upper": 120.0,
                },
                "n": {
                    "type": "uniform",
                    "lower": 0.5,
                    "upper": 6.0,
                },
                "q": {
                    "type": "uniform",
                    "lower": 0.4,
                    "upper": 1.0,
                },
                "theta": {
                    "type": "uniform",
                    "lower": "-pi/2",
                    "upper": "pi/2",
                },
                "x0": {
                    "type": "uniform",
                    "lower": -8.0,
                    "upper": 8.0,
                    "relative_to": "target_x",
                },
                "y0": {
                    "type": "uniform",
                    "lower": -8.0,
                    "upper": 8.0,
                    "relative_to": "target_y",
                },
                "sky": {
                    "type": "uniform",
                    "lower": -50.0,
                    "upper": 50.0,
                },
            },
            "bulge_disk": {
                "re_bulge": {
                    "type": "uniform",
                    "lower": 1.0,
                    "upper": 30.0,
                },
                "n_bulge": {
                    "type": "uniform",
                    "lower": 1.5,
                    "upper": 6.0,
                },
                "q_bulge": {
                    "type": "uniform",
                    "lower": 0.4,
                    "upper": 1.0,
                },
                "theta_bulge": {
                    "type": "uniform",
                    "lower": "-pi/2",
                    "upper": "pi/2",
                },
                "re_disk": {
                    "type": "uniform",
                    "lower": 10.0,
                    "upper": 120.0,
                },
                "q_disk": {
                    "type": "uniform",
                    "lower": 0.4,
                    "upper": 1.0,
                },
                "theta_disk": {
                    "type": "uniform",
                    "lower": "-pi/2",
                    "upper": "pi/2",
                },
                "x0": {
                    "type": "uniform",
                    "lower": -8.0,
                    "upper": 8.0,
                    "relative_to": "target_x",
                },
                "y0": {
                    "type": "uniform",
                    "lower": -8.0,
                    "upper": 8.0,
                    "relative_to": "target_y",
                },
                "sky": {
                    "type": "uniform",
                    "lower": -50.0,
                    "upper": 50.0,
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

    unsupported_data_keys = sorted(
        {
            "noise_map_path",
            "psf_path",
        }
        & config["data"].keys()
    )

    if unsupported_data_keys:
        raise ValueError(
            "Unsupported configuration entries in section 'data': "
            f"{unsupported_data_keys}. "
            "External noise maps and direct PSF paths are not yet "
            "supported by the single-band production workflow."
        )

    if "root" not in config["outputs"]:
        raise ValueError("Config section 'outputs' must contain 'root'.")

    if "masking" in config:
        raise ValueError(
            "Unsupported top-level configuration section 'masking'. "
            "The current single-band workflow constructs its automatic "
            "fitting mask during image preparation and does not expose "
            "masking strategy toggles."
        )

    psf_construction_config = (
        config
        .get("psf", {})
        .get("empirical", {})
        .get("construction", {})
    )

    if "auto_size" in psf_construction_config:
        raise ValueError(
            "Unsupported configuration key "
            "'psf.empirical.construction.auto_size'. "
            "Empirical PSF sizing is controlled by candidate-selection "
            "cutout limits."
        )

    if "normalize" in psf_construction_config:
        raise ValueError(
            "Unsupported configuration key "
            "'psf.empirical.construction.normalize'. "
            "BayMorph always normalizes PSF kernels to unit sum."
        )

    background_config = (
        config
        .get("fit", {})
        .get("background", {})
    )

    if "model" in background_config:
        background_model = str(
            background_config["model"]
        ).strip().lower()

        if background_model not in {
            "constant",
            "plane",
        }:
            raise ValueError(
                "Unsupported fit.background.model "
                f"{background_model!r}. "
                "Available models: 'constant', 'plane'."
            )

    preparation_config = config.get("preparation", {})

    unsupported_preparation_entries = []

    if "method" in preparation_config.get("background", {}):
        unsupported_preparation_entries.append(
            "preparation.background.method"
        )

    if "method" in preparation_config.get("noise", {}):
        unsupported_preparation_entries.append(
            "preparation.noise.method"
        )

    if unsupported_preparation_entries:
        raise ValueError(
            "Unsupported fixed configuration entries: "
            f"{unsupported_preparation_entries}. "
            "The single-band production workflow currently uses empirical "
            "background estimation and MAD-based noise estimation."
        )

    detection_config = preparation_config.get("detection", {})

    if "npixels" in detection_config:
        raise ValueError(
            "Unsupported configuration key "
            "'preparation.detection.npixels'. "
            "Use 'preparation.detection.n_pixels' instead."
        )



def validate_fit_priors(
    config: dict[str, Any],
) -> None:
    """Validate that every sampled morphology parameter has a typed prior."""
    from baymorph.models import get_model
    from baymorph.pipeline.prior_config import (
        parse_prior_config,
        prior_config_for_parameter,
    )

    fit_config = config.get(
        "fit",
        {},
    )

    configured_priors = fit_config.get(
        "priors",
        {},
    )

    references = {
        "target_x": 0.0,
        "target_y": 0.0,
    }

    for model_name in fit_config.get(
        "models",
        [],
    ):
        if model_name not in configured_priors:
            raise ValueError(
                f"Missing prior configuration for model: {model_name}"
            )

        model_priors = configured_priors[
            model_name
        ]

        model_definition = get_model(
            model_name
        )

        missing = []

        for parameter_name in model_definition.parameter_names:
            if parameter_name == "sky":
                continue

            try:
                prior = prior_config_for_parameter(
                    parameter_name,
                    model_priors,
                )
            except ValueError:
                missing.append(
                    parameter_name
                )
                continue

            parse_prior_config(
                prior,
                references=references,
            )

        if missing:
            names = ", ".join(
                missing
            )

            raise ValueError(
                f"Missing prior configuration for model "
                f"{model_name!r}: {names}."
            )



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

    user_sampler = dict(
        config.get(
            "fit",
            {},
        ).get(
            "sampler",
            {},
        )
    )

    sampler_name = str(
        user_sampler.get(
            "name",
            DEFAULT_CONFIG["fit"]["sampler"]["name"],
        )
    ).strip().lower()

    if sampler_name not in SAMPLER_DEFAULTS:
        available = ", ".join(
            sorted(SAMPLER_DEFAULTS)
        )
        raise ValueError(
            f"Unknown sampler {sampler_name!r}. "
            f"Available samplers: {available}."
        )

    config_without_sampler = deepcopy(config)

    if "fit" in config_without_sampler:
        config_without_sampler["fit"] = dict(
            config_without_sampler["fit"]
        )
        config_without_sampler["fit"].pop(
            "sampler",
            None,
        )

    resolved = deep_update(
        deepcopy(DEFAULT_CONFIG),
        config_without_sampler,
    )

    resolved["fit"]["sampler"] = deep_update(
        deepcopy(
            SAMPLER_DEFAULTS[
                sampler_name
            ]
        ),
        user_sampler,
    )

    resolved["fit"]["sampler"]["name"] = sampler_name

    validate_sampler_config(
        resolved["fit"]["sampler"]
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


def _estimate_psf_half_size_limits(
    *,
    seeing_fwhm_arcsec: float,
    pixel_scale_arcsec: float,
) -> tuple[int, int]:
    """Estimate PSF candidate cutout limits from observational sampling.

    Parameters
    ----------
    seeing_fwhm_arcsec : float
        Seeing full width at half maximum in arcseconds.
    pixel_scale_arcsec : float
        Image pixel scale in arcseconds per pixel.

    Returns
    -------
    tuple of int
        Minimum and maximum PSF candidate cutout half-sizes, in pixels.

    Raises
    ------
    ValueError
        If either input is non-positive.
    """
    if seeing_fwhm_arcsec <= 0:
        raise ValueError("seeing_fwhm_arcsec must be positive.")

    if pixel_scale_arcsec <= 0:
        raise ValueError("pixel_scale_arcsec must be positive.")

    seeing_fwhm_pixels = (
        seeing_fwhm_arcsec
        / pixel_scale_arcsec
    )

    min_half_size = int(
        math.ceil(4.0 * seeing_fwhm_pixels)
    )
    max_half_size = int(
        math.ceil(10.0 * seeing_fwhm_pixels)
    )

    return min_half_size, max_half_size


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
    """
    resolved = deepcopy(config)

    # ------------------------------------------------------------------
    # Target geometry
    # ------------------------------------------------------------------

    data_cfg = resolved.get("data")

    if data_cfg is not None:
        if data_cfg.get("target_position") == "center":
            data_cfg["target_position"] = {
                "x": observation.metadata.center_x,
                "y": observation.metadata.center_y,
            }

    # ------------------------------------------------------------------
    # Image calibration
    # ------------------------------------------------------------------

    if data_cfg is not None:
        pixel_scale = data_cfg.get("pixel_scale")

        if pixel_scale in (None, "auto"):
            metadata_pixel_scale = observation.metadata.pixel_scale

            if metadata_pixel_scale is not None:
                data_cfg["pixel_scale"] = metadata_pixel_scale

    # ------------------------------------------------------------------
    # PSF candidate selection
    # ------------------------------------------------------------------

    candidate_cfg = (
        resolved
        .get("psf", {})
        .get("empirical", {})
        .get("candidate_selection")
    )

    if candidate_cfg is not None:
        #
        # Automatic half-size limits from seeing
        #
        min_half_size = candidate_cfg.get("min_half_size")
        max_half_size = candidate_cfg.get("max_half_size")

        needs_min_half_size = min_half_size == "auto"
        needs_max_half_size = max_half_size == "auto"

        if needs_min_half_size or needs_max_half_size:
            seeing_fwhm = observation.metadata.seeing_fwhm

            pixel_scale = None
            if data_cfg is not None:
                pixel_scale = data_cfg.get("pixel_scale")

            if (
                seeing_fwhm is not None
                and isinstance(pixel_scale, (int, float))
            ):
                (
                    estimated_min_half_size,
                    estimated_max_half_size,
                ) = _estimate_psf_half_size_limits(
                    seeing_fwhm_arcsec=seeing_fwhm,
                    pixel_scale_arcsec=float(pixel_scale),
                )

                if needs_min_half_size:
                    candidate_cfg["min_half_size"] = (
                        estimated_min_half_size
                    )

                if needs_max_half_size:
                    candidate_cfg["max_half_size"] = (
                        estimated_max_half_size
                    )

        #
        # Automatic saturation threshold
        #
        if candidate_cfg.get("max_peak") == "auto":
            saturation = observation.metadata.saturation

            if saturation is not None:
                candidate_cfg["max_peak"] = saturation
            else:
                candidate_cfg["max_peak"] = (
                    DEFAULT_CONFIG["psf"]["empirical"]
                    ["candidate_selection"]["max_peak"]
                )

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

def validate_sampler_config(
    sampler_config: dict[str, Any],
) -> None:
    """Validate the BayMorph sampler-configuration envelope.

    Native sampler keyword arguments are intentionally not interpreted here.
    They are forwarded by the selected backend to the corresponding upstream
    constructor or run method.
    """
    name = str(
        sampler_config.get(
            "name",
            "",
        )
    ).strip().lower()

    if name not in SAMPLER_DEFAULTS:
        available = ", ".join(
            sorted(SAMPLER_DEFAULTS)
        )
        raise ValueError(
            f"Unknown sampler {name!r}. "
            f"Available samplers: {available}."
        )

    init_config = sampler_config.get(
        "init",
        {},
    )
    run_config = sampler_config.get(
        "run",
        {},
    )

    if not isinstance(
        init_config,
        dict,
    ):
        raise TypeError(
            "fit.sampler.init must be a mapping."
        )

    if not isinstance(
        run_config,
        dict,
    ):
        raise TypeError(
            "fit.sampler.run must be a mapping."
        )

    execution_config = sampler_config.get(
        "execution",
        {},
    )

    if not isinstance(
        execution_config,
        dict,
    ):
        raise TypeError(
            "fit.sampler.execution must be a mapping."
        )

    resolve_sampler_execution(
        execution_config
    )

    if name == "dynesty":
        mode = str(
            sampler_config.get(
                "mode",
                "static",
            )
        ).strip().lower()

        if mode not in {
            "static",
            "dynamic",
        }:
            raise ValueError(
                "Dynesty sampler mode must be "
                "'static' or 'dynamic'."
            )

    elif "mode" in sampler_config:
        raise ValueError(
            f"Sampler {name!r} does not define a BayMorph mode."
        )
