"""Configuration loading, validation, and resolution for BayMorph pipelines."""

from __future__ import annotations

import math
from copy import deepcopy
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import yaml

from baymorph.core.correlated_noise import (
    FIELD_SEPARATED_GROWTH_PIXELS,
    SKY_CORRELATION_ESTIMATORS,
)
from baymorph.core.observation import Observation, resolve_seeing_fwhm_pixels
from baymorph.fitting.execution import resolve_sampler_execution
from baymorph.likelihood.factory import normalize_likelihood_type

REQUIRED_TOP_LEVEL_KEYS = ("object", "data", "outputs")

LIKELIHOOD_DEFAULTS = {
    # AD-047: the noise model of every fit run from a configuration. Its
    # factorization is fixed; the effective gain has no default and every
    # configuration declares it.
    "correlated_gaussian": {
        "type": "correlated_gaussian",
        "include_normalization": True,
        "factorization": {"type": "nearest_neighbor", "max_neighbors": 24},
    }
}
"""Defaults of each likelihood type, applied only to a block of that type."""

DEFAULT_CONTAMINANT_GROWTH_PIXELS = FIELD_SEPARATED_GROWTH_PIXELS
"""Default growth of the contaminant masks, in pixels, when the preparation
masks its detected segments (``data.mask_mode: union``): the growth with
which the likelihood support and the field-separated sky correlation were
validated (AD-051). A curated mask used as it is, with ``exact`` or
``replace``, is not grown."""


def default_contaminant_growth_pixels(mask_mode: str) -> int:
    """Return the default growth of the contaminant masks for a mask mode."""
    if str(mask_mode).strip().lower() == "union":
        return DEFAULT_CONTAMINANT_GROWTH_PIXELS

    return 0


SAMPLER_DEFAULTS = {
    "pocomc": {
        "name": "pocomc",
        "execution": {"processes": "auto", "progress": True},
        "init": {},
        "run": {},
    },
    "dynesty": {
        "name": "dynesty",
        "mode": "static",
        "execution": {"processes": "auto", "progress": True},
        "init": {},
        "run": {},
    },
    "nautilus": {
        "name": "nautilus",
        "execution": {"processes": "auto", "progress": True},
        "init": {},
        "run": {},
    },
}


DEFAULT_CONFIG = {
    "data": {
        "hdu": None,
        "mask_path": None,
        "mask_mode": "union",
        "pixel_scale": None,
        "target_position": None,
        "cutout": {"mode": "fixed", "size": None},
    },
    "outputs": {"overwrite": False},
    "preparation": {
        # Mesh-background settings that no preparation step reads any more.
        # They stay because the preparation section is part of the identity
        # of persisted preparation products.
        "background": {"box_size": 30, "filter_size": 3},
        "noise": {
            # The estimator of the sky correlation of AD-047;
            # "field_separated" (AD-051) is declared for a sky that also
            # holds a large-scale field.
            "sky_correlation": "autocovariance"
        },
        "masking": {
            "negative_threshold_sigma": 5.0,
            # Only the segments that still hold a source above the smooth
            # light of the target are masked; "all_segments" masks every
            # detected segment, as before this rule.
            "contaminant_rule": "local_excess",
            # Pixels within this distance of a masked segment are excluded
            # from the sky region and the likelihood support (AD-051); 0
            # when data.mask_mode is "exact" or "replace".
            "contaminant_growth_pixels": DEFAULT_CONTAMINANT_GROWTH_PIXELS,
        },
        "detection": {"n_sigma": 3.0, "n_levels": 32, "n_pixels": 20},
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
                "min_peak_snr": 30.0,
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
                "centroid_tolerance_pixels": 0.05,
                # Radii of the builder of AD-046, validated on the test
                # bench of the PSF workstream; they enter the PSF identity.
                "sky_inner_radius": 80.0,
                "sky_outer_radius": 160.0,
                "mask_growth_pixels": 5,
                "support_half_size": 64,
            },
        },
    },
    "fit": {
        "models": ["single_sersic"],
        "likelihood": deepcopy(LIKELIHOOD_DEFAULTS["correlated_gaussian"]),
        "sampler": {"name": "nautilus"},
        "comparison": {"enabled": False, "metric": "bayes_factor"},
        "background": {
            "model": "constant",
            "priors": {
                "background_level": {"type": "uniform", "lower": -50.0, "upper": 50.0},
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
                "re": {"type": "uniform", "lower": 1.0, "upper": 120.0},
                "n": {"type": "uniform", "lower": 0.5, "upper": 6.0},
                "q": {"type": "uniform", "lower": 0.4, "upper": 1.0},
                "theta": {"type": "uniform", "lower": "-pi/2", "upper": "pi/2"},
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
            },
            "double_sersic": {
                "re_bulge": {"type": "uniform", "lower": 1.0, "upper": 30.0},
                "n_bulge": {"type": "uniform", "lower": 1.5, "upper": 6.0},
                "re_disk": {"type": "uniform", "lower": 10.0, "upper": 120.0},
                "n_disk": {"type": "uniform", "lower": 0.5, "upper": 3.0},
                "q": {"type": "uniform", "lower": 0.4, "upper": 1.0},
                "theta": {"type": "uniform", "lower": "-pi/2", "upper": "pi/2"},
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
            },
            "sersic_point_source": {
                "re": {"type": "uniform", "lower": 1.0, "upper": 120.0},
                "n": {"type": "uniform", "lower": 0.5, "upper": 6.0},
                "q": {"type": "uniform", "lower": 0.4, "upper": 1.0},
                "theta": {"type": "uniform", "lower": "-pi/2", "upper": "pi/2"},
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
            },
            "bulge_disk": {
                "re_bulge": {"type": "uniform", "lower": 1.0, "upper": 30.0},
                "n_bulge": {"type": "uniform", "lower": 1.5, "upper": 6.0},
                "q_bulge": {"type": "uniform", "lower": 0.4, "upper": 1.0},
                "theta_bulge": {"type": "uniform", "lower": "-pi/2", "upper": "pi/2"},
                "re_disk": {"type": "uniform", "lower": 10.0, "upper": 120.0},
                "q_disk": {"type": "uniform", "lower": 0.4, "upper": 1.0},
                "theta_disk": {"type": "uniform", "lower": "-pi/2", "upper": "pi/2"},
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
            },
            "boxy_sersic": {
                "re": {"type": "uniform", "lower": 1.0, "upper": 120.0},
                "n": {"type": "uniform", "lower": 0.5, "upper": 6.0},
                "q": {"type": "uniform", "lower": 0.4, "upper": 1.0},
                "c0": {"type": "uniform", "lower": -1.0, "upper": 1.0},
                "theta": {"type": "uniform", "lower": "-pi/2", "upper": "pi/2"},
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
            },
            "core_sersic": {
                "re": {"type": "uniform", "lower": 1.0, "upper": 120.0},
                "n": {"type": "uniform", "lower": 0.5, "upper": 6.0},
                "rb": {"type": "uniform", "lower": 0.5, "upper": 20.0},
                "gamma": {"type": "uniform", "lower": 0.0, "upper": 0.9},
                "alpha": {"type": "uniform", "lower": 0.5, "upper": 2.0},
                "q": {"type": "uniform", "lower": 0.4, "upper": 1.0},
                "theta": {"type": "uniform", "lower": "-pi/2", "upper": "pi/2"},
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
            },
            "bulge_broken_disk": {
                "re_bulge": {"type": "uniform", "lower": 1.0, "upper": 30.0},
                "n_bulge": {"type": "uniform", "lower": 1.5, "upper": 6.0},
                "q_bulge": {"type": "uniform", "lower": 0.4, "upper": 1.0},
                "theta_bulge": {"type": "uniform", "lower": "-pi/2", "upper": "pi/2"},
                "h_in_disk": {"type": "uniform", "lower": 2.0, "upper": 80.0},
                "h_out_disk": {"type": "uniform", "lower": 2.0, "upper": 80.0},
                "rb_disk": {"type": "uniform", "lower": 5.0, "upper": 300.0},
                "wb_disk": {"type": "uniform", "lower": 0.5, "upper": 20.0},
                "q_disk": {"type": "uniform", "lower": 0.4, "upper": 1.0},
                "theta_disk": {"type": "uniform", "lower": "-pi/2", "upper": "pi/2"},
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
            },
            "bulge_disk_bar": {
                "re_bulge": {"type": "uniform", "lower": 1.0, "upper": 30.0},
                "n_bulge": {"type": "uniform", "lower": 1.5, "upper": 6.0},
                "q_bulge": {"type": "uniform", "lower": 0.4, "upper": 1.0},
                "theta_bulge": {"type": "uniform", "lower": "-pi/2", "upper": "pi/2"},
                "re_disk": {"type": "uniform", "lower": 10.0, "upper": 120.0},
                "q_disk": {"type": "uniform", "lower": 0.4, "upper": 1.0},
                "theta_disk": {"type": "uniform", "lower": "-pi/2", "upper": "pi/2"},
                "a_bar": {"type": "uniform", "lower": 5.0, "upper": 100.0},
                "alpha_bar": {"type": "uniform", "lower": 2.0, "upper": 5.0},
                "c0_bar": {"type": "uniform", "lower": 0.0, "upper": 3.5},
                "q_bar": {"type": "uniform", "lower": 0.2, "upper": 0.7},
                "theta_bar": {"type": "uniform", "lower": "-pi/2", "upper": "pi/2"},
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
            },
            "bulge_disk_ring": {
                "re_bulge": {"type": "uniform", "lower": 1.0, "upper": 30.0},
                "n_bulge": {"type": "uniform", "lower": 1.5, "upper": 6.0},
                "q_bulge": {"type": "uniform", "lower": 0.4, "upper": 1.0},
                "theta_bulge": {"type": "uniform", "lower": "-pi/2", "upper": "pi/2"},
                "re_disk": {"type": "uniform", "lower": 10.0, "upper": 120.0},
                "q_disk": {"type": "uniform", "lower": 0.4, "upper": 1.0},
                "theta_disk": {"type": "uniform", "lower": "-pi/2", "upper": "pi/2"},
                "r_ring": {"type": "uniform", "lower": 5.0, "upper": 200.0},
                "sigma_ring": {"type": "uniform", "lower": 2.0, "upper": 20.0},
                "q_ring": {"type": "uniform", "lower": 0.5, "upper": 1.0},
                "theta_ring": {"type": "uniform", "lower": "-pi/2", "upper": "pi/2"},
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
            },
            "bulge_disk_spiral": {
                "re_bulge": {"type": "uniform", "lower": 1.0, "upper": 30.0},
                "n_bulge": {"type": "uniform", "lower": 1.5, "upper": 6.0},
                "q_bulge": {"type": "uniform", "lower": 0.4, "upper": 1.0},
                "theta_bulge": {"type": "uniform", "lower": "-pi/2", "upper": "pi/2"},
                "re_disk": {"type": "uniform", "lower": 10.0, "upper": 120.0},
                "q_disk": {"type": "uniform", "lower": 0.4, "upper": 1.0},
                "theta_disk": {"type": "uniform", "lower": "-pi/2", "upper": "pi/2"},
                "amplitude_spiral": {"type": "uniform", "lower": 0.0, "upper": 1.0},
                "winding_spiral": {"type": "uniform", "lower": -6.0, "upper": 6.0},
                "phase_spiral": {"type": "uniform", "lower": 0.0, "upper": "pi"},
                "r_in_spiral": {"type": "uniform", "lower": 1.0, "upper": 60.0},
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
        Path to a YAML configuration file.

    Returns
    -------
    dict
        Raw configuration dictionary loaded from disk.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
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


def _validate_psf_builder_radii(construction: dict[str, Any]) -> None:
    """Validate the radii of the empirical-PSF builder of AD-046."""
    for key in ("sky_inner_radius", "sky_outer_radius"):
        if key not in construction:
            continue

        value = construction[key]

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(
                f"psf.empirical.construction.{key} must be a finite " "positive number."
            )

    for key, minimum in (("mask_growth_pixels", 0), ("support_half_size", 1)):
        if key not in construction:
            continue

        value = construction[key]

        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < minimum
        ):
            raise ValueError(
                f"psf.empirical.construction.{key} must be an integer of at "
                f"least {minimum}."
            )

    support = construction.get("support_half_size", 64)
    inner = construction.get("sky_inner_radius", 80.0)
    outer = construction.get("sky_outer_radius", 160.0)

    if not support < inner < outer:
        raise ValueError(
            "psf.empirical.construction needs support_half_size < "
            "sky_inner_radius < sky_outer_radius, so that the sky of each "
            "star is measured beyond its light."
        )


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
        {"noise_map_path", "psf_path"} & config["data"].keys()
    )

    if unsupported_data_keys:
        raise ValueError(
            "Unsupported configuration entries in section 'data': "
            f"{unsupported_data_keys}. "
            "External noise maps and direct PSF paths are not yet "
            "supported by the single-band production workflow."
        )

    if "external_mask_path" in config["data"]:
        raise ValueError(
            "Config data.external_mask_path is no longer supported. "
            "Declare the curated mask under data.mask_path."
        )

    hdu = config["data"].get("hdu")

    if hdu is not None and (
        isinstance(hdu, bool)
        or not (
            (isinstance(hdu, int) and hdu >= 0)
            or (isinstance(hdu, str) and hdu.strip())
        )
    ):
        raise ValueError(
            "Config data.hdu must be a non-negative HDU index or a non-empty "
            f"extension name; got {hdu!r}."
        )

    mask_mode = str(config["data"].get("mask_mode", "union")).strip().lower()

    if mask_mode not in {"union", "exact", "replace"}:
        raise ValueError(
            "Config data.mask_mode must be one of 'union', 'exact', or " "'replace'."
        )

    if mask_mode in {"exact", "replace"} and not config["data"].get("mask_path"):
        raise ValueError(
            "Config data.mask_path is required when data.mask_mode is "
            f"{mask_mode!r}."
        )

    if "root" not in config["outputs"]:
        raise ValueError("Config section 'outputs' must contain 'root'.")

    if "masking" in config:
        raise ValueError(
            "Unsupported top-level configuration section 'masking'. "
            "The current single-band workflow constructs its automatic "
            "fitting mask during image preparation. The only masking control "
            "it exposes is 'preparation.masking'."
        )

    psf_construction_config = (
        config.get("psf", {}).get("empirical", {}).get("construction", {})
    )
    psf_candidate_config = (
        config.get("psf", {}).get("empirical", {}).get("candidate_selection", {})
    )

    if "max_peak" in psf_candidate_config:
        max_peak = psf_candidate_config["max_peak"]

        if max_peak == "auto":
            raise ValueError(
                "psf.empirical.candidate_selection.max_peak='auto' is "
                "unsafe because FITS saturation-keyword units are not "
                "guaranteed to match the science-image units. Supply an "
                "explicit finite positive numerical max_peak."
            )

        if (
            isinstance(max_peak, bool)
            or not isinstance(max_peak, Real)
            or not math.isfinite(float(max_peak))
            or float(max_peak) <= 0.0
        ):
            raise ValueError(
                "psf.empirical.candidate_selection.max_peak must be a "
                "finite positive number."
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

    if "centroid_tolerance_pixels" in psf_construction_config:
        tolerance = psf_construction_config["centroid_tolerance_pixels"]

        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, Real)
            or not math.isfinite(float(tolerance))
            or float(tolerance) <= 0.0
        ):
            raise ValueError(
                "psf.empirical.construction.centroid_tolerance_pixels "
                "must be a finite positive number."
            )

    _validate_psf_builder_radii(psf_construction_config)

    fit_config = config.get("fit", {})

    allowed_fit_keys = {
        "models",
        "likelihood",
        "sampler",
        "comparison",
        "background",
        "priors",
    }

    unknown_fit_keys = sorted(set(fit_config) - allowed_fit_keys)

    if unknown_fit_keys:
        raise ValueError(
            "Unsupported configuration entries in section 'fit': "
            f"{unknown_fit_keys}. Available entries: "
            f"{sorted(allowed_fit_keys)}."
        )

    background_config = fit_config.get("background", {})

    if "model" in background_config:
        background_model = str(background_config["model"]).strip().lower()

        if background_model not in {"constant", "plane"}:
            raise ValueError(
                "Unsupported fit.background.model "
                f"{background_model!r}. "
                "Available models: 'constant', 'plane'."
            )

    preparation_config = config.get("preparation", {})

    unsupported_preparation_entries = []

    if "method" in preparation_config.get("background", {}):
        unsupported_preparation_entries.append("preparation.background.method")

    if "method" in preparation_config.get("noise", {}):
        unsupported_preparation_entries.append("preparation.noise.method")

    if unsupported_preparation_entries:
        raise ValueError(
            "Unsupported fixed configuration entries: "
            f"{unsupported_preparation_entries}. "
            "The single-band production workflow currently uses empirical "
            "background estimation, and measures the noise scale from "
            "residuals about a plane fitted to the sky region (AD-041)."
        )

    detection_config = preparation_config.get("detection", {})

    if "npixels" in detection_config:
        raise ValueError(
            "Unsupported configuration key "
            "'preparation.detection.npixels'. "
            "Use 'preparation.detection.n_pixels' instead."
        )

    masking_config = preparation_config.get("masking", {})

    unsupported_masking_entries = sorted(
        set(masking_config)
        - {"negative_threshold_sigma", "contaminant_rule", "contaminant_growth_pixels"}
    )

    if unsupported_masking_entries:
        raise ValueError(
            "Unsupported configuration entries in preparation.masking: "
            f"{unsupported_masking_entries}. "
            "Available entries: ['contaminant_growth_pixels', "
            "'contaminant_rule', 'negative_threshold_sigma']."
        )

    contaminant_rule = masking_config.get("contaminant_rule", "local_excess")

    if contaminant_rule not in ("local_excess", "all_segments"):
        raise ValueError(
            "Config preparation.masking.contaminant_rule must be "
            "'local_excess' or 'all_segments', not "
            f"{contaminant_rule!r}."
        )

    mask_mode = str(config.get("data", {}).get("mask_mode", "union")).strip().lower()
    contaminant_growth = masking_config.get(
        "contaminant_growth_pixels", default_contaminant_growth_pixels(mask_mode)
    )

    if (
        isinstance(contaminant_growth, bool)
        or not isinstance(contaminant_growth, Integral)
        or contaminant_growth < 0
    ):
        raise ValueError(
            "Config preparation.masking.contaminant_growth_pixels must be a "
            "non-negative integer number of pixels."
        )

    if contaminant_growth > 0 and mask_mode in {"exact", "replace"}:
        raise ValueError(
            "Config preparation.masking.contaminant_growth_pixels grows the "
            "detected contaminant segments, which data.mask_mode "
            f"{mask_mode!r} does not use: set it to 0, or use mask_mode "
            "'union'."
        )

    noise_config = preparation_config.get("noise", {})
    unsupported_noise_entries = sorted(
        set(noise_config) - {"sky_correlation", "method"}
    )

    if unsupported_noise_entries:
        raise ValueError(
            "Unsupported configuration entries in preparation.noise: "
            f"{unsupported_noise_entries}. "
            "Available entries: ['sky_correlation']."
        )

    sky_correlation = noise_config.get("sky_correlation", "autocovariance")

    if (
        not isinstance(sky_correlation, str)
        or sky_correlation not in SKY_CORRELATION_ESTIMATORS
    ):
        raise ValueError(
            "Config preparation.noise.sky_correlation must be one of "
            f"{list(SKY_CORRELATION_ESTIMATORS)}, not {sky_correlation!r}."
        )

    if (
        sky_correlation == "field_separated"
        and contaminant_growth != FIELD_SEPARATED_GROWTH_PIXELS
    ):
        requirement = (
            f"{FIELD_SEPARATED_GROWTH_PIXELS} px, which data.mask_mode "
            f"{mask_mode!r} does not apply"
            if mask_mode in {"exact", "replace"}
            else f"{FIELD_SEPARATED_GROWTH_PIXELS} px"
        )
        raise ValueError(
            "Config preparation.noise.sky_correlation 'field_separated' was "
            "validated on sky regions whose detected contaminant masks are "
            "grown by preparation.masking.contaminant_growth_pixels: "
            f"{requirement}."
        )

    negative_threshold = masking_config.get("negative_threshold_sigma")

    if negative_threshold is not None:
        if isinstance(negative_threshold, bool) or not isinstance(
            negative_threshold, Real
        ):
            raise ValueError(
                "Config preparation.masking.negative_threshold_sigma must be "
                "a number of noise scales, or null to disable the criterion."
            )

        if (
            not math.isfinite(float(negative_threshold))
            or float(negative_threshold) <= 0.0
        ):
            raise ValueError(
                "Config preparation.masking.negative_threshold_sigma must be "
                "finite and positive."
            )


def _prior_support(prior) -> tuple[float, float]:
    """Return the smallest and largest value a parsed prior can take."""
    from baymorph.fitting.priors import FixedPrior, NormalPrior

    if isinstance(prior, FixedPrior):
        return prior.value, prior.value

    if isinstance(prior, NormalPrior):
        return -math.inf, math.inf

    return float(prior.lower), float(prior.upper)


def _require_validated_support(
    prior,
    bounds: tuple[float | None, float | None],
    *,
    model_name: str,
    parameter_name: str,
) -> None:
    """Reject a prior whose support leaves the validated rendering domain."""
    low, high = _prior_support(prior)
    lower, upper = bounds

    if (lower is not None and low < lower) or (upper is not None and high > upper):
        domain = (
            f"[{'-inf' if lower is None else f'{lower:g}'}, "
            f"{'inf' if upper is None else f'{upper:g}'}]"
        )

        raise ValueError(
            f"The prior of fit.priors.{model_name}.{parameter_name} spans "
            f"[{low:g}, {high:g}], beyond {domain}, the domain in which the "
            f"rendering of {model_name!r} is validated (AD-048)."
        )


def validate_fit_priors(config: dict[str, Any]) -> None:
    """Validate that every sampled morphology parameter has a typed prior.

    For the families of AD-048 the support of every prior must also lie in
    the domain in which the rendering of the model is validated.
    """
    from baymorph.models import get_model
    from baymorph.models.validated_domains import validated_domain
    from baymorph.pipeline.prior_config import (
        parse_prior_config,
        prior_config_for_parameter,
    )

    fit_config = config.get("fit", {})

    configured_priors = fit_config.get("priors", {})

    references = {"target_x": 0.0, "target_y": 0.0}

    for model_name in fit_config.get("models", []):
        if model_name not in configured_priors:
            raise ValueError(f"Missing prior configuration for model: {model_name}")

        model_priors = configured_priors[model_name]

        if "sky" in model_priors:
            raise ValueError(
                f"fit.priors.{model_name}.sky is not a supported prior "
                "location: the background level is owned by "
                "fit.background.priors.background_level, not by a "
                "per-model morphology prior. Move this prior there."
            )

        model_definition = get_model(model_name)
        domain = validated_domain(model_name)

        missing = []

        for parameter_name in model_definition.parameter_names:
            if parameter_name == "sky":
                continue

            try:
                prior = prior_config_for_parameter(parameter_name, model_priors)
            except ValueError:
                missing.append(parameter_name)
                continue

            parsed = parse_prior_config(prior, references=references)

            if parameter_name in domain:
                _require_validated_support(
                    parsed,
                    domain[parameter_name],
                    model_name=model_name,
                    parameter_name=parameter_name,
                )

        if missing:
            names = ", ".join(missing)

            raise ValueError(
                f"Missing prior configuration for model " f"{model_name!r}: {names}."
            )


def resolve_likelihood_config(likelihood: Any) -> Any:
    """Return a likelihood block completed with the defaults of its type.

    The defaults of a type apply only to a block of that type, so a block of
    another type keeps exactly its own options. A missing block takes the
    defaults of the correlated Gaussian likelihood of AD-047.
    """
    if likelihood is None:
        return deepcopy(LIKELIHOOD_DEFAULTS["correlated_gaussian"])

    if isinstance(likelihood, str):
        likelihood = {"type": likelihood}

    if not isinstance(likelihood, dict):
        return deepcopy(likelihood)

    likelihood_type = normalize_likelihood_type(
        str(likelihood.get("type", "correlated_gaussian"))
    )
    defaults = LIKELIHOOD_DEFAULTS.get(likelihood_type)

    if defaults is None:
        return deepcopy(likelihood)

    return deep_update(deepcopy(defaults), deepcopy(likelihood))


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

    user_sampler = dict(config.get("fit", {}).get("sampler", {}))

    sampler_name = (
        str(user_sampler.get("name", DEFAULT_CONFIG["fit"]["sampler"]["name"]))
        .strip()
        .lower()
    )

    if sampler_name not in SAMPLER_DEFAULTS:
        available = ", ".join(sorted(SAMPLER_DEFAULTS))
        raise ValueError(
            f"Unknown sampler {sampler_name!r}. " f"Available samplers: {available}."
        )

    config_without_sampler = deepcopy(config)
    user_likelihood = None

    if "fit" in config_without_sampler:
        config_without_sampler["fit"] = dict(config_without_sampler["fit"])
        config_without_sampler["fit"].pop("sampler", None)
        user_likelihood = config_without_sampler["fit"].pop("likelihood", None)

    resolved = deep_update(deepcopy(DEFAULT_CONFIG), config_without_sampler)
    resolved["fit"]["likelihood"] = resolve_likelihood_config(user_likelihood)

    # The default growth of the contaminant masks depends on the mask mode:
    # a curated mask used as it is is not grown.
    user_masking = (config.get("preparation") or {}).get("masking") or {}
    if "contaminant_growth_pixels" not in user_masking:
        resolved["preparation"]["masking"]["contaminant_growth_pixels"] = (
            default_contaminant_growth_pixels(
                resolved["data"].get("mask_mode", "union")
            )
        )

    resolved["fit"]["sampler"] = deep_update(
        deepcopy(SAMPLER_DEFAULTS[sampler_name]), user_sampler
    )

    resolved["fit"]["sampler"]["name"] = sampler_name

    validate_sampler_config(resolved["fit"]["sampler"])

    if resolved["fit"]["comparison"]["enabled"] is True:
        comparison_models = resolved["fit"]["models"]
        expected_models = {"single_sersic", "double_sersic"}

        if len(comparison_models) != 2 or set(comparison_models) != expected_models:
            raise ValueError(
                "Enabled single-band model comparison requires exactly "
                "fit.models=['single_sersic', 'double_sersic'] in either "
                "order."
            )

    resolved["object"] = dict(resolved["object"])
    resolved["data"] = dict(resolved["data"])
    resolved["outputs"] = dict(resolved["outputs"])

    resolved["object"]["name"] = str(resolved["object"]["name"])
    resolved["data"]["image_path"] = str(Path(resolved["data"]["image_path"]))
    resolved["outputs"]["root"] = str(Path(resolved["outputs"]["root"]))

    return resolved


def _estimate_psf_half_size_limits(*, seeing_fwhm_pixels: float) -> tuple[int, int]:
    """Estimate PSF candidate cutout limits from the seeing in pixels."""
    if seeing_fwhm_pixels <= 0:
        raise ValueError("seeing_fwhm_pixels must be positive.")

    return (
        int(math.ceil(4.0 * seeing_fwhm_pixels)),
        int(math.ceil(10.0 * seeing_fwhm_pixels)),
    )


def resolve_runtime_config(
    config: dict[str, Any], observation: Observation
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

    data_cfg = resolved.get("data")

    if data_cfg is not None:
        if data_cfg.get("target_position") == "center":
            data_cfg["target_position"] = {
                "x": observation.metadata.center_x,
                "y": observation.metadata.center_y,
            }

    if data_cfg is not None:
        pixel_scale = data_cfg.get("pixel_scale")

        if pixel_scale in (None, "auto"):
            metadata_pixel_scale = observation.metadata.pixel_scale

            if metadata_pixel_scale is not None:
                data_cfg["pixel_scale"] = metadata_pixel_scale

    candidate_cfg = (
        resolved.get("psf", {}).get("empirical", {}).get("candidate_selection")
    )

    if candidate_cfg is not None:
        min_half_size = candidate_cfg.get("min_half_size")
        max_half_size = candidate_cfg.get("max_half_size")

        needs_min_half_size = min_half_size == "auto"
        needs_max_half_size = max_half_size == "auto"

        if needs_min_half_size or needs_max_half_size:
            pixel_scale_arcsec = None

            if data_cfg is not None:
                pixel_scale = data_cfg.get("pixel_scale")

                if isinstance(pixel_scale, (int, float)):
                    pixel_scale_arcsec = float(pixel_scale)

            seeing_fwhm_pixels = resolve_seeing_fwhm_pixels(
                observation.metadata, pixel_scale_arcsec=pixel_scale_arcsec
            )

            candidate_defaults = DEFAULT_CONFIG["psf"]["empirical"][
                "candidate_selection"
            ]

            if seeing_fwhm_pixels is None:
                estimated_min_half_size = candidate_defaults["min_half_size"]
                estimated_max_half_size = candidate_defaults["max_half_size"]
            else:
                estimated_min_half_size, estimated_max_half_size = (
                    _estimate_psf_half_size_limits(
                        seeing_fwhm_pixels=seeing_fwhm_pixels
                    )
                )

            if needs_min_half_size:
                candidate_cfg["min_half_size"] = estimated_min_half_size

            if needs_max_half_size:
                candidate_cfg["max_half_size"] = estimated_max_half_size

        if candidate_cfg.get("max_peak") == "auto":
            raise ValueError(
                "psf.empirical.candidate_selection.max_peak='auto' is "
                "unsafe for observed data because FITS saturation-keyword "
                "units are not guaranteed to match the science-image "
                "units. Supply an explicit finite numerical max_peak."
            )

    return resolved


def load_resolved_config(path: str | Path) -> dict[str, Any]:
    """Load, validate, and resolve a BayMorph YAML configuration.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a YAML configuration file.

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
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value

    return result


def validate_sampler_config(sampler_config: dict[str, Any]) -> None:
    """Validate the BayMorph sampler-configuration envelope.

    Native sampler keyword arguments are intentionally not interpreted here.
    They are forwarded by the selected backend to the corresponding upstream
    constructor or run method.
    """
    name = str(sampler_config.get("name", "")).strip().lower()

    if name not in SAMPLER_DEFAULTS:
        available = ", ".join(sorted(SAMPLER_DEFAULTS))
        raise ValueError(
            f"Unknown sampler {name!r}. " f"Available samplers: {available}."
        )

    init_config = sampler_config.get("init", {})
    run_config = sampler_config.get("run", {})

    if not isinstance(init_config, dict):
        raise TypeError("fit.sampler.init must be a mapping.")

    if not isinstance(run_config, dict):
        raise TypeError("fit.sampler.run must be a mapping.")

    execution_config = sampler_config.get("execution", {})

    if not isinstance(execution_config, dict):
        raise TypeError("fit.sampler.execution must be a mapping.")

    resolve_sampler_execution(execution_config)

    if name == "dynesty":
        mode = str(sampler_config.get("mode", "static")).strip().lower()

        if mode not in {"static", "dynamic"}:
            raise ValueError("Dynesty sampler mode must be " "'static' or 'dynamic'.")

    elif "mode" in sampler_config:
        raise ValueError(f"Sampler {name!r} does not define a BayMorph mode.")
