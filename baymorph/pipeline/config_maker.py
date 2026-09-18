"""Configuration generation from astronomical FITS observations."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from baymorph.core.observation import (
    Observation,
    resolve_seeing_fwhm_pixels,
)
from baymorph.pipeline.prior_config import parse_prior_config
from baymorph.pipeline.config import (
    resolve_config,
    validate_fit_priors,
)
from baymorph.pipeline.multi_observation_config import (
    build_multi_observation_parameter_layout,
    resolve_multi_observation_observation_configs,
)


TargetPosition = str | Mapping[str, float] | Sequence[float]


_OBSERVATION_OPTION_KEYS = {
    "object_name",
    "overwrite",
    "cutout_mode",
    "fixed_size",
    "psf_type",
    "gaussian_fwhm_pixels",
    "likelihood",
}


def _finite_positive(
    value: Any,
) -> float | None:
    """Return a finite positive float or None."""
    if value is None:
        return None

    try:
        result = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if (
        not math.isfinite(
            result
        )
        or result <= 0.0
    ):
        return None

    return result


def _normalize_target_position(
    target_position: TargetPosition | None,
) -> str | dict[str, float]:
    """Normalize an explicit target-identification seed."""
    if target_position is None:
        raise ValueError(
            "target_position is required. Supply (x, y), a mapping with "
            "'x' and 'y', or the explicit value 'center'."
        )

    if isinstance(
        target_position,
        str,
    ):
        value = target_position.strip().lower()

        if value != "center":
            raise ValueError(
                "String target_position must be exactly 'center'."
            )

        return "center"

    if isinstance(
        target_position,
        Mapping,
    ):
        if set(
            target_position
        ) != {
            "x",
            "y",
        }:
            raise ValueError(
                "Target-position mappings must contain exactly 'x' and 'y'."
            )

        x = float(
            target_position[
                "x"
            ]
        )
        y = float(
            target_position[
                "y"
            ]
        )

    elif isinstance(
        target_position,
        Sequence,
    ):
        if len(
            target_position
        ) != 2:
            raise ValueError(
                "Target-position sequences must contain exactly two values."
            )

        x = float(
            target_position[
                0
            ]
        )
        y = float(
            target_position[
                1
            ]
        )

    else:
        raise TypeError(
            "target_position must be 'center', an x/y mapping, "
            "or a two-element sequence."
        )

    if not (
        math.isfinite(
            x
        )
        and math.isfinite(
            y
        )
    ):
        raise ValueError(
            "Target-position coordinates must be finite."
        )

    return {
        "x": x,
        "y": y,
    }


def _cutout_config(
    mode: str,
    *,
    fixed_size: int | None,
) -> dict[str, Any]:
    """Build a standard cutout configuration."""
    mode = str(
        mode
    ).strip().lower()

    if mode == "auto":
        if fixed_size is not None:
            raise ValueError(
                "fixed_size cannot be supplied when cutout_mode='auto'."
            )

        return {
            "mode": "auto",
            "initial_factor": 4.0,
            "sky_margin": 20.0,
            "min_size": 100,
            "max_size": None,
        }

    if mode == "fixed":
        if fixed_size is None:
            raise ValueError(
                "fixed_size is required when cutout_mode='fixed'."
            )

        fixed_size = int(
            fixed_size
        )

        if fixed_size <= 0:
            raise ValueError(
                "fixed_size must be positive."
            )

        return {
            "mode": "fixed",
            "size": fixed_size,
        }

    raise ValueError(
        "cutout_mode must be 'auto' or 'fixed'."
    )



def _empirical_psf_config() -> dict[str, Any]:
    """Build empirical-PSF configuration without unit assumptions.

    FITS saturation keywords are intentionally not converted into a
    maximum-peak threshold. Production configurations must explicitly set
    that threshold after resolving its units against the science image.
    """
    return {
        "type": "empirical",
        "empirical": {
            "candidate_selection": {
                "auto": True,
            },
        },
    }


def _gaussian_psf_config(
    observation: Observation,
    *,
    pixel_scale: float | None,
    gaussian_fwhm_pixels: float | None,
) -> dict[str, Any]:
    """Build an explicitly requested Gaussian-PSF configuration."""
    if gaussian_fwhm_pixels is not None:
        fwhm_pixels = _finite_positive(
            gaussian_fwhm_pixels
        )

        if fwhm_pixels is None:
            raise ValueError(
                "gaussian_fwhm_pixels must be finite and positive."
            )
    else:
        fwhm_pixels = resolve_seeing_fwhm_pixels(
            observation.metadata,
            pixel_scale_arcsec=pixel_scale,
        )

        if fwhm_pixels is None:
            if (
                observation.metadata.seeing_fwhm is not None
                and observation.metadata.seeing_fwhm_unit is None
            ):
                reason = "the FITS FWHM unit is unresolved"
            else:
                reason = "no explicitly convertible FITS FWHM is available"

            raise ValueError(
                "Gaussian PSF generation requires gaussian_fwhm_pixels "
                f"because {reason}."
            )

    return {
        "type": "gaussian",
        "gaussian": {
            "fwhm": float(
                fwhm_pixels
            ),
        },
    }


def _write_yaml(
    path: Path,
    config: Mapping[str, Any],
) -> None:
    """Write one YAML configuration."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        yaml.safe_dump(
            dict(
                config
            ),
            stream,
            sort_keys=False,
        )


def _normalize_sampler_config(
    sampler: str | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize a public sampler specification."""
    if sampler is None:
        return None

    if isinstance(
        sampler,
        str,
    ):
        name = sampler.strip().lower()

        if not name:
            raise ValueError(
                "sampler must be a non-empty string."
            )

        return {
            "name": name,
        }

    if not isinstance(
        sampler,
        Mapping,
    ):
        raise TypeError(
            "sampler must be a string, mapping, or None."
        )

    config = deepcopy(
        dict(
            sampler
        )
    )

    if "name" in config:
        name = str(
            config["name"]
        ).strip().lower()

        if not name:
            raise ValueError(
                "sampler.name must be non-empty."
            )

        config["name"] = name

    return config


def _normalize_model_priors(
    priors: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Normalize explicit scientific model-prior overrides."""
    if priors is None:
        return {}

    if not isinstance(priors, Mapping):
        raise TypeError(
            "model_priors must be a mapping."
        )

    normalized: dict[str, dict[str, Any]] = {}

    for parameter_name, prior_config in priors.items():
        name = str(parameter_name).strip()

        if not name:
            raise ValueError(
                "Model-prior parameter names must be non-empty."
            )

        if not isinstance(prior_config, Mapping):
            raise TypeError(
                f"Prior configuration for {name!r} must be a mapping."
            )

        prior_mapping = dict(prior_config)
        relative_to = prior_mapping.get("relative_to")
        references = None

        if relative_to == "target_x":
            references = {"target_x": 0.0}
        elif relative_to == "target_y":
            references = {"target_y": 0.0}

        parse_prior_config(
            prior_mapping,
            references=references,
        )

        normalized[name] = prior_mapping

    return normalized


def make_config_from_fits(
    fits_path: str | Path,
    *,
    target_position: TargetPosition | None = None,
    output_path: str | Path | None = None,
    object_name: str | None = None,
    model_name: str = "single_sersic",
    model_priors: Mapping[str, Any] | None = None,
    likelihood: Mapping[str, Any] | None = None,
    sampler: str | Mapping[str, Any] | None = "nautilus",
    outputs_root: str | Path = "runs",
    overwrite: bool = False,
    cutout_mode: str = "auto",
    fixed_size: int | None = None,
    psf_type: str = "empirical",
    gaussian_fwhm_pixels: float | None = None,
    _require_complete_priors: bool = True,
) -> dict[str, Any]:
    """Create a runnable BayMorph configuration from one FITS image.

    Parameters
    ----------
    fits_path : str or pathlib.Path
        Input two-dimensional astronomical FITS image.
    target_position : mapping, sequence, or {"center"}
        Explicit target-identification seed. The value ``"center"`` requests
        the detector centre and does not imply automatic target discovery.
    output_path : str or pathlib.Path, optional
        YAML path to write. If omitted, no file is written.
    object_name : str, optional
        Analysis object name. The FITS filename stem is used by default.
    likelihood : mapping, optional
        Explicit likelihood configuration. If omitted, BayMorph uses the
        standard single-observation likelihood configuration.
        Explicit inference-region configuration. If omitted, BayMorph uses the
        standard single-observation inference region.
    outputs_root : str or pathlib.Path, default="runs"
        BayMorph output root.
    overwrite : bool, default=False
        Whether BayMorph products may overwrite existing run products.
    cutout_mode : {"auto", "fixed"}, default="auto"
        Image-cutout strategy.
    fixed_size : int, optional
        Fixed cutout size. Required only for ``cutout_mode="fixed"``.
    psf_type : {"empirical", "gaussian"}, default="empirical"
        PSF strategy.
    gaussian_fwhm_pixels : float, optional
        Gaussian PSF FWHM in detector pixels. When Gaussian mode is requested
        and this value is omitted, BayMorph uses FITS FWHM metadata only when
        its unit is explicitly declared as pixels, or as arcseconds with a
        valid celestial pixel scale.

    Returns
    -------
    dict
        Minimal user-facing BayMorph configuration.

    Notes
    -----
    This function does not estimate morphology parameters or construct priors
    from image data. Model priors remain controlled by the normal BayMorph
    configuration layer.

    The target position is intentionally explicit because it identifies which
    detected source BayMorph should analyse. Using ``"center"`` is therefore
    an explicit assumption by the caller, not target detection.
    """
    fits_path = Path(
        fits_path
    )

    if not fits_path.exists():
        raise FileNotFoundError(
            f"FITS file not found: {fits_path}"
        )

    observation = Observation.from_fits(
        fits_path
    )

    resolved_name = (
        fits_path.stem
        if object_name is None
        else str(
            object_name
        ).strip()
    )

    if not resolved_name:
        raise ValueError(
            "object_name must be non-empty."
        )

    pixel_scale = _finite_positive(
        observation.metadata.pixel_scale
    )

    normalized_target = _normalize_target_position(
        target_position
    )

    cutout = _cutout_config(
        cutout_mode,
        fixed_size=fixed_size,
    )

    normalized_psf_type = str(
        psf_type
    ).strip().lower()

    if normalized_psf_type == "empirical":
        psf = _empirical_psf_config()

    elif normalized_psf_type == "gaussian":
        psf = _gaussian_psf_config(
            observation,
            pixel_scale=pixel_scale,
            gaussian_fwhm_pixels=gaussian_fwhm_pixels,
        )

    else:
        raise ValueError(
            "psf_type must be 'empirical' or 'gaussian'."
        )

    model_name = str(
        model_name
    ).strip()

    if not model_name:
        raise ValueError(
            "model_name must be non-empty."
        )

    sampler_config = _normalize_sampler_config(
        sampler
    )

    normalized_model_priors = _normalize_model_priors(
        model_priors
    )

    fit_config: dict[str, Any] = {
        "models": [
            model_name,
        ],
    }

    if normalized_model_priors:
        fit_config["priors"] = {
            model_name: normalized_model_priors,
        }

    if likelihood is not None:
        if not isinstance(likelihood, Mapping):
            raise TypeError(
                "likelihood must be a mapping or None."
            )

        fit_config["likelihood"] = deepcopy(
            dict(likelihood)
        )

    if sampler_config is not None:
        fit_config["sampler"] = sampler_config

    config: dict[str, Any] = {
        "object": {
            "name": resolved_name,
        },
        "data": {
            "image_path": str(
                fits_path
            ),
            "pixel_scale": pixel_scale,
            "target_position": normalized_target,
            "cutout": cutout,
        },
        "outputs": {
            "root": str(
                Path(
                    outputs_root
                )
            ),
            "overwrite": bool(
                overwrite
            ),
        },
        "psf": psf,
        "fit": fit_config,
    }

    resolved = resolve_config(
        config
    )

    if _require_complete_priors:
        validate_fit_priors(
            resolved
        )

    if output_path is not None:
        serialized_config = deepcopy(
            config
        )

        serialized_config[
            "data"
        ][
            "image_path"
        ] = str(
            fits_path.resolve()
        )

        serialized_outputs_root = Path(
            outputs_root
        )

        if not serialized_outputs_root.is_absolute():
            serialized_outputs_root = (
                Path.cwd()
                / serialized_outputs_root
            )

        serialized_config[
            "outputs"
        ][
            "root"
        ] = str(
            serialized_outputs_root.resolve()
        )

        _write_yaml(
            Path(
                output_path
            ),
            serialized_config,
        )

        config = serialized_config

    return config


def _normalize_observations(
    observations: Mapping[str, str | Path],
) -> dict[str, Path]:
    """Validate and normalize a multi-observation FITS mapping."""
    if (
        not isinstance(
            observations,
            Mapping,
        )
        or not observations
    ):
        raise ValueError(
            "observations must be a non-empty mapping."
        )

    normalized: dict[str, Path] = {}

    for name, fits_path in observations.items():
        if (
            not isinstance(
                name,
                str,
            )
            or not name
        ):
            raise ValueError(
                "Observation identifiers must be non-empty strings."
            )

        path = Path(
            fits_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"FITS file not found for observation '{name}': {path}"
            )

        normalized[
            name
        ] = path

    return normalized


def _normalize_target_positions(
    target_positions: Mapping[str, TargetPosition],
    *,
    observation_names: tuple[str, ...],
) -> dict[str, TargetPosition]:
    """Require exactly one explicit target seed per observation."""
    if not isinstance(
        target_positions,
        Mapping,
    ):
        raise TypeError(
            "target_positions must be a mapping."
        )

    supplied = set(
        target_positions
    )
    expected = set(
        observation_names
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

        raise ValueError(
            "target_positions must define exactly the configured "
            f"observations. Missing: {missing}. Extra: {extra}."
        )

    return {
        name: target_positions[
            name
        ]
        for name in observation_names
    }


def _normalize_observation_options(
    observation_options: Mapping[str, Mapping[str, Any]] | None,
    *,
    observation_names: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Normalize optional per-observation configuration overrides."""
    if observation_options is None:
        return {
            name: {}
            for name in observation_names
        }

    if not isinstance(
        observation_options,
        Mapping,
    ):
        raise TypeError(
            "observation_options must be a mapping."
        )

    unknown_observations = sorted(
        set(
            observation_options
        ).difference(
            observation_names
        )
    )

    if unknown_observations:
        raise ValueError(
            "observation_options contains unknown observation identifiers: "
            f"{unknown_observations}."
        )

    normalized = {
        name: {}
        for name in observation_names
    }

    for name, options in observation_options.items():
        if not isinstance(
            options,
            Mapping,
        ):
            raise TypeError(
                f"Options for observation '{name}' must be a mapping."
            )

        unknown_keys = sorted(
            set(
                options
            ).difference(
                _OBSERVATION_OPTION_KEYS
            )
        )

        if unknown_keys:
            raise ValueError(
                f"Unsupported options for observation '{name}': "
                f"{unknown_keys}."
            )

        normalized[
            name
        ] = dict(
            options
        )

    return normalized


def _safe_observation_config_filename(
    observation_name: str,
) -> str:
    """Return a filesystem-safe YAML filename for one observation."""
    stem = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        observation_name,
    ).strip(
        "._"
    )

    if not stem:
        stem = "observation"

    return f"{stem}.yaml"


def _observation_config_paths(
    observation_names: tuple[str, ...],
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Return collision-free observation YAML paths."""
    paths = {
        name: (
            output_dir
            / _safe_observation_config_filename(
                name
            )
        )
        for name in observation_names
    }

    filenames = [
        path.name
        for path in paths.values()
    ]

    if len(
        filenames
    ) != len(
        set(
            filenames
        )
    ):
        raise ValueError(
            "Observation identifiers produce colliding configuration "
            "filenames. Rename the affected observation identifiers."
        )

    return paths


def _validate_sky_frame_wcs(
    observations: Mapping[str, Path],
) -> None:
    """Require a WCS for every sky-frame observation."""
    for name, fits_path in observations.items():
        observation = Observation.from_fits(
            fits_path
        )

        if observation.metadata.wcs is None:
            raise ValueError(
                "Sky-frame multi-observation configuration requires a "
                f"valid WCS for observation '{name}'."
            )


def _relative_config_path(
    path: Path,
    *,
    joint_directory: Path,
) -> str:
    """Return a YAML path relative to the declaring joint configuration."""
    relative = os.path.relpath(
        path,
        start=joint_directory,
    )

    return Path(
        relative
    ).as_posix()


def make_multi_observation_config(
    observations: Mapping[str, str | Path],
    *,
    target_positions: Mapping[str, TargetPosition],
    parameters: Mapping[str, Any],
    output_dir: str | Path,
    joint_output_path: str | Path | None = None,
    analysis_name: str = "joint_analysis",
    model_name: str = "single_sersic",
    parameter_frame: str = "sky",
    sampler: str | Mapping[str, Any] | None = "nautilus",
    likelihood: Mapping[str, Any] | None = None,
    outputs_root: str | Path = "runs",
    overwrite: bool = False,
    observation_options: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate observation YAML files and one joint BayMorph configuration.

    Parameters
    ----------
    observations : mapping
        Mapping from opaque user-defined observation identifiers to FITS
        image paths.
    target_positions : mapping
        Explicit target-identification seed for every observation.
    parameters : mapping
        Complete multi-observation parameter specification. Coupling and
        typed priors must be declared explicitly for every model parameter.
    output_dir : str or pathlib.Path
        Directory in which individual observation YAML files are written.
    joint_output_path : str or pathlib.Path, optional
        Joint YAML path. Defaults to ``output_dir / "joint.yaml"``.
    analysis_name : str, default="joint_analysis"
        Name of the joint analysis output tree.
    model_name : str, default="single_sersic"
        Registered BayMorph morphology model.
    parameter_frame : {"sky", "detector"}, default="sky"
        Inferential parameter frame. Sky coordinates are preferred for
        heterogeneous native-grid observations of the same object.
    sampler : mapping, optional
        Explicit inference-backend configuration. If omitted, the standard
        resolved BayMorph sampler configuration is reused.
    likelihood : mapping, optional
        Joint image-likelihood configuration. The default is a normalized
        Gaussian image likelihood.
    outputs_root : str or pathlib.Path, default="runs"
        Shared output root for observation preparation and joint products.
    overwrite : bool, default=False
        Default overwrite policy for individual observation products.
    observation_options : mapping, optional
        Per-observation overrides accepted by ``make_config_from_fits``:
        ``object_name``, ``overwrite``, ``cutout_mode``, ``fixed_size``,
        ``psf_type``, ``gaussian_fwhm_pixels``, and ``likelihood``.

    Returns
    -------
    dict
        Joint multi-observation configuration.

    Notes
    -----
    Observation identifiers remain opaque and are never interpreted as bands,
    instruments, filters, or epochs.

    The function does not estimate morphological priors from the image data.
    Parameter coupling and prior distributions are supplied explicitly through
    ``parameters``.

    When ``parameter_frame="sky"``, every FITS observation must contain a
    valid WCS because the production inference problem transforms the shared
    physical geometry onto each native detector grid.
    """
    normalized_observations = _normalize_observations(
        observations
    )

    observation_names = tuple(
        normalized_observations
    )

    normalized_targets = _normalize_target_positions(
        target_positions,
        observation_names=observation_names,
    )

    normalized_options = _normalize_observation_options(
        observation_options,
        observation_names=observation_names,
    )

    if not isinstance(
        parameters,
        Mapping,
    ):
        raise TypeError(
            "parameters must be a mapping."
        )

    analysis_name = str(
        analysis_name
    ).strip()

    if not analysis_name:
        raise ValueError(
            "analysis_name must be non-empty."
        )
    model_name = str(
        model_name
    ).strip()

    if not model_name:
        raise ValueError(
            "model_name must be non-empty."
        )

    normalized_sampler = _normalize_sampler_config(
        sampler
    )


    parameter_frame = str(
        parameter_frame
    ).strip().lower()

    if parameter_frame not in {
        "sky",
        "detector",
    }:
        raise ValueError(
            "parameter_frame must be 'sky' or 'detector'."
        )

    if parameter_frame == "sky":
        _validate_sky_frame_wcs(
            normalized_observations
        )

    output_dir = Path(
        output_dir
    )

    joint_path = (
        output_dir
        / "joint.yaml"
        if joint_output_path is None
        else Path(
            joint_output_path
        )
    )

    observation_paths = _observation_config_paths(
        observation_names,
        output_dir=output_dir,
    )

    observation_configs: dict[
        str,
        dict[str, Any],
    ] = {}

    for name in observation_names:
        options = normalized_options[
            name
        ]

        observation_configs[
            name
        ] = make_config_from_fits(
            normalized_observations[
                name
            ],
            target_position=normalized_targets[
                name
            ],
            object_name=options.get(
                "object_name"
            ),
            model_name=model_name,
            likelihood=options.get(
                "likelihood"
            ),
            sampler=normalized_sampler,
            _require_complete_priors=False,
            outputs_root=outputs_root,
            overwrite=bool(
                options.get(
                    "overwrite",
                    overwrite,
                )
            ),
            cutout_mode=str(
                options.get(
                    "cutout_mode",
                    "auto",
                )
            ),
            fixed_size=options.get(
                "fixed_size"
            ),
            psf_type=str(
                options.get(
                    "psf_type",
                    "empirical",
                )
            ),
            gaussian_fwhm_pixels=options.get(
                "gaussian_fwhm_pixels"
            ),
        )

    object_names = [
        config[
            "object"
        ][
            "name"
        ]
        for config in observation_configs.values()
    ]

    if len(
        object_names
    ) != len(
        set(
            object_names
        )
    ):
        raise ValueError(
            "Individual observation configurations must have distinct "
            "object names so their preparation products do not collide. "
            "Use observation_options to provide explicit object_name values."
        )

    if normalized_sampler is None:
        first_config = observation_configs[
            observation_names[
                0
            ]
        ]

        resolved_first = resolve_config(
            first_config
        )

        sampler_config = deepcopy(
            resolved_first[
                "fit"
            ][
                "sampler"
            ]
        )

    else:
        sampler_config = deepcopy(
            normalized_sampler
        )

    if likelihood is None:
        likelihood_config = {
            "type": "gaussian",
            "include_normalization": True,
        }

    else:
        if not isinstance(
            likelihood,
            Mapping,
        ):
            raise TypeError(
                "likelihood must be a mapping."
            )

        likelihood_config = deepcopy(
            dict(
                likelihood
            )
        )

    joint_config: dict[str, Any] = {
        "observations": {
            name: {
                "config_path": _relative_config_path(
                    observation_paths[
                        name
                    ],
                    joint_directory=joint_path.parent,
                ),
            }
            for name in observation_names
        },
        "outputs": {
            "root": str(
                Path(
                    outputs_root
                )
            ),
            "analysis_name": analysis_name,
        },
        "fit": {
            "models": [
                str(
                    model_name
                )
            ],
            "likelihood": likelihood_config,
            "sampler": sampler_config,
            "multi_observation": {
                "parameter_frame": parameter_frame,
                "parameters": deepcopy(
                    dict(
                        parameters
                    )
                ),
            },
        },
    }

    # Validate the complete statistical contract before creating any YAML file.
    build_multi_observation_parameter_layout(
        joint_config
    )

    for name in observation_names:
        _write_yaml(
            observation_paths[
                name
            ],
            observation_configs[
                name
            ],
        )

    _write_yaml(
        joint_path,
        joint_config,
    )

    # Verify that all relative paths are consumable from the joint YAML.
    resolve_multi_observation_observation_configs(
        joint_config,
        base_dir=joint_path.parent,
    )

    return joint_config
