"""Model-agnostic parameter construction for multi-observation inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from baymorph.core.observation import ObservationMetadata
from baymorph.fitting.priors import UniformPrior
from baymorph.models import get_model
from baymorph.models.geometry import (
    ModelGeometry,
    get_model_geometry,
    sky_parameter_names,
)
from baymorph.observation.background import (
    BACKGROUND_PARAMETER_NAMES,
    background_parameter_names,
)
from baymorph.pipeline.config import DEFAULT_CONFIG
from baymorph.pipeline.opening import read_observation_metadata
from baymorph.pipeline.prior_config import parse_prior_config

Bounds = Sequence[float]


def _uniform(lower: float, upper: float) -> dict[str, float | str]:
    """Return one YAML-compatible uniform prior."""
    lower = float(lower)
    upper = float(upper)

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("Uniform prior bounds must be finite and ordered.")

    return {"type": "uniform", "lower": lower, "upper": upper}


def _typed_uniform_bounds(
    config: Mapping[str, Any], *, references: Mapping[str, float] | None = None
) -> tuple[float, float]:
    """Resolve one typed uniform prior into finite bounds."""
    prior = parse_prior_config(config, references=references)

    if not isinstance(prior, UniformPrior):
        raise ValueError(
            "This automatic parameter transformation requires " "a uniform prior."
        )

    return (float(prior.lower), float(prior.upper))


def _center_default_offsets(
    config: Mapping[str, Any], *, reference_name: str
) -> tuple[float, float]:
    """Return detector-coordinate offsets around an explicit target seed."""
    if config.get("relative_to") != reference_name:
        raise ValueError(
            "Automatic centre priors require a relative uniform prior "
            f"with relative_to={reference_name!r}."
        )

    return _typed_uniform_bounds(config, references={reference_name: 0.0})


def _normalize_bounds(
    value: Bounds, *, parameter_name: str, observation_id: str | None = None
) -> tuple[float, float]:
    """Validate one explicit prior interval."""
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        context = (
            f" for observation {observation_id!r}" if observation_id is not None else ""
        )

        raise ValueError(
            f"Prior bounds for parameter {parameter_name!r}{context} "
            "must contain exactly two values."
        )

    lower = float(value[0])
    upper = float(value[1])

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(
            f"Prior bounds for parameter {parameter_name!r} "
            "must be finite and ordered."
        )

    return lower, upper


def _normalize_target_center(
    value: Sequence[float], *, observation_id: str
) -> tuple[float, float]:
    """Validate one detector-space target centre."""
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise ValueError(
            f"Target centre for observation {observation_id!r} "
            "must contain exactly two coordinates."
        )

    x = float(value[0])
    y = float(value[1])

    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError(
            f"Target centre for observation {observation_id!r} "
            "must contain finite coordinates."
        )

    return x, y


def _geometry_parameter_sets(
    geometry: ModelGeometry,
) -> tuple[set[str], set[str], set[str]]:
    """Return radius, axis-ratio, and position-angle parameter sets."""
    radii = {structure.radius for structure in geometry.elliptical_structures}

    axis_ratios = {structure.axis_ratio for structure in geometry.elliptical_structures}

    position_angles = {
        structure.position_angle for structure in geometry.elliptical_structures
    }

    return (radii, axis_ratios, position_angles)


def _sky_name(detector_name: str, *, geometry: ModelGeometry) -> str:
    """Return the sky-frame name corresponding to one detector parameter."""
    if detector_name == geometry.centre_x:
        return "ra"

    if detector_name == geometry.centre_y:
        return "dec"

    radii, _, position_angles = _geometry_parameter_sets(geometry)

    if detector_name in radii:
        if detector_name == "re":
            return "re_arcsec"

        return "re_arcsec_" + detector_name.removeprefix("re_")

    if detector_name in position_angles:
        if detector_name == "theta":
            return "position_angle"

        return "position_angle_" + detector_name.removeprefix("theta_")

    return detector_name


def _default_prior_entry(
    *, detector_name: str, model_defaults: Mapping[str, Any], geometry: ModelGeometry
) -> Mapping[str, Any] | None:
    """Return the default prior entry, including geometry aliases."""
    direct = model_defaults.get(detector_name)

    if isinstance(direct, Mapping):
        return direct

    if detector_name in BACKGROUND_PARAMETER_NAMES["plane"]:
        background_default = DEFAULT_CONFIG["fit"]["background"]["priors"].get(
            detector_name
        )

        if isinstance(background_default, Mapping):
            return background_default

    _, axis_ratios, position_angles = _geometry_parameter_sets(geometry)

    alias = None

    if detector_name in axis_ratios:
        alias = "q"

    elif detector_name in position_angles:
        alias = "theta"

    if alias is None:
        return None

    candidate = model_defaults.get(alias)

    if isinstance(candidate, Mapping):
        return candidate

    return None


def _coupled_prior(
    prior: Mapping[str, Any], *, observation_ids: tuple[str, ...], is_shared: bool
) -> dict[str, Any]:
    """Apply one prior either globally or independently."""
    if is_shared:
        return {"coupling": "shared", "prior": dict(prior)}

    return {
        "coupling": "independent",
        "priors": {observation_id: dict(prior) for observation_id in observation_ids},
    }


def _explicit_bounds_specification(
    value: Any,
    *,
    parameter_name: str,
    observation_ids: tuple[str, ...],
    is_shared: bool,
) -> dict[str, Any]:
    """Build a parameter specification from explicit uniform bounds."""
    if is_shared:
        lower, upper = _normalize_bounds(value, parameter_name=parameter_name)

        return {"coupling": "shared", "prior": _uniform(lower, upper)}

    if isinstance(value, Mapping):
        if set(value) != set(observation_ids):
            raise ValueError(
                f"Explicit bounds for independent parameter "
                f"{parameter_name!r} must define exactly the configured "
                "observations."
            )

        priors = {
            observation_id: _uniform(
                *_normalize_bounds(
                    value[observation_id],
                    parameter_name=parameter_name,
                    observation_id=observation_id,
                )
            )
            for observation_id in observation_ids
        }

    else:
        lower, upper = _normalize_bounds(value, parameter_name=parameter_name)

        priors = {
            observation_id: _uniform(lower, upper) for observation_id in observation_ids
        }

    return {"coupling": "independent", "priors": priors}


def _sky_center_bounds(
    *,
    metadata: ObservationMetadata,
    target_center: tuple[float, float],
    x_offsets: tuple[float, float],
    y_offsets: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Convert a local detector-space centre box into RA/Dec bounds."""
    wcs = metadata.wcs

    if wcs is None:
        raise ValueError("Sky-frame parameter construction requires a celestial WCS.")

    x0, y0 = target_center
    x_lower, x_upper = x_offsets
    y_lower, y_upper = y_offsets

    x_values = np.asarray([x0 + x_lower, x0, x0 + x_upper], dtype=float)

    y_values = np.asarray([y0 + y_lower, y0, y0 + y_upper], dtype=float)

    x_grid, y_grid = np.meshgrid(x_values, y_values)

    ra, dec = wcs.pixel_to_world_values(x_grid.ravel(), y_grid.ravel())

    center_ra, center_dec = wcs.pixel_to_world_values(x0, y0)

    ra = np.asarray(ra, dtype=float)

    dec = np.asarray(dec, dtype=float)

    center_ra = float(center_ra)
    center_dec = float(center_dec)

    if (
        not np.all(np.isfinite(ra))
        or not np.all(np.isfinite(dec))
        or not np.isfinite(center_ra)
        or not np.isfinite(center_dec)
    ):
        raise ValueError("WCS transformation produced non-finite sky coordinates.")

    delta_ra = (ra - center_ra + 180.0) % 360.0 - 180.0

    ra_lower = float(center_ra + np.min(delta_ra))

    ra_upper = float(center_ra + np.max(delta_ra))

    dec_lower = float(np.min(dec))

    dec_upper = float(np.max(dec))

    if ra_lower < 0.0 or ra_upper >= 360.0:
        raise ValueError(
            "Automatic RA bounds cross the 0/360 degree boundary. "
            "Supply explicit prior_bounds for 'ra'."
        )

    return ((ra_lower, ra_upper), (dec_lower, dec_upper))


def _center_parameter_specifications(
    *,
    observations: Mapping[str, ObservationMetadata],
    target_centers: Mapping[str, tuple[float, float]],
    observation_ids: tuple[str, ...],
    x_offsets: tuple[float, float],
    y_offsets: tuple[float, float],
    shared: set[str],
) -> dict[str, dict[str, Any]]:
    """Build sky-coordinate centre specifications."""
    bounds = {
        observation_id: _sky_center_bounds(
            metadata=observations[observation_id],
            target_center=target_centers[observation_id],
            x_offsets=x_offsets,
            y_offsets=y_offsets,
        )
        for observation_id in observation_ids
    }

    ra_bounds = {
        observation_id: bounds[observation_id][0] for observation_id in observation_ids
    }

    dec_bounds = {
        observation_id: bounds[observation_id][1] for observation_id in observation_ids
    }

    specifications: dict[str, dict[str, Any]] = {}

    for name, per_observation_bounds in (("ra", ra_bounds), ("dec", dec_bounds)):
        if name in shared:
            lower = min(value[0] for value in per_observation_bounds.values())

            upper = max(value[1] for value in per_observation_bounds.values())

            specifications[name] = {
                "coupling": "shared",
                "prior": _uniform(lower, upper),
            }

        else:
            specifications[name] = {
                "coupling": "independent",
                "priors": {
                    observation_id: _uniform(*per_observation_bounds[observation_id])
                    for observation_id in observation_ids
                },
            }

    return specifications


def make_multi_parameters(
    model: str,
    observations: Mapping[str, str | Path],
    *,
    target_centers: Mapping[str, Sequence[float]],
    prior_bounds: Mapping[str, Any] | None = None,
    shared: Sequence[str],
    parameter_frame: str = "sky",
    background_model: str = "constant",
) -> dict[str, Any]:
    """Build a model-agnostic multi-observation parameter specification.

    Parameters
    ----------
    model : str
        Registered BayMorph model name.
    observations : mapping
        Mapping from opaque observation identifiers to FITS paths.
    target_centers : mapping
        Explicit detector-space target centres for every observation.
    prior_bounds : mapping, optional
        Explicit uniform-prior bounds keyed by parameter name. Shared
        parameters accept one ``(lower, upper)`` pair. Independent parameters
        accept either one common pair or an observation-indexed mapping.
        Bounds are required whenever BayMorph has no standard
        data-independent prior for the parameter.
    shared : sequence of str
        Parameters explicitly coupled across observations. Every remaining
        parameter is independent.
    parameter_frame : {"sky", "detector"}, default="sky"
        Parameter frame used by the joint inference.
    background_model : {"constant", "plane"}, default="constant"
        Residual-background family fitted independently in every
        observation. Its parameters follow the morphology contract and
        always use ``independent`` coupling (AD-044).

    Returns
    -------
    dict
        YAML-compatible ``fit.multi_observation.parameters`` mapping.

    Notes
    -----
    This function is model-agnostic. Parameter names and geometric roles are
    obtained from the registered model definition.

    Data-dependent prior scales are never inferred here. Parameters
    without a standard data-independent prior require explicit
    ``prior_bounds``.

    Detector-space radius defaults are converted to arcseconds independently
    for each observation in sky-frame inference. A shared sky-frame radius
    therefore requires explicit bounds because detector-pixel defaults do not
    define a unique common physical prior.
    """
    definition = get_model(model)

    geometry = get_model_geometry(model)

    if not isinstance(observations, Mapping) or not observations:
        raise ValueError("observations must be a non-empty mapping.")

    observation_ids = tuple(observations)

    expected_ids = set(observation_ids)

    if set(target_centers) != expected_ids:
        raise ValueError(
            "target_centers must define exactly the configured observations."
        )

    normalized_centers = {
        observation_id: _normalize_target_center(
            target_centers[observation_id], observation_id=observation_id
        )
        for observation_id in observation_ids
    }

    parameter_frame = str(parameter_frame).strip().lower()

    if parameter_frame not in {"sky", "detector"}:
        raise ValueError("parameter_frame must be 'sky' or 'detector'.")

    background_model = str(background_model).strip().lower()
    background_names = background_parameter_names(background_model)

    parameter_names = (
        sky_parameter_names(model)
        if parameter_frame == "sky"
        else tuple(definition.parameter_names)
    ) + background_names

    shared_set = {str(name) for name in shared}

    shared_background = sorted(shared_set.intersection(background_names))

    if shared_background:
        raise ValueError(
            "Residual-background parameters are observation-specific and "
            f"cannot be shared: {shared_background}."
        )

    unsupported_shared = sorted(shared_set.difference(parameter_names))

    if unsupported_shared:
        raise ValueError(
            "shared contains parameters outside the model contract: "
            f"{unsupported_shared}."
        )

    overrides = {} if prior_bounds is None else dict(prior_bounds)

    unknown_overrides = sorted(set(overrides).difference(parameter_names))

    if unknown_overrides:
        raise ValueError(
            "prior_bounds contains parameters outside the model contract: "
            f"{unknown_overrides}."
        )

    model_defaults = DEFAULT_CONFIG["fit"]["priors"].get(model, {})

    radii, _, _ = _geometry_parameter_sets(geometry)

    detector_to_frame = {
        detector_name: (
            _sky_name(detector_name, geometry=geometry)
            if parameter_frame == "sky"
            else detector_name
        )
        for detector_name in definition.parameter_names
    }

    detector_to_frame.update({name: name for name in background_names})

    frame_to_detector = {
        frame_name: detector_name
        for detector_name, frame_name in detector_to_frame.items()
    }

    loaded_observations: dict[str, ObservationMetadata] = {}

    if parameter_frame == "sky":
        loaded_observations = {
            observation_id: read_observation_metadata(
                Path(observations[observation_id])
            )
            for observation_id in observation_ids
        }

        for observation_id, metadata in loaded_observations.items():
            if metadata.wcs is None:
                raise ValueError(
                    "Sky-frame parameter construction requires a celestial "
                    f"WCS for observation {observation_id!r}."
                )

    specifications: dict[str, dict[str, Any]] = {}

    if parameter_frame == "sky":
        x_default = _default_prior_entry(
            detector_name=geometry.centre_x,
            model_defaults=model_defaults,
            geometry=geometry,
        )

        y_default = _default_prior_entry(
            detector_name=geometry.centre_y,
            model_defaults=model_defaults,
            geometry=geometry,
        )

        if "ra" not in overrides or "dec" not in overrides:
            if x_default is None or y_default is None:
                raise ValueError(
                    "Automatic sky-centre priors require detector-centre "
                    "defaults. Supply explicit prior_bounds for "
                    "'ra' and 'dec'."
                )

            x_offsets = _center_default_offsets(x_default, reference_name="target_x")

            y_offsets = _center_default_offsets(y_default, reference_name="target_y")

            center_specs = _center_parameter_specifications(
                observations=loaded_observations,
                target_centers=normalized_centers,
                observation_ids=observation_ids,
                x_offsets=x_offsets,
                y_offsets=y_offsets,
                shared=shared_set,
            )

            for name in ("ra", "dec"):
                if name not in overrides:
                    specifications[name] = center_specs[name]

    for parameter_name in parameter_names:
        if parameter_name in specifications:
            continue

        is_shared = parameter_name in shared_set

        if parameter_name in overrides:
            specifications[parameter_name] = _explicit_bounds_specification(
                overrides[parameter_name],
                parameter_name=parameter_name,
                observation_ids=observation_ids,
                is_shared=is_shared,
            )
            continue

        detector_name = frame_to_detector[parameter_name]

        default = _default_prior_entry(
            detector_name=detector_name,
            model_defaults=model_defaults,
            geometry=geometry,
        )

        if default is None:
            raise ValueError(
                f"No standard prior is available for parameter "
                f"{parameter_name!r}. Supply explicit prior_bounds."
            )

        if "type" in default:
            if "relative_to" in default:
                if parameter_frame != "detector":
                    raise ValueError(
                        f"Relative detector prior for parameter "
                        f"{parameter_name!r} cannot be used directly in "
                        "sky-frame inference."
                    )

                if detector_name == geometry.centre_x:
                    reference_name = "target_x"
                    center_index = 0

                elif detector_name == geometry.centre_y:
                    reference_name = "target_y"
                    center_index = 1

                else:
                    raise ValueError(
                        f"Relative prior for parameter "
                        f"{parameter_name!r} is unsupported."
                    )

                lower_offset, upper_offset = _center_default_offsets(
                    default, reference_name=reference_name
                )

                per_observation = {
                    observation_id: (
                        normalized_centers[observation_id][center_index] + lower_offset,
                        normalized_centers[observation_id][center_index] + upper_offset,
                    )
                    for observation_id in observation_ids
                }

                if is_shared:
                    lower = min(value[0] for value in per_observation.values())
                    upper = max(value[1] for value in per_observation.values())

                    specifications[parameter_name] = {
                        "coupling": "shared",
                        "prior": _uniform(lower, upper),
                    }

                else:
                    specifications[parameter_name] = {
                        "coupling": "independent",
                        "priors": {
                            observation_id: _uniform(*per_observation[observation_id])
                            for observation_id in observation_ids
                        },
                    }

                continue

            if parameter_frame == "sky" and detector_name in radii:
                if is_shared:
                    raise ValueError(
                        f"Shared sky-frame radius "
                        f"{parameter_name!r} requires explicit "
                        "prior_bounds because detector-pixel defaults "
                        "do not define a unique common physical prior."
                    )

                lower, upper = _typed_uniform_bounds(default)

                radius_priors = {}

                for observation_id in observation_ids:
                    pixel_scale = loaded_observations[observation_id].pixel_scale

                    if (
                        pixel_scale is None
                        or not np.isfinite(pixel_scale)
                        or pixel_scale <= 0.0
                    ):
                        raise ValueError(
                            "Sky-frame radius construction requires a "
                            "positive pixel scale for observation "
                            f"{observation_id!r}."
                        )

                    radius_priors[observation_id] = _uniform(
                        lower * float(pixel_scale), upper * float(pixel_scale)
                    )

                specifications[parameter_name] = {
                    "coupling": "independent",
                    "priors": radius_priors,
                }

                continue

            parse_prior_config(default)

            specifications[parameter_name] = _coupled_prior(
                default, observation_ids=observation_ids, is_shared=is_shared
            )

            continue

        raise ValueError(
            f"Unsupported untyped standard prior specification for parameter "
            f"{parameter_name!r}. Standard priors must use the typed prior "
            "schema or explicit prior_bounds must be supplied."
        )

    return {
        parameter_name: specifications[parameter_name]
        for parameter_name in parameter_names
    }
