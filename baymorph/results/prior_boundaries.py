"""Posterior diagnostics for contact with declared prior boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from baymorph.results.samples import PosteriorSamples


_SYMBOLIC_BOUNDS = {
    "-pi": -np.pi,
    "-pi/2": -np.pi / 2.0,
    "0": 0.0,
    "pi/2": np.pi / 2.0,
    "pi": np.pi,
}

_PARAMETER_ALIASES = {
    "background_level": "sky",
}


def _numeric_bound(
    value: Any,
) -> float | None:
    """Resolve a finite numeric or supported symbolic prior bound."""
    if isinstance(
        value,
        str,
    ):
        normalized = value.strip().lower().replace(
            " ",
            "",
        )
        resolved = _SYMBOLIC_BOUNDS.get(
            normalized
        )

        return (
            None
            if resolved is None
            else float(
                resolved
            )
        )

    try:
        resolved = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(
        resolved
    ):
        return None

    return resolved


def _prior_measure_positions(
    values: np.ndarray,
    prior: Mapping[str, Any],
) -> np.ndarray | None:
    """Map samples to their position in a supported prior measure."""
    lower = _numeric_bound(
        prior.get(
            "lower"
        )
    )
    upper = _numeric_bound(
        prior.get(
            "upper"
        )
    )

    if (
        lower is None
        or upper is None
        or not lower < upper
    ):
        return None

    prior_type = str(
        prior.get(
            "type",
            "uniform",
        )
    ).strip().lower()

    samples = np.asarray(
        values,
        dtype=float,
    )

    if prior_type == "uniform":
        return (
            samples
            - lower
        ) / (
            upper
            - lower
        )

    if prior_type == "log_uniform":
        positions = np.full(
            samples.shape,
            np.nan,
            dtype=float,
        )

        positive = (
            samples > 0.0
        )

        if lower <= 0.0:
            return None

        positions[
            positive
        ] = (
            np.log(
                samples[
                    positive
                ]
                / lower
            )
            / np.log(
                upper
                / lower
            )
        )

        return positions

    return None


def audit_posterior_prior_boundaries(
    posterior: PosteriorSamples,
    priors: Mapping[str, Mapping[str, Any]],
    *,
    edge_fraction: float = 1.0e-3,
    mass_threshold: float = 1.0e-2,
    parameter_aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Measure weighted posterior mass near supported prior boundaries.

    Parameters
    ----------
    posterior : PosteriorSamples
        Weighted backend-neutral posterior samples.
    priors : mapping
        Prior configurations keyed by model parameter name.
    edge_fraction : float, default=1e-3
        Width of each inspected edge in prior-measure coordinates.
    mass_threshold : float, default=1e-2
        Posterior-mass threshold used to declare boundary contact.
    parameter_aliases : mapping, optional
        Mapping from posterior parameter names to configuration prior names.

    Returns
    -------
    dict
        JSON-compatible overall and per-parameter diagnostics.

    Notes
    -----
    A boundary-contact status does not make the numerical evidence undefined.
    It indicates that posterior summaries and evidence interpretation may be
    sensitive to the declared prior support.
    """
    if not isinstance(
        posterior,
        PosteriorSamples,
    ):
        raise TypeError(
            "posterior must be a PosteriorSamples instance."
        )

    if not (
        0.0
        < edge_fraction
        < 0.5
    ):
        raise ValueError(
            "edge_fraction must lie in (0, 0.5)."
        )

    if not (
        0.0
        < mass_threshold
        <= 1.0
    ):
        raise ValueError(
            "mass_threshold must lie in (0, 1]."
        )

    aliases = dict(
        _PARAMETER_ALIASES
    )

    if parameter_aliases is not None:
        aliases.update(
            {
                str(key): str(value)
                for key, value in parameter_aliases.items()
            }
        )

    parameter_diagnostics: dict[str, dict[str, Any]] = {}
    flagged_parameters: list[str] = []

    for column, parameter_name in enumerate(
        posterior.parameter_names
    ):
        prior_name = aliases.get(
            parameter_name,
            parameter_name,
        )

        prior = priors.get(
            prior_name
        )

        if not isinstance(
            prior,
            Mapping,
        ):
            continue

        positions = _prior_measure_positions(
            posterior.samples[
                :,
                column,
            ],
            prior,
        )

        if positions is None:
            continue

        valid = (
            np.isfinite(
                positions
            )
            & np.isfinite(
                posterior.weights
            )
            & (
                posterior.weights
                >= 0.0
            )
        )

        valid_weight = float(
            np.sum(
                posterior.weights[
                    valid
                ]
            )
        )

        if valid_weight <= 0.0:
            continue

        normalized_weights = (
            posterior.weights[
                valid
            ]
            / valid_weight
        )
        valid_positions = positions[
            valid
        ]

        lower_mass = float(
            np.sum(
                normalized_weights[
                    valid_positions
                    <= edge_fraction
                ]
            )
        )
        upper_mass = float(
            np.sum(
                normalized_weights[
                    valid_positions
                    >= (
                        1.0
                        - edge_fraction
                    )
                ]
            )
        )
        outside_mass = float(
            np.sum(
                normalized_weights[
                    (
                        valid_positions
                        < 0.0
                    )
                    | (
                        valid_positions
                        > 1.0
                    )
                ]
            )
        )

        boundary_contact = bool(
            lower_mass
            >= mass_threshold
            or upper_mass
            >= mass_threshold
            or outside_mass
            > 0.0
        )

        parameter_diagnostics[
            parameter_name
        ] = {
            "prior_name": prior_name,
            "prior_type": str(
                prior.get(
                    "type",
                    "uniform",
                )
            ),
            "lower_edge_mass": lower_mass,
            "upper_edge_mass": upper_mass,
            "outside_prior_mass": outside_mass,
            "boundary_contact": boundary_contact,
        }

        if boundary_contact:
            flagged_parameters.append(
                parameter_name
            )

    if not parameter_diagnostics:
        status = "not_evaluated"
    elif flagged_parameters:
        status = "boundary_contact"
    else:
        status = "clear"

    return {
        "status": status,
        "edge_fraction": float(
            edge_fraction
        ),
        "mass_threshold": float(
            mass_threshold
        ),
        "flagged_parameters": flagged_parameters,
        "evidence_prior_sensitivity_warning": bool(
            flagged_parameters
        ),
        "parameters": parameter_diagnostics,
    }
