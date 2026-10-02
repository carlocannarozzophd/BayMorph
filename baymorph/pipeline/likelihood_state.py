"""Likelihood-state preparation for real-image inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from baymorph.likelihood.factory import normalize_likelihood_type
from baymorph.likelihood.vecchia_gaussian import (
    VecchiaFactorization,
)
from baymorph.pipeline.noise_calibration import (
    correlated_noise_calibration_path,
    load_correlated_noise_calibration,
    reject_correlated_noise_calibration_settings,
    validate_correlated_noise_calibration_sources,
)
from baymorph.pipeline.vecchia_factorization import (
    load_vecchia_factorization_artifact,
    load_vecchia_factorization_manifest,
    vecchia_factorization_manifest_path,
    vecchia_factorization_path,
)


@dataclass(frozen=True)
class RealFitLikelihoodState:
    """Fixed observation-level inputs required by one real-image likelihood.

    Parameters
    ----------
    likelihood_type : str
        Normalized likelihood-family identifier.
    noise : ndarray
        Marginal noise field on the prepared observation footprint.
    correlation_kernel : ndarray, optional
        Fixed stationary two-dimensional correlation kernel.
    vecchia_factorization : VecchiaFactorization, optional
        Source-validated nearest-neighbour conditional factorization.
    metadata : dict, optional
        Scientific and numerical provenance of the likelihood state.
    """

    likelihood_type: str
    noise: np.ndarray
    correlation_kernel: np.ndarray | None = None
    vecchia_factorization: VecchiaFactorization | None = None
    metadata: dict[str, Any] | None = None


def require_complete_likelihood_normalization(
    likelihood_states: RealFitLikelihoodState | Mapping[str, RealFitLikelihoodState],
) -> None:
    """Require complete normalization before absolute-evidence inference."""
    if isinstance(likelihood_states, RealFitLikelihoodState):
        states = {"observation": likelihood_states}
    else:
        states = dict(likelihood_states)

    incomplete = []
    for observation, state in states.items():
        metadata = state.metadata or {}
        if metadata.get("normalization_complete", False) is not True:
            reason = metadata.get(
                "normalization_reason",
                "normalization status is not complete",
            )
            incomplete.append(f"{observation}: {reason}")

    if incomplete:
        raise ValueError(
            "Absolute Bayesian evidence requires a completely normalized "
            "likelihood for every observation. Incomplete: "
            + "; ".join(incomplete)
            + "."
        )


def prepare_real_fit_likelihood_state(
    config: dict[str, Any],
    image_products,
) -> RealFitLikelihoodState:
    """Prepare fixed likelihood inputs on the prepared observation footprint.

    Independent-pixel likelihoods reuse the prepared noise map.

    The correlated Gaussian likelihood of AD-047 also takes its marginal
    scale from the prepared noise map, and loads the persisted sky
    correlation of the same prepared observation footprint.

    Parameters
    ----------
    config : dict
        Resolved BayMorph configuration.
    image_products : object
        Prepared observation products defining the statistical image
        footprint.

    Returns
    -------
    RealFitLikelihoodState
        Fixed noise and correlation inputs required by the configured
        likelihood.
    """
    likelihood_cfg = config[
        "fit"
    ].get(
        "likelihood",
        {
            "type": "gaussian",
        },
    )

    if likelihood_cfg is None:
        likelihood_cfg = {
            "type": "gaussian",
        }

    if isinstance(
        likelihood_cfg,
        str,
    ):
        likelihood_type = likelihood_cfg
        likelihood_options: dict[
            str,
            Any,
        ] = {}
    else:
        likelihood_type = str(
            likelihood_cfg.get(
                "type",
                "gaussian",
            )
        )

        likelihood_options = dict(
            likelihood_cfg
        )

    likelihood_type = (
        _normalize_likelihood_type(
            likelihood_type
        )
    )

    if "noise" in likelihood_options:
        raise ValueError(
            "Real-image inference does not accept fit.likelihood.noise. "
            "The persisted prepared-noise product is authoritative."
        )

    if (
        likelihood_type
        != "correlated_gaussian"
    ):
        noise = np.array(
            image_products.noise,
            dtype=float,
            copy=True,
        )

        noise.setflags(
            write=False
        )

        return RealFitLikelihoodState(
            likelihood_type=likelihood_type,
            noise=noise,
            correlation_kernel=None,
            vecchia_factorization=None,
            metadata=_normalization_metadata(
                likelihood_type,
                likelihood_options,
            ),
        )

    include_normalization = bool(
        likelihood_options.get(
            "include_normalization",
            False,
        )
    )

    if "correlation_logdet" in likelihood_options:
        raise ValueError(
            "Real-image correlated Gaussian inference does not accept a "
            "configuration-supplied correlation_logdet."
        )

    factorization_config = likelihood_options.get(
        "factorization"
    )
    max_neighbors = 64
    min_conditional_variance = 1.0e-12

    if factorization_config is not None:
        if not isinstance(
            factorization_config,
            Mapping,
        ):
            raise TypeError(
                "Correlated Gaussian factorization must be a mapping."
            )

        factorization_options = dict(
            factorization_config
        )
        factorization_type = str(
            factorization_options.pop(
                "type",
                "",
            )
        ).strip().lower()

        if factorization_type != "nearest_neighbor":
            raise ValueError(
                "Correlated Gaussian factorization.type must be exactly "
                "'nearest_neighbor'."
            )

        max_neighbors = factorization_options.pop(
            "max_neighbors",
            64,
        )
        min_conditional_variance = factorization_options.pop(
            "min_conditional_variance",
            1.0e-12,
        )

        if factorization_options:
            unknown = ", ".join(
                sorted(
                    factorization_options
                )
            )
            raise ValueError(
                "Unsupported nearest-neighbour factorization options: "
                f"{unknown}."
            )

    elif include_normalization:
        raise ValueError(
            "Normalized correlated Gaussian evidence requires "
            "factorization.type='nearest_neighbor' and its validated "
            "immutable artifact."
        )

    reject_correlated_noise_calibration_settings(
        config
    )

    try:
        calibration = (
            load_correlated_noise_calibration(
                config
            )
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Correlated Gaussian fitting requires a persisted "
            "correlated-noise calibration. Build the correlated-noise "
            "calibration before fitting."
        ) from exc

    validate_correlated_noise_calibration_sources(
        calibration,
        image_products,
    )

    vecchia_factorization = None
    vecchia_manifest = None
    vecchia_path = None

    if factorization_config is not None:
        vecchia_factorization = (
            load_vecchia_factorization_artifact(
                config,
                max_neighbors=max_neighbors,
                min_conditional_variance=(
                    min_conditional_variance
                ),
            )
        )
        vecchia_manifest = (
            load_vecchia_factorization_manifest(
                config,
                max_neighbors=max_neighbors,
            )
        )
        vecchia_path = vecchia_factorization_path(
            config,
            max_neighbors=max_neighbors,
        )

        if (
            include_normalization
            and not np.isfinite(
                vecchia_factorization
                .correlation_logdet
            )
        ):
            raise ValueError(
                "Normalized nearest-neighbour inference requires a finite "
                "log determinant from the validated factorization."
            )

    noise = np.array(
        image_products.noise,
        dtype=float,
        copy=True,
    )

    prepared_shape = np.asarray(
        image_products.prepared_data
    ).shape

    if noise.shape != prepared_shape:
        raise ValueError(
            "The prepared noise map does not match the prepared "
            "observation footprint."
        )

    if (
        not np.all(
            np.isfinite(
                noise
            )
        )
        or np.any(
            noise <= 0.0
        )
    ):
        raise ValueError(
            "The prepared noise map must be finite and strictly positive "
            "on the prepared observation footprint."
        )

    kernel = np.array(
        calibration.correlation_kernel,
        dtype=float,
        copy=True,
    )

    noise.setflags(
        write=False
    )
    kernel.setflags(
        write=False
    )

    metadata = {
        "type": (
            "correlated_gaussian"
        ),
        "factorization": (
            "nearest_neighbor_conditional_gaussian"
            if vecchia_factorization is not None
            else calibration.metadata.get(
                "factorization",
                "C = S R S",
            )
        ),
        "approximate_likelihood": (
            vecchia_factorization is not None
        ),
        "calibration_path": str(
            correlated_noise_calibration_path(
                config
            )
        ),
        "calibration_schema_version": (
            calibration.metadata.get(
                "schema_version"
            )
        ),
        "correlation_model": (
            calibration.metadata.get(
                "correlation_model"
            )
        ),
        "correlation_estimator": (
            calibration.metadata.get(
                "estimator"
            )
        ),
        "marginal_model": (
            "prepared_sky_scale"
        ),
        "max_lag": (
            calibration.metadata.get(
                "max_lag"
            )
        ),
        "vecchia_factorization_path": (
            None
            if vecchia_path is None
            else str(
                vecchia_path
            )
        ),
        "vecchia_factorization_manifest_path": (
            None
            if vecchia_path is None
            else str(
                vecchia_factorization_manifest_path(
                    config,
                    max_neighbors=max_neighbors,
                )
            )
        ),
        "vecchia_factorization_payload_sha256": (
            None
            if vecchia_manifest is None
            else vecchia_manifest.get(
                "payload_sha256"
            )
        ),
        "vecchia_factorization_schema": (
            None
            if vecchia_manifest is None
            else vecchia_manifest.get(
                "schema"
            )
        ),
        "vecchia_factorization_schema_version": (
            None
            if vecchia_manifest is None
            else vecchia_manifest.get(
                "schema_version"
            )
        ),
        "vecchia_factorization_algorithm": (
            None
            if vecchia_manifest is None
            else vecchia_manifest.get(
                "algorithm"
            )
        ),
        "vecchia_max_neighbors": (
            None
            if vecchia_factorization is None
            else int(
                vecchia_factorization.max_neighbors
            )
        ),
        "normalization_complete": bool(
            include_normalization
            and vecchia_factorization is not None
        ),
        "evidence_scope": (
            "approximate_absolute"
            if (
                include_normalization
                and vecchia_factorization is not None
            )
            else "posterior_only"
        ),
        "normalization_reason": (
            "complete normalized nearest-neighbour approximate "
            "correlated Gaussian likelihood"
            if (
                include_normalization
                and vecchia_factorization is not None
            )
            else (
                "correlated Gaussian normalization is disabled"
            )
        ),
        "fit_statistic": (
            "nearest_neighbor_innovation_quadratic"
            if vecchia_factorization is not None
            else "mahalanobis_chi2"
        ),
        "residual_diagnostic": (
            "marginal_standardized"
        ),
    }

    return RealFitLikelihoodState(
        likelihood_type=(
            "correlated_gaussian"
        ),
        noise=noise,
        correlation_kernel=kernel,
        vecchia_factorization=vecchia_factorization,
        metadata=metadata,
    )


def _normalization_metadata(
    likelihood_type: str,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe whether one likelihood is complete for absolute evidence."""
    if likelihood_type == "gaussian":
        complete = bool(options.get("include_normalization", True))
        reason = (
            "complete independent Gaussian normalization"
            if complete
            else "independent Gaussian normalization is disabled"
        )
    elif likelihood_type == "cash":
        complete = bool(options.get("include_data_term", False))
        reason = (
            "complete Poisson data normalization"
            if complete
            else "Poisson data factorial term is omitted"
        )
    elif likelihood_type == "poisson_gaussian":
        complete = True
        reason = "complete marginalized Poisson-Gaussian normalization"
    else:
        complete = False
        reason = f"normalization contract is undefined for {likelihood_type!r}"

    return {
        "type": likelihood_type,
        "normalization_complete": complete,
        "evidence_scope": "absolute" if complete else "posterior_only",
        "normalization_reason": reason,
    }


_normalize_likelihood_type = normalize_likelihood_type
