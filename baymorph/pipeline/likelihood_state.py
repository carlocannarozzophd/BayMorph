"""Likelihood-state preparation for real-image inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from baymorph.pipeline.noise_calibration import (
    correlated_noise_calibration_path,
    load_correlated_noise_calibration,
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
    metadata : dict, optional
        Scientific and numerical provenance of the likelihood state.
    """

    likelihood_type: str
    noise: np.ndarray
    correlation_kernel: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


def prepare_real_fit_likelihood_state(
    config: dict[str, Any],
    image_products,
) -> RealFitLikelihoodState:
    """Prepare fixed likelihood inputs on the prepared observation footprint.

    Independent-pixel likelihoods reuse the prepared noise map.

    The correlated Gaussian likelihood loads the persisted within-observation
    calibration on the same prepared observation footprint.

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
            metadata={
                "type": likelihood_type,
                "normalization_complete": True,
            },
        )

    include_normalization = bool(
        likelihood_options.get(
            "include_normalization",
            False,
        )
    )

    if (
        include_normalization
        and likelihood_options.get(
            "correlation_logdet"
        )
        is None
    ):
        raise ValueError(
            "Correlated Gaussian real-image inference cannot use "
            "include_normalization=True without an explicit "
            "correlation_logdet. Set include_normalization=False "
            "for posterior inference with the current fixed-noise "
            "calibration."
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

    noise = np.array(
        calibration.marginal_sigma,
        dtype=float,
        copy=True,
    )

    prepared_shape = np.asarray(
        image_products.prepared_data
    ).shape

    if noise.shape != prepared_shape:
        raise ValueError(
            "Correlated-noise marginal scale does not match the "
            "prepared observation footprint."
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
            "Correlated-noise marginal scale must be finite and "
            "strictly positive on the prepared observation footprint."
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
            calibration.metadata.get(
                "factorization",
                "C = S R S",
            )
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
        "marginal_model": (
            calibration.metadata.get(
                "marginal_model"
            )
        ),
        "max_lag": (
            calibration.metadata.get(
                "max_lag"
            )
        ),
        "normalization_complete": (
            include_normalization
        ),
        "fit_statistic": (
            "mahalanobis_chi2"
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
        metadata=metadata,
    )


def _normalize_likelihood_type(
    likelihood_type: str,
) -> str:
    """Normalize likelihood aliases used by the pipeline."""
    normalized = str(
        likelihood_type
    ).strip().lower()

    aliases = {
        "normal": (
            "gaussian"
        ),
        "correlated-gaussian": (
            "correlated_gaussian"
        ),
        "gaussian_correlated": (
            "correlated_gaussian"
        ),
        "gaussian-correlated": (
            "correlated_gaussian"
        ),
        "poisson": (
            "cash"
        ),
        "cash_statistic": (
            "cash"
        ),
        "poisson-gaussian": (
            "poisson_gaussian"
        ),
        "poisson+gaussian": (
            "poisson_gaussian"
        ),
    }

    return aliases.get(
        normalized,
        normalized,
    )
