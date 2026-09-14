"""Adapters between BayMorph inference products and validation results."""

from __future__ import annotations

from typing import Any

import numpy as np

from baymorph.results import InferenceResult, RealFitProducts
from baymorph.validation.experiment import RecoveryFit


def recovery_fit_from_inference_result(
    result: InferenceResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> RecoveryFit:
    """Convert a canonical BayMorph inference result into a recovery fit.

    Posterior medians are used as point estimates. The 16th and 84th
    posterior percentiles define the default recovery interval.

    Parameters
    ----------
    result : InferenceResult
        Canonical backend-neutral BayMorph inference result.
    metadata : dict, optional
        Additional validation metadata.

    Returns
    -------
    RecoveryFit
        Validation-compatible posterior recovery summary.
    """
    if not isinstance(
        result,
        InferenceResult,
    ):
        raise TypeError(
            "result must be an InferenceResult instance."
        )

    parameter_names = tuple(
        result.parameter_names
    )

    if not parameter_names:
        raise ValueError(
            "BayMorph inference result must contain at least one parameter."
        )

    estimates: dict[str, float] = {}
    intervals: dict[
        str,
        tuple[float, float],
    ] = {}

    for name in parameter_names:
        summary = result.summary(
            name
        )

        estimates[name] = float(
            summary["median"]
        )
        intervals[name] = (
            float(summary["q16"]),
            float(summary["q84"]),
        )

    effective_sample_size = (
        result.effective_sample_size
    )

    fit_metadata: dict[str, Any] = {
        "adapter": "baymorph_inference_result",
        "effective_sample_size": (
            None
            if effective_sample_size is None
            else float(
                effective_sample_size
            )
        ),
    }

    backend = result.provenance.get(
        "backend"
    )
    if backend is not None:
        fit_metadata["backend"] = backend

    if metadata is not None:
        fit_metadata.update(
            metadata
        )

    likelihood_evaluations = None

    for key in (
        "likelihood_evaluations",
        "calls",
        "ncall",
        "n_calls",
    ):
        value = result.diagnostics.get(
            key
        )

        if value is not None:
            likelihood_evaluations = int(
                np.sum(value)
                if isinstance(
                    value,
                    (list, tuple, np.ndarray),
                )
                else value
            )
            break

    return RecoveryFit(
        estimates=estimates,
        intervals=intervals,
        log_evidence=result.logz,
        log_evidence_error=result.logz_error,
        likelihood_evaluations=likelihood_evaluations,
        metadata=fit_metadata,
    )


def recovery_fit_from_baymorph(
    fit_products: RealFitProducts,
    *,
    metadata: dict[str, Any] | None = None,
) -> RecoveryFit:
    """Convert real-fit products into a validation recovery result."""
    posterior = fit_products.posterior

    if posterior is None:
        raise ValueError(
            "BayMorph fit products must contain posterior results."
        )

    fit_metadata: dict[str, Any] = {
        "adapter": "baymorph",
        "model_name": fit_products.model_name,
        "chi2": float(
            fit_products.chi2
        ),
        "reduced_chi2": float(
            fit_products.reduced_chi2
        ),
        "ndof": int(
            fit_products.ndof
        ),
    }

    if metadata is not None:
        fit_metadata.update(
            metadata
        )

    return recovery_fit_from_inference_result(
        posterior,
        metadata=fit_metadata,
    )
