"""Adapters between BayMorph fitting products and validation results."""

from __future__ import annotations

from typing import Any

import numpy as np

from baymorph.results import RealFitProducts
from baymorph.validation.experiment import RecoveryFit


def recovery_fit_from_baymorph(
    fit_products: RealFitProducts,
    *,
    metadata: dict[str, Any] | None = None,
) -> RecoveryFit:
    """Convert BayMorph fitting products into a validation recovery result.

    Posterior medians are used as point estimates, while the 16th and 84th
    percentiles define the default posterior intervals.

    Parameters
    ----------
    fit_products : RealFitProducts
        Products returned by a BayMorph Bayesian fit.
    metadata : dict, optional
        Additional metadata attached to the normalized recovery result.

    Returns
    -------
    RecoveryFit
        Validation-compatible fitting result.

    Raises
    ------
    ValueError
        If posterior results or evidence values are unavailable.
    """
    posterior = fit_products.posterior

    if posterior is None:
        raise ValueError(
            "BayMorph fit products must contain posterior results."
        )

    parameter_names = tuple(posterior.parameter_names)

    if not parameter_names:
        raise ValueError(
            "BayMorph posterior must contain at least one parameter."
        )

    estimates: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}

    for name in parameter_names:
        summary = posterior.summary(name)

        estimates[name] = float(summary["median"])
        intervals[name] = (
            float(summary["q16"]),
            float(summary["q84"]),
        )

    logz = np.asarray(posterior.logz, dtype=float)

    if logz.size == 0:
        raise ValueError(
            "BayMorph posterior must contain Bayesian-evidence values."
        )

    log_evidence = float(logz[-1])

    log_evidence_error = None
    if posterior.logzerr is not None:
        logzerr = np.asarray(posterior.logzerr, dtype=float)

        if logzerr.size == 0:
            raise ValueError(
                "BayMorph posterior evidence errors cannot be empty."
            )

        log_evidence_error = float(logzerr[-1])

    fit_metadata = {
        "adapter": "baymorph",
        "model_name": fit_products.model_name,
        "chi2": float(fit_products.chi2),
        "reduced_chi2": float(fit_products.reduced_chi2),
        "ndof": int(fit_products.ndof),
    }

    if metadata is not None:
        fit_metadata.update(metadata)

    return RecoveryFit(
        estimates=estimates,
        intervals=intervals,
        log_evidence=log_evidence,
        log_evidence_error=log_evidence_error,
        likelihood_evaluations=None,
        metadata=fit_metadata,
    )