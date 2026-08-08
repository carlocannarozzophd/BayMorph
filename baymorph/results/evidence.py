"""Backend-neutral Bayesian-evidence representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EvidenceEstimate:
    """One estimate of the logarithmic Bayesian evidence.

    Parameters
    ----------
    logz : float
        Estimated logarithmic marginal likelihood.
    uncertainty : float, optional
        Reported uncertainty on ``logz``.
    method : str
        Stable identifier for the evidence-estimation method.
    uncertainty_kind : str, optional
        Description of the statistical meaning of ``uncertainty``.
    diagnostics : dict, optional
        Method-specific evidence diagnostics.
    provenance : dict, optional
        Information describing how the estimate was produced.

    Notes
    -----
    Evidence uncertainties produced by different algorithms need not have
    the same statistical interpretation. ``uncertainty_kind`` therefore
    belongs to the scientific result rather than being implicit in the
    backend implementation.
    """

    logz: float
    method: str
    uncertainty: float | None = None
    uncertainty_kind: str | None = None
    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )
    provenance: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Validate evidence metadata."""
        self.logz = float(
            self.logz
        )

        if np.isnan(
            self.logz
        ):
            raise ValueError(
                "logz cannot be NaN."
            )

        method = str(
            self.method
        ).strip()

        if not method:
            raise ValueError(
                "Evidence method cannot be empty."
            )

        self.method = method

        if self.uncertainty is not None:
            uncertainty = float(
                self.uncertainty
            )

            if (
                not np.isfinite(
                    uncertainty
                )
                or uncertainty < 0.0
            ):
                raise ValueError(
                    "Evidence uncertainty must be finite "
                    "and non-negative."
                )

            self.uncertainty = uncertainty

        if self.uncertainty_kind is not None:
            uncertainty_kind = str(
                self.uncertainty_kind
            ).strip()

            if not uncertainty_kind:
                raise ValueError(
                    "uncertainty_kind cannot be empty."
                )

            self.uncertainty_kind = uncertainty_kind

        self.diagnostics = dict(
            self.diagnostics
        )
        self.provenance = dict(
            self.provenance
        )
