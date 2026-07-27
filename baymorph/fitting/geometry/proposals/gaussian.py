"""Gaussian constrained proposal engine in transformed parameter space."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from baymorph.fitting.geometry.nested.live_points import LivePoints
from baymorph.fitting.parameters import ParameterSet

from .base import ProposalEngine, ProposalFailure, ProposalResult


class GaussianProposalEngine(ProposalEngine):
    """Generate constrained proposals from a live-point Gaussian approximation.

    The proposal distribution is built in proposal space from the current live
    points. Candidate vectors are transformed back to physical coordinates
    before prior-bound checks and likelihood evaluation.

    Parameters
    ----------
    covariance_scale : float, optional
        Multiplicative scale applied to the regularized live-point covariance.
    regularization : float, optional
        Relative diagonal covariance regularization.
    absolute_regularization : float, optional
        Minimum diagonal covariance regularization.
    max_attempts : int, optional
        Maximum number of generated candidates for one accepted proposal.
    """

    identifier = "gaussian"

    def __init__(
        self,
        *,
        covariance_scale: float = 4.0,
        regularization: float = 1.0e-6,
        absolute_regularization: float = 1.0e-12,
        max_attempts: int = 100_000,
    ) -> None:
        covariance_scale = float(covariance_scale)
        regularization = float(regularization)
        absolute_regularization = float(absolute_regularization)

        if not np.isfinite(covariance_scale) or covariance_scale <= 0.0:
            raise ValueError("covariance_scale must be a positive finite scalar.")
        if not np.isfinite(regularization) or regularization < 0.0:
            raise ValueError("regularization must be a non-negative finite scalar.")
        if (
            not np.isfinite(absolute_regularization)
            or absolute_regularization <= 0.0
        ):
            raise ValueError(
                "absolute_regularization must be a positive finite scalar."
            )
        if int(max_attempts) != max_attempts or int(max_attempts) <= 0:
            raise ValueError("max_attempts must be a positive integer.")

        self.covariance_scale = covariance_scale
        self.regularization = regularization
        self.absolute_regularization = absolute_regularization
        self.max_attempts = int(max_attempts)

    def propose(
        self,
        *,
        live_points: LivePoints,
        logl_min: float,
        parameter_set: ParameterSet,
        likelihood: Callable[[np.ndarray], float],
        rng: np.random.Generator,
    ) -> ProposalResult:
        """Draw Gaussian candidates until the strict constraint is satisfied."""
        if not callable(likelihood):
            raise TypeError("likelihood must be callable.")

        threshold = float(logl_min)
        if np.isnan(threshold):
            raise ValueError("logl_min must not be NaN.")
        if parameter_set.ndim != live_points.ndim:
            raise ValueError(
                "ParameterSet dimensionality must match the live-point geometry."
            )

        geometry = live_points.geometry
        geometry.validate_parameter_names(parameter_set.names)

        proposal_values = np.asarray(
            live_points.proposal_values,
            dtype=float,
        )
        if proposal_values.ndim != 2:
            raise ValueError("Live-point proposal values must be two-dimensional.")
        if proposal_values.shape[1] != parameter_set.ndim:
            raise ValueError(
                "Live-point proposal dimensionality must match ParameterSet."
            )
        if not np.all(np.isfinite(proposal_values)):
            raise ValueError("Live-point proposal values must be finite.")

        mean = np.mean(proposal_values, axis=0)
        covariance = self._regularized_covariance(proposal_values)

        ncall = 0
        bounds_rejections = 0

        for attempt in range(1, self.max_attempts + 1):
            proposal_candidate = rng.multivariate_normal(
                mean=mean,
                cov=covariance,
                check_valid="raise",
            )
            physical_candidate = geometry.from_proposal_space(
                proposal_candidate
            )

            if not self._within_bounds(
                values=physical_candidate,
                parameter_set=parameter_set,
            ):
                bounds_rejections += 1
                continue

            logl = float(likelihood(physical_candidate))
            ncall += 1

            if np.isnan(logl):
                raise ValueError(
                    "Likelihood returned NaN during Gaussian proposal."
                )

            if logl > threshold:
                return ProposalResult(
                    values=physical_candidate,
                    logl=logl,
                    ncall=ncall,
                    attempts=attempt,
                    engine=self.identifier,
                    diagnostics={
                        "acceptance_fraction": 1.0 / attempt,
                        "likelihood_fraction": ncall / attempt,
                        "bounds_rejection_fraction": (
                            bounds_rejections / attempt
                        ),
                        "covariance_scale": self.covariance_scale,
                    },
                )

        raise ProposalFailure(
            "GaussianProposalEngine failed to satisfy the likelihood "
            f"constraint after {self.max_attempts} attempts and "
            f"{ncall} likelihood evaluations."
        )

    def _regularized_covariance(
        self,
        proposal_values: np.ndarray,
    ) -> np.ndarray:
        """Return a finite positive-definite covariance matrix."""
        ndim = proposal_values.shape[1]

        if proposal_values.shape[0] <= 1:
            covariance = np.zeros((ndim, ndim), dtype=float)
        else:
            covariance = np.asarray(
                np.cov(proposal_values, rowvar=False),
                dtype=float,
            )

        if covariance.ndim == 0:
            covariance = covariance.reshape(1, 1)

        covariance = 0.5 * (covariance + covariance.T)

        diagonal = np.diag(covariance)
        finite_positive_diagonal = diagonal[
            np.isfinite(diagonal) & (diagonal > 0.0)
        ]

        if finite_positive_diagonal.size:
            scale = float(np.max(finite_positive_diagonal))
        else:
            scale = 1.0

        jitter = max(
            self.absolute_regularization,
            self.regularization * scale,
        )

        covariance = np.nan_to_num(
            covariance,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        covariance += np.eye(ndim, dtype=float) * jitter
        covariance *= self.covariance_scale

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(
            eigenvalues,
            self.absolute_regularization,
        )

        return (eigenvectors * eigenvalues) @ eigenvectors.T

    @staticmethod
    def _within_bounds(
        *,
        values: np.ndarray,
        parameter_set: ParameterSet,
    ) -> bool:
        """Return whether a physical candidate lies inside every prior bound."""
        values = np.asarray(values, dtype=float)

        if values.shape != (parameter_set.ndim,):
            raise ValueError(
                "Physical proposal candidate has incompatible dimensionality."
            )
        if not np.all(np.isfinite(values)):
            return False

        return all(
            parameter.lower <= value <= parameter.upper
            for value, parameter in zip(
                values,
                parameter_set.parameters,
                strict=True,
            )
        )
