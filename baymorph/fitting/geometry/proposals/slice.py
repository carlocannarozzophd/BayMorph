"""Whitened constrained slice proposals in prior-cube coordinates."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import norm

from baymorph.fitting.geometry.nested.live_points import LivePoints
from baymorph.fitting.parameters import ParameterSet
from baymorph.fitting.priors import (
    BetaPrior,
    LogUniformPrior,
    NormalPrior,
    UniformPrior,
)

from .base import ProposalEngine, ProposalFailure, ProposalResult

if TYPE_CHECKING:
    from baymorph.fitting.geometry.geometry_model import GeometryModel


class WhitenedSliceProposalEngine(ProposalEngine):
    """Generate constrained replacements with affine-whitened slice sampling.

    Sampling is performed in the unit prior cube so that the target measure is
    uniform before the hard nested-sampling likelihood constraint is applied.
    The current live-point covariance defines a fixed affine whitening scale
    for each replacement chain. One-dimensional stepping-out and shrinkage
    updates are then repeated along random directions in the whitened space.

    This engine is experimental. It does not yet implement clustering or
    periodic prior-cube wrapping, and it is not selected by the production
    sampler factory.

    Parameters
    ----------
    n_repeats : int or None, optional
        Number of one-dimensional slice updates per replacement. If omitted,
        three updates per sampled dimension are used.
    width : float, optional
        Initial one-dimensional bracket width in whitened coordinates.
    regularization : float, optional
        Relative diagonal regularization of the live-point covariance.
    absolute_regularization : float, optional
        Minimum absolute covariance regularization.
    max_steps_out : int, optional
        Maximum bracket expansions on each side of one slice.
    max_shrink : int, optional
        Maximum shrinkage candidates for one slice update.
    """

    identifier = "whitened_slice"

    def __init__(
        self,
        *,
        n_repeats: int | None = None,
        width: float = 1.0,
        regularization: float = 1.0e-8,
        absolute_regularization: float = 1.0e-12,
        max_steps_out: int = 100,
        max_shrink: int = 1_000,
    ) -> None:
        if n_repeats is not None:
            if isinstance(n_repeats, bool) or not isinstance(
                n_repeats,
                int,
            ):
                raise TypeError("n_repeats must be an integer or None.")
            if n_repeats <= 0:
                raise ValueError("n_repeats must be positive when provided.")

        width = float(width)
        regularization = float(regularization)
        absolute_regularization = float(absolute_regularization)

        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("width must be a positive finite scalar.")
        if not np.isfinite(regularization) or regularization < 0.0:
            raise ValueError(
                "regularization must be a non-negative finite scalar."
            )
        if (
            not np.isfinite(absolute_regularization)
            or absolute_regularization <= 0.0
        ):
            raise ValueError(
                "absolute_regularization must be a positive finite scalar."
            )

        self._validate_positive_integer(
            name="max_steps_out",
            value=max_steps_out,
        )
        self._validate_positive_integer(
            name="max_shrink",
            value=max_shrink,
        )

        self.n_repeats = n_repeats
        self.width = width
        self.regularization = regularization
        self.absolute_regularization = absolute_regularization
        self.max_steps_out = max_steps_out
        self.max_shrink = max_shrink

    def propose(
        self,
        *,
        live_points: LivePoints,
        logl_min: float,
        parameter_set: ParameterSet,
        likelihood: Callable[[np.ndarray], float],
        rng: np.random.Generator,
    ) -> ProposalResult:
        """Run one constrained slice chain and return its final point."""
        if not callable(likelihood):
            raise TypeError("likelihood must be callable.")

        threshold = float(logl_min)

        if np.isnan(threshold):
            raise ValueError("logl_min must not be NaN.")

        if parameter_set.ndim != live_points.ndim:
            raise ValueError(
                "ParameterSet dimensionality must match the live-point geometry."
            )

        if live_points.size < 2:
            raise ValueError(
                "Whitened slice proposals require at least two live points."
            )

        geometry = live_points.geometry
        geometry.validate_parameter_names(
            parameter_set.names
        )

        unit_live = np.stack(
            [
                self._physical_to_unit_cube(
                    parameter_set,
                    point.values,
                )
                for point in live_points
            ],
            axis=0,
        )

        cholesky, condition_number = self._whitening_cholesky(
            unit_live
        )

        valid_start_indices = np.flatnonzero(
            live_points.logl > threshold
        )

        if valid_start_indices.size == 0:
            raise ProposalFailure(
                "No live point satisfies the strict likelihood constraint."
            )

        start_index = int(
            rng.choice(valid_start_indices)
        )

        current_unit = unit_live[start_index].copy()
        current_physical = live_points[start_index].values.copy()
        current_logl = float(
            live_points[start_index].logl
        )

        n_repeats = (
            self.n_repeats
            if self.n_repeats is not None
            else max(
                1,
                3 * parameter_set.ndim,
            )
        )

        ncall = 0
        attempts = 0
        cube_rejections = 0
        step_out_attempts = 0
        shrink_attempts = 0

        for _ in range(n_repeats):
            direction = rng.normal(
                size=parameter_set.ndim
            )

            direction_norm = float(
                np.linalg.norm(direction)
            )

            if (
                not np.isfinite(direction_norm)
                or direction_norm == 0.0
            ):
                raise ProposalFailure(
                    "Failed to generate a finite non-zero slice direction."
                )

            direction /= direction_norm

            unit_direction = (
                cholesky
                @ direction
            )

            (
                current_unit,
                current_physical,
                current_logl,
                step_ncall,
                step_attempts,
                step_cube_rejections,
                step_step_out_attempts,
                step_shrink_attempts,
            ) = self._slice_step(
                current_unit=current_unit,
                current_logl=current_logl,
                unit_direction=unit_direction,
                logl_min=threshold,
                parameter_set=parameter_set,
                likelihood=likelihood,
                geometry=geometry,
                rng=rng,
            )

            ncall += step_ncall
            attempts += step_attempts
            cube_rejections += step_cube_rejections
            step_out_attempts += step_step_out_attempts
            shrink_attempts += step_shrink_attempts

        return ProposalResult(
            values=current_physical,
            logl=current_logl,
            ncall=ncall,
            attempts=attempts,
            engine=self.identifier,
            diagnostics={
                "n_repeats": float(n_repeats),
                "likelihood_calls_per_slice": (
                    ncall / n_repeats
                ),
                "attempts_per_slice": (
                    attempts / n_repeats
                ),
                "cube_rejection_fraction": (
                    cube_rejections / attempts
                    if attempts
                    else 0.0
                ),
                "step_out_attempts": float(
                    step_out_attempts
                ),
                "shrink_attempts": float(
                    shrink_attempts
                ),
                "whitening_condition_number": (
                    condition_number
                ),
            },
        )

    def _slice_step(
        self,
        *,
        current_unit: np.ndarray,
        current_logl: float,
        unit_direction: np.ndarray,
        logl_min: float,
        parameter_set: ParameterSet,
        likelihood: Callable[[np.ndarray], float],
        geometry: GeometryModel,
        rng: np.random.Generator,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
        int,
        int,
        int,
        int,
        int,
    ]:
        """Perform one stepping-out and shrinkage slice update."""
        if not current_logl > logl_min:
            raise ValueError(
                "Slice starting point must satisfy the strict constraint."
            )

        left = (
            -float(rng.random())
            * self.width
        )
        right = left + self.width

        ncall = 0
        attempts = 0
        cube_rejections = 0
        step_out_attempts = 0
        shrink_attempts = 0

        for _ in range(self.max_steps_out):
            (
                inside,
                _,
                _,
                call_increment,
                cube_rejected,
            ) = self._evaluate_offset(
                offset=left,
                current_unit=current_unit,
                unit_direction=unit_direction,
                logl_min=logl_min,
                parameter_set=parameter_set,
                likelihood=likelihood,
                geometry=geometry,
            )

            attempts += 1
            step_out_attempts += 1
            ncall += call_increment
            cube_rejections += int(
                cube_rejected
            )

            if not inside:
                break

            left -= self.width
        else:
            raise ProposalFailure(
                "Slice stepping-out failed to bracket the left boundary."
            )

        for _ in range(self.max_steps_out):
            (
                inside,
                _,
                _,
                call_increment,
                cube_rejected,
            ) = self._evaluate_offset(
                offset=right,
                current_unit=current_unit,
                unit_direction=unit_direction,
                logl_min=logl_min,
                parameter_set=parameter_set,
                likelihood=likelihood,
                geometry=geometry,
            )

            attempts += 1
            step_out_attempts += 1
            ncall += call_increment
            cube_rejections += int(
                cube_rejected
            )

            if not inside:
                break

            right += self.width
        else:
            raise ProposalFailure(
                "Slice stepping-out failed to bracket the right boundary."
            )

        for _ in range(self.max_shrink):
            offset = float(
                rng.uniform(
                    left,
                    right,
                )
            )

            (
                inside,
                physical,
                logl,
                call_increment,
                cube_rejected,
            ) = self._evaluate_offset(
                offset=offset,
                current_unit=current_unit,
                unit_direction=unit_direction,
                logl_min=logl_min,
                parameter_set=parameter_set,
                likelihood=likelihood,
                geometry=geometry,
            )

            attempts += 1
            shrink_attempts += 1
            ncall += call_increment
            cube_rejections += int(
                cube_rejected
            )

            if inside:
                return (
                    current_unit
                    + offset
                    * unit_direction,
                    physical,
                    logl,
                    ncall,
                    attempts,
                    cube_rejections,
                    step_out_attempts,
                    shrink_attempts,
                )

            if offset < 0.0:
                left = offset
            elif offset > 0.0:
                right = offset
            else:
                raise ProposalFailure(
                    "Slice shrinkage returned the unchanged starting point."
                )

        raise ProposalFailure(
            "Slice shrinkage failed to find a constrained point after "
            f"{self.max_shrink} candidates."
        )

    def _evaluate_offset(
        self,
        *,
        offset: float,
        current_unit: np.ndarray,
        unit_direction: np.ndarray,
        logl_min: float,
        parameter_set: ParameterSet,
        likelihood: Callable[[np.ndarray], float],
        geometry: GeometryModel,
    ) -> tuple[
        bool,
        np.ndarray,
        float,
        int,
        bool,
    ]:
        """Evaluate one scalar displacement along the current slice line."""
        candidate_unit = (
            current_unit
            + offset
            * unit_direction
        )

        if not self._in_unit_cube(
            candidate_unit
        ):
            return (
                False,
                np.empty(
                    parameter_set.ndim,
                    dtype=float,
                ),
                -np.inf,
                0,
                True,
            )

        physical = parameter_set.transform(
            candidate_unit
        )

        geometry.to_proposal_space(
            physical
        )

        logl = float(
            likelihood(
                physical
            )
        )

        if np.isnan(logl):
            raise ValueError(
                "Likelihood returned NaN during whitened slice proposal."
            )

        return (
            logl > logl_min,
            physical,
            logl,
            1,
            False,
        )

    def _whitening_cholesky(
        self,
        unit_values: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Return a regularized Cholesky factor of live-point covariance."""
        values = np.asarray(
            unit_values,
            dtype=float,
        )

        if values.ndim != 2:
            raise ValueError(
                "Unit live-point values must be two-dimensional."
            )

        if values.shape[0] < 2:
            raise ValueError(
                "At least two live points are required for whitening."
            )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "Unit live-point values must be finite."
            )

        covariance = np.atleast_2d(
            np.asarray(
                np.cov(
                    values,
                    rowvar=False,
                    ddof=1,
                ),
                dtype=float,
            )
        )

        covariance = 0.5 * (
            covariance
            + covariance.T
        )

        covariance = np.nan_to_num(
            covariance,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        diagonal = np.diag(
            covariance
        )

        positive = diagonal[
            np.isfinite(diagonal)
            & (diagonal > 0.0)
        ]

        scale = (
            float(np.max(positive))
            if positive.size
            else 1.0
        )

        jitter = max(
            self.absolute_regularization,
            self.regularization
            * scale,
        )

        covariance += (
            np.eye(
                values.shape[1],
                dtype=float,
            )
            * jitter
        )

        eigenvalues = np.linalg.eigvalsh(
            covariance
        )

        minimum = float(
            np.min(eigenvalues)
        )
        maximum = float(
            np.max(eigenvalues)
        )

        if (
            minimum <= 0.0
            or not np.isfinite(minimum)
            or not np.isfinite(maximum)
        ):
            raise ProposalFailure(
                "Regularized live-point covariance is not positive definite."
            )

        return (
            np.linalg.cholesky(
                covariance
            ),
            maximum / minimum,
        )

    @classmethod
    def _physical_to_unit_cube(
        cls,
        parameter_set: ParameterSet,
        physical_values: np.ndarray,
    ) -> np.ndarray:
        """Map one physical vector exactly back to supported prior coordinates."""
        values = np.asarray(
            physical_values,
            dtype=float,
        )

        if values.shape != (
            parameter_set.ndim,
        ):
            raise ValueError(
                "Physical vector must contain one value per sampled parameter."
            )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "Physical values must be finite."
            )

        unit = np.asarray(
            [
                cls._physical_to_unit(
                    parameter.prior,
                    value,
                )
                for parameter, value in zip(
                    parameter_set.parameters,
                    values,
                    strict=True,
                )
            ],
            dtype=float,
        )

        if not cls._in_unit_cube(
            unit
        ):
            raise ValueError(
                "Inverse prior transform produced values outside the unit cube."
            )

        return unit

    @staticmethod
    def _physical_to_unit(
        prior,
        physical_value: float,
    ) -> float:
        """Map one physical value through a supported built-in prior CDF."""
        value = float(
            physical_value
        )

        if not np.isfinite(value):
            raise ValueError(
                "Physical value must be finite."
            )

        if isinstance(
            prior,
            UniformPrior,
        ):
            WhitenedSliceProposalEngine._validate_support(
                value,
                prior.lower,
                prior.upper,
            )

            return float(
                (value - prior.lower)
                / (
                    prior.upper
                    - prior.lower
                )
            )

        if isinstance(
            prior,
            LogUniformPrior,
        ):
            WhitenedSliceProposalEngine._validate_support(
                value,
                prior.lower,
                prior.upper,
            )

            log_lower = np.log(
                prior.lower
            )
            log_upper = np.log(
                prior.upper
            )

            unit = (
                np.log(value)
                - log_lower
            ) / (
                log_upper
                - log_lower
            )

            return float(
                np.clip(
                    unit,
                    0.0,
                    1.0,
                )
            )

        if isinstance(
            prior,
            NormalPrior,
        ):
            unit = norm.cdf(
                (
                    value
                    - prior.mean
                )
                / prior.sigma
            )

            return float(
                np.clip(
                    unit,
                    0.0,
                    1.0,
                )
            )

        if isinstance(
            prior,
            BetaPrior,
        ):
            WhitenedSliceProposalEngine._validate_support(
                value,
                prior.lower,
                prior.upper,
            )

            scaled = (
                value
                - prior.lower
            ) / (
                prior.upper
                - prior.lower
            )

            unit = beta_distribution.cdf(
                scaled,
                a=prior.alpha,
                b=prior.beta,
            )

            return float(
                np.clip(
                    unit,
                    0.0,
                    1.0,
                )
            )

        raise TypeError(
            "Whitened native slice sampling does not yet support inverse "
            f"prior transforms for {type(prior).__name__}."
        )

    @staticmethod
    def _validate_support(
        value: float,
        lower: float,
        upper: float,
    ) -> None:
        """Validate one finite physical value against closed prior support."""
        if (
            value < lower
            or value > upper
        ):
            raise ValueError(
                "Physical value lies outside prior support."
            )

    @staticmethod
    def _in_unit_cube(
        values: np.ndarray,
    ) -> bool:
        """Return whether all coordinates lie inside the closed unit cube."""
        return bool(
            np.all(
                np.isfinite(values)
            )
            and np.all(
                values >= 0.0
            )
            and np.all(
                values <= 1.0
            )
        )

    @staticmethod
    def _validate_positive_integer(
        *,
        name: str,
        value: int,
    ) -> None:
        if isinstance(
            value,
            bool,
        ) or not isinstance(
            value,
            int,
        ):
            raise TypeError(
                f"{name} must be an integer."
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be positive."
            )
