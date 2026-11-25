"""Pareto-smoothed importance sampling for posterior reweighting.

The weights of draws from an approximate density are stabilized by
replacing their largest values with the expected order statistics of a
generalized Pareto distribution fitted to the upper tail. The fitted shape
``k`` diagnoses the reliability of the reweighted estimates: below 0.5 the
weights have a finite variance, from 0.5 to 0.7 the estimates are usable
with enough draws, above 0.7 they are unreliable (Vehtari et al., Pareto
smoothed importance sampling). The shape is estimated by the method of
Zhang and Stephens (2009), with the weakly informative prior on ``k`` of
Vehtari et al.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

FloatArray = NDArray[np.float64]

# Prior of Zhang and Stephens on the inverse scale, and the weakly
# informative prior of Vehtari et al. on the shape: ten pseudo-observations
# at 0.5.
_INVERSE_SCALE_PRIOR = 3.0
_SHAPE_PRIOR_COUNT = 10.0
_SHAPE_PRIOR_VALUE = 0.5

# Fewer tail draws than this cannot support a Pareto fit.
_MINIMUM_TAIL = 5


@dataclass(frozen=True)
class SmoothedWeights:
    """Pareto-smoothed, self-normalized importance weights.

    Parameters
    ----------
    log_weights : ndarray
        Logarithms of the smoothed weights, normalized to unit sum. A draw
        outside the support of the target keeps ``-inf``.
    pareto_k : float
        Estimated shape of the generalized Pareto tail. ``0.0`` when the
        weights are constant within rounding, which bounds them.
    effective_sample_size : float
        ``1 / sum(w**2)`` of the smoothed normalized weights.
    tail_length : int
        Number of largest weights the tail fit used.
    """

    log_weights: FloatArray
    pareto_k: float
    effective_sample_size: float
    tail_length: int

    @property
    def weights(self) -> FloatArray:
        """Return the smoothed weights, normalized to unit sum."""
        return np.exp(self.log_weights)


def generalized_pareto_fit(exceedances: FloatArray) -> tuple[float, float]:
    """Fit a generalized Pareto distribution to positive exceedances.

    Parameters
    ----------
    exceedances : ndarray
        Positive values above a threshold, sorted in increasing order.

    Returns
    -------
    tuple of float
        The shape ``k`` and the scale ``sigma`` of the distribution
        ``F(x) = 1 - (1 + k x / sigma) ** (-1 / k)``.

    Notes
    -----
    The estimator of Zhang and Stephens (2009) averages the profile
    likelihood over a grid of the inverse scale ``k / sigma``. The shape is
    then shrunk towards 0.5 by ten pseudo-observations, as in Vehtari et al.
    """
    values = np.asarray(exceedances, dtype=float)

    if values.ndim != 1 or values.size < _MINIMUM_TAIL:
        raise ValueError(f"A Pareto fit needs at least {_MINIMUM_TAIL} exceedances.")

    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Exceedances must be finite and positive.")

    if np.any(np.diff(values) < 0.0):
        raise ValueError("Exceedances must be sorted in increasing order.")

    count = values.size
    grid_size = 30 + int(np.sqrt(count))
    quartile = values[int(count / 4.0 + 0.5) - 1]

    # The grid holds minus the inverse scale, as in Zhang and Stephens.
    grid = 1.0 - np.sqrt(grid_size / (np.arange(1, grid_size + 1) - 0.5))
    grid /= _INVERSE_SCALE_PRIOR * quartile
    grid += 1.0 / values[-1]

    shapes = np.mean(np.log1p(-grid[:, None] * values[None, :]), axis=1)
    profile = count * (np.log(-grid / shapes) - shapes - 1.0)

    # A grid point far below the best overflows to a weight of exactly zero.
    with np.errstate(over="ignore"):
        grid_weights = 1.0 / np.sum(np.exp(profile[None, :] - profile[:, None]), axis=1)

    kept = grid_weights >= 10.0 * np.finfo(float).eps
    grid = grid[kept]
    grid_weights = grid_weights[kept] / np.sum(grid_weights[kept])

    inverse_scale = float(np.sum(grid * grid_weights))
    shape = float(np.mean(np.log1p(-inverse_scale * values)))
    scale = -shape / inverse_scale
    shape = (count * shape + _SHAPE_PRIOR_COUNT * _SHAPE_PRIOR_VALUE) / (
        count + _SHAPE_PRIOR_COUNT
    )

    return shape, scale


def _generalized_pareto_quantile(
    probabilities: FloatArray,
    shape: float,
    scale: float,
) -> FloatArray:
    """Return quantiles of a generalized Pareto distribution."""
    if abs(shape) < np.finfo(float).eps:
        return -scale * np.log1p(-probabilities)

    return scale * np.expm1(-shape * np.log1p(-probabilities)) / shape


def pareto_smoothed_log_weights(log_weights: FloatArray) -> SmoothedWeights:
    """Smooth importance weights and return their reliability diagnostics.

    Parameters
    ----------
    log_weights : ndarray
        Logarithms of the raw importance weights, target over proposal, up to
        a constant. ``-inf`` marks a draw outside the support of the target
        and counts as a draw of zero weight. NaN and ``+inf`` are invalid.

    Returns
    -------
    SmoothedWeights
        The smoothed normalized log weights, the Pareto shape, the effective
        sample size and the tail length.

    Raises
    ------
    ValueError
        If the array is not one-dimensional with at least two entries, holds
        NaN or ``+inf``, or every weight is zero.
    """
    values = np.array(log_weights, dtype=float)

    if values.ndim != 1 or values.size < 2:
        raise ValueError("log_weights must be one-dimensional with two entries.")

    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("log_weights must not hold NaN or +inf.")

    finite = np.isfinite(values)

    if not np.any(finite):
        raise ValueError("Every importance weight is zero.")

    values -= np.max(values[finite])
    count = values.size
    tail_length = int(np.ceil(min(0.2 * count, 3.0 * np.sqrt(count))))
    spread = float(np.max(values[finite]) - np.min(values[finite]))

    if spread <= 1e-12:
        # Constant weights are bounded: nothing to smooth.
        pareto_k = 0.0
    else:
        order = np.argsort(values)
        cutoff = max(
            float(values[order[count - tail_length - 1]]),
            float(np.log(np.finfo(float).tiny)),
        )
        tail = np.flatnonzero(values > cutoff)

        if tail.size < _MINIMUM_TAIL:
            pareto_k = float("inf")
        else:
            tail = tail[np.argsort(values[tail])]
            exceedances = np.exp(values[tail]) - np.exp(cutoff)
            positive = exceedances > 0.0

            if np.count_nonzero(positive) < _MINIMUM_TAIL:
                pareto_k = float("inf")
            else:
                pareto_k, scale = generalized_pareto_fit(exceedances[positive])
                probabilities = (np.arange(tail.size) + 0.5) / tail.size
                smoothed = _generalized_pareto_quantile(
                    probabilities,
                    pareto_k,
                    scale,
                )
                values[tail] = np.log(smoothed + np.exp(cutoff))
                # The smoothed tail never exceeds the largest raw weight.
                values[values > 0.0] = 0.0

    finite = np.isfinite(values)
    values[finite] -= logsumexp(values[finite])
    weights = np.exp(values)

    return SmoothedWeights(
        log_weights=values,
        pareto_k=float(pareto_k),
        effective_sample_size=float(1.0 / np.sum(weights**2)),
        tail_length=tail_length,
    )


__all__ = [
    "SmoothedWeights",
    "generalized_pareto_fit",
    "pareto_smoothed_log_weights",
]
