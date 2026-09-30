"""Statistical utilities for BayMorph."""

from baymorph.stats.robust import (
    robust_sigma,
    truncated_gaussian_sigma,
    truncated_variance_ratio,
)
from baymorph.stats.weighted import (
    mahalanobis_distance_squared,
    posterior_summary,
    weighted_covariance,
    weighted_quantile,
)

__all__ = [
    "mahalanobis_distance_squared",
    "posterior_summary",
    "robust_sigma",
    "truncated_gaussian_sigma",
    "truncated_variance_ratio",
    "weighted_covariance",
    "weighted_quantile",
]