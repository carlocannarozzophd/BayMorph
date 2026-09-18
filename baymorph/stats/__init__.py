"""Statistical utilities for BayMorph."""

from baymorph.stats.robust import (
    robust_mad,
    robust_rms,
    robust_sigma,
)
from baymorph.stats.weighted import (
    posterior_summary,
    weighted_quantile,
)

__all__ = [
    "posterior_summary",
    "robust_mad",
    "robust_rms",
    "robust_sigma",
    "weighted_quantile",
]