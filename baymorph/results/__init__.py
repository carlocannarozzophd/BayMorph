"""Result containers for BayMorph."""

from .fit import RealFitProducts
from .posterior import PosteriorResult

__all__ = [
    "PosteriorResult",
    "RealFitProducts",
]