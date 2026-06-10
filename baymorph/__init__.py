"""BayMorph: Bayesian morphological modelling of astronomical images."""

from .simulation import SyntheticSersic, add_gaussian_noise, add_poisson_noise

__version__ = "0.0.1"

__all__ = [
    "SyntheticSersic",
    "add_gaussian_noise",
    "add_poisson_noise",
]