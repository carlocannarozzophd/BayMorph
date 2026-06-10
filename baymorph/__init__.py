"""BayMorph: Bayesian morphological modelling of astronomical images."""

from .core import (
    BackgroundEstimate,
    ImageData,
    estimate_background_sigma,
    estimate_noise_map,
    make_constant_noise_map,
    subtract_background,
)
from .fitting import GridSearchResult, grid_search_1d, DynestyFluxFitter, DynestyFluxResult
from .likelihood import gaussian_log_likelihood
from .models import SersicModel
from .simulation import (
    GaussianPSF,
    SyntheticObservation,
    SyntheticSersic,
    add_gaussian_noise,
    add_poisson_noise,
    convolve_with_psf,
    make_observation,
)

__version__ = "0.0.1"

__all__ = [
    "ImageData",
    "BackgroundEstimate",
    "estimate_background_sigma",
    "subtract_background",
    "make_constant_noise_map",
    "estimate_noise_map",
    "GridSearchResult",
    "grid_search_1d",
    "gaussian_log_likelihood",
    "SersicModel",
    "SyntheticSersic",
    "SyntheticObservation",
    "GaussianPSF",
    "convolve_with_psf",
    "make_observation",
    "add_gaussian_noise",
    "add_poisson_noise",
    "DynestyFluxFitter",
    "DynestyFluxResult",
]