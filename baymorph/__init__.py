"""BayMorph: Bayesian morphological modelling of astronomical images."""

from .core import (
    BackgroundEstimate,
    ImageData,
    estimate_background_sigma,
    estimate_noise_map,
    make_constant_noise_map,
    subtract_background,
)
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
    "SyntheticSersic",
    "SyntheticObservation",
    "GaussianPSF",
    "convolve_with_psf",
    "make_observation",
    "add_gaussian_noise",
    "add_poisson_noise",
]