"""BayMorph: Bayesian morphological modelling of astronomical images."""

from .core import (
    BackgroundEstimate,
    ImageData,
    estimate_background_sigma,
    estimate_background_photutils,  
    estimate_noise_map,
    make_constant_noise_map,
    subtract_background,
    detect_sources_mask,
)
from .fitting import (
    GridSearchResult, grid_search_1d,
    DynestyFluxFitter, DynestyFluxResult,
    DynestyFluxReFitter, DynestyFluxReResult,
    DynestyFluxReNFitter, DynestyFluxReNResult,
    DynestyFitter,DynestyResult,
    Parameter, ParameterSet
)
from .likelihood import gaussian_log_likelihood
from .models import (
    SersicModel,
    ConvolvedModel,
    CompositeModel,
    DoubleSersicModel,
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
from .plotting import (
    plot_data_model_residual,
    plot_1d_posterior, plot_2d_samples,
    corner_plot,
    erf_sigma_to_percentiles,
    finalize_figure,
    make_plot,
    set_plot_style,
    style_axes,
    show_log_image,
    plot_log_data_model_residual,
)

from .results import PosteriorResult

from .pipeline import (
    RealFITSPreparation,
    prepare_real_fits_image,
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
    "ConvolvedModel",
    "CompositeModel",
    "DoubleSersicModel",
    "SyntheticSersic",
    "SyntheticObservation",
    "GaussianPSF",
    "convolve_with_psf",
    "make_observation",
    "add_gaussian_noise",
    "add_poisson_noise",
    "DynestyFluxFitter",
    "DynestyFluxResult",
    "plot_1d_posterior",
    "plot_2d_samples",
    "set_plot_style",
    "style_axes",
    "make_plot",
    "finalize_figure",
    "corner_plot",
    "erf_sigma_to_percentiles",
    "plot_data_model_residual",
    "DynestyFluxReNFitter",
    "DynestyFluxReNResult",
    "PosteriorResult",
    "show_log_image",
    "plot_log_data_model_residual",
    "DynestyFitter",
    "DynestyResult",
    "estimate_background_photutils",
    "detect_sources_mask",
    "Parameter",
    "ParameterSet",
    "RealFITSPreparation",
    "prepare_real_fits_image",
]