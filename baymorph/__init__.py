"""Stable top-level public API for BayMorph."""

from baymorph.core import (
    ImageData,
    Observation,
    ObservationMetadata,
)
from baymorph.models import (
    CompositeModel,
    ConvolvedModel,
    DoubleSersicModel,
    SersicModel,
)
from baymorph.pipeline.advisor import (
    ObservationAdvisor,
    ObservationInspection,
    PriorGuidance,
)
from baymorph.pipeline.config_maker import (
    make_config_from_fits,
    make_multi_observation_config,
)
from baymorph.pipeline.multi_parameters import make_multi_parameters
from baymorph.results import (
    EvidenceEstimate,
    InferenceResult,
    PosteriorSamples,
    RealFitProducts,
)
from baymorph.runner import BayMorphRunner
from baymorph.simulation import (
    EmpiricalPSF,
    GaussianPSF,
    SyntheticObservation,
    SyntheticSersic,
    make_observation,
)

__version__ = "0.3.0"

__all__ = [
    "BayMorphRunner",
    "CompositeModel",
    "ConvolvedModel",
    "DoubleSersicModel",
    "EmpiricalPSF",
    "EvidenceEstimate",
    "GaussianPSF",
    "ImageData",
    "InferenceResult",
    "Observation",
    "ObservationAdvisor",
    "ObservationInspection",
    "ObservationMetadata",
    "PosteriorSamples",
    "PriorGuidance",
    "RealFitProducts",
    "SersicModel",
    "SyntheticObservation",
    "SyntheticSersic",
    "make_config_from_fits",
    "make_observation",
    "make_multi_observation_config",
    "make_multi_parameters",
]
