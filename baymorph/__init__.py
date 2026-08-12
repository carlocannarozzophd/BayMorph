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
from baymorph.pipeline.config_maker import (
    make_config_from_fits,
)
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
    "ObservationMetadata",
    "PosteriorSamples",
    "RealFitProducts",
    "SersicModel",
    "SyntheticObservation",
    "SyntheticSersic",
    "make_config_from_fits",
]
