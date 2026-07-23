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
from baymorph.results import (
    PosteriorResult,
    RealFitProducts,
)
from baymorph.runner import BayMorphRunner
from baymorph.simulation import (
    EmpiricalPSF,
    GaussianPSF,
    SyntheticObservation,
    SyntheticSersic,
)

__version__ = "0.2.0"

STABLE_API = [
    "BayMorphRunner",
    "Observation",
    "ObservationMetadata",
    "ImageData",
    "SersicModel",
    "DoubleSersicModel",
    "CompositeModel",
    "ConvolvedModel",
    "GaussianPSF",
    "EmpiricalPSF",
    "SyntheticObservation",
    "SyntheticSersic",
    "PosteriorResult",
    "RealFitProducts",
]

__all__ = STABLE_API