"""Stable top-level public API for BayMorph."""

from baymorph.analysis.model_screening import (
    ModelScreeningReport,
    ModelScreeningScore,
)
from baymorph.core import (
    ImageData,
    Observation,
    ObservationMetadata,
)
from baymorph.models import (
    CompositeModel,
    DoubleSersicModel,
    SersicModel,
)
from baymorph.observation import ConvolvedModel
from baymorph.pipeline.advisor import (
    ObservationAdvisor,
    ObservationConfigProposal,
    ObservationInspection,
    PriorGuidance,
)
from baymorph.pipeline.config_maker import (
    make_config_from_fits,
    make_multi_observation_config,
)
from baymorph.pipeline.model_screening_advisor import (
    SingleObservationModelAdvisor,
    SingleObservationScreeningResult,
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
from baymorph.analysis.model_screening_persistence import (
    ModelScreeningClassifierArtifact,
    load_model_screening_classifier,
)

__version__ = "0.5.0"

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
    "ModelScreeningClassifierArtifact",
    "ModelScreeningReport",
    "ModelScreeningScore",
    "Observation",
    "ObservationAdvisor",
    "ObservationConfigProposal",
    "ObservationInspection",
    "ObservationMetadata",
    "PosteriorSamples",
    "PriorGuidance",
    "RealFitProducts",
    "SersicModel",
    "SingleObservationModelAdvisor",
    "SingleObservationScreeningResult",
    "SyntheticObservation",
    "SyntheticSersic",
    "load_model_screening_classifier",
    "make_config_from_fits",
    "make_multi_observation_config",
    "make_multi_parameters",
    "make_observation",
]
