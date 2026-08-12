"""Result containers for BayMorph."""

from baymorph.results.evidence import EvidenceEstimate
from baymorph.results.fit import RealFitProducts
from baymorph.results.inference import InferenceResult
from baymorph.results.multi_observation import (
    MultiObservationFitProducts,
    ObservationFitProducts,
)
from baymorph.results.posterior import PosteriorResult
from baymorph.results.samples import PosteriorSamples


__all__ = [
    "EvidenceEstimate",
    "InferenceResult",
    "MultiObservationFitProducts",
    "ObservationFitProducts",
    "PosteriorResult",
    "PosteriorSamples",
    "RealFitProducts",
]
