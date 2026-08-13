"""Compatibility alias for the legacy multi-band likelihood name."""

from baymorph.likelihood.multi_observation import (
    MultiObservationLikelihood,
)

MultiBandLikelihood = MultiObservationLikelihood

__all__ = [
    "MultiBandLikelihood",
]
