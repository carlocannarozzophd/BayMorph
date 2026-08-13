"""Compatibility aliases for legacy multi-band parameter-routing names."""

from baymorph.fitting.multi_observation import (
    CouplingMode,
    MultiObservationParameterLayout,
    MultiObservationParameterSpec,
)

MultiBandParameterLayout = MultiObservationParameterLayout
MultiBandParameterSpec = MultiObservationParameterSpec

__all__ = [
    "CouplingMode",
    "MultiBandParameterLayout",
    "MultiBandParameterSpec",
]
