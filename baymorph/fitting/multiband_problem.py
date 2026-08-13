"""Compatibility alias for the legacy multi-band inference-problem name."""

from baymorph.fitting.multi_observation_problem import (
    MultiObservationInferenceProblem,
)

MultiBandInferenceProblem = MultiObservationInferenceProblem

__all__ = [
    "MultiBandInferenceProblem",
]
