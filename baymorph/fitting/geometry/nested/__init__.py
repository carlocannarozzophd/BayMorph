"""Geometry-aware nested-sampling state and containers."""

from .config import NestedSamplerConfig
from .evidence import (
    EvidenceAccumulator,
    EvidenceUpdate,
    LogSpaceEvidenceAccumulator,
)
from .iteration import IterationPlan, IterationResult
from .live_points import LivePoint, LivePoints
from .run import NestedRun
from .sampler import GeometryNestedSampler
from .shrinkage import (
    ExpectedShrinkageModel,
    ShrinkageModel,
)
from .state import DeadPoint, NestedState
from .termination import (
    AnyTerminationCriterion,
    EvidenceTerminationCriterion,
    MaximumIterationTerminationCriterion,
    MaximumLikelihoodCallTerminationCriterion,
    TerminationCriterion,
    TerminationDecision,
)

__all__ = [
    "AnyTerminationCriterion",
    "DeadPoint",
    "EvidenceAccumulator",
    "EvidenceTerminationCriterion",
    "EvidenceUpdate",
    "ExpectedShrinkageModel",
    "GeometryNestedSampler",
    "IterationPlan",
    "IterationResult",
    "LivePoint",
    "LivePoints",
    "LogSpaceEvidenceAccumulator",
    "MaximumIterationTerminationCriterion",
    "MaximumLikelihoodCallTerminationCriterion",
    "NestedRun",
    "NestedSamplerConfig",
    "NestedState",
    "ShrinkageModel",
    "TerminationCriterion",
    "TerminationDecision",
]
