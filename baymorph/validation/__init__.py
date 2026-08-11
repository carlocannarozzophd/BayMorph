"""Scientific validation infrastructure for BayMorph."""

from baymorph.validation.benchmark import BenchmarkResult, benchmark_call
from baymorph.validation.experiment import (
    RecoveryExperiment,
    RecoveryFit,
)
from baymorph.validation.multi_observation_campaign import MultiObservationNoiseCampaign
from baymorph.validation.multi_observation_preparation_free import (
    PreparationFreeMultiObservationCase,
    PreparationFreeMultiObservationNoiseCampaign,
    build_preparation_free_multi_observation_case,
)
from baymorph.validation.multi_observation_workspace import MultiObservationValidationWorkspace
from baymorph.validation.multi_observation_truth import (
    MultiObservationGroundTruth,
    ObservationTruth,
)
from baymorph.validation.metrics import (
    bias,
    coverage_fraction,
    fractional_error,
    interval_contains_truth,
    parameter_bias,
    scatter,
)
from baymorph.validation.recovery import ParameterRecovery, RecoveryResult
from baymorph.validation.report import summarize_recovery
from baymorph.validation.simulation_grid import SimulationCase, SimulationGrid
from baymorph.validation.truth import GroundTruth
from baymorph.validation.baymorph_adapter import (
    recovery_fit_from_baymorph,
    recovery_fit_from_inference_result,
)
from baymorph.validation.simulation import build_synthetic_observation
from baymorph.validation.workspace import ValidationWorkspace

__all__ = [
    "MultiObservationNoiseCampaign",
    "PreparationFreeMultiObservationCase",
    "PreparationFreeMultiObservationNoiseCampaign",
    "MultiObservationValidationWorkspace",
    "MultiObservationGroundTruth",
    "ObservationTruth",
    "BenchmarkResult",
    "GroundTruth",
    "ParameterRecovery",
    "RecoveryExperiment",
    "RecoveryFit",
    "RecoveryResult",
    "SimulationCase",
    "SimulationGrid",
    "benchmark_call",
    "bias",
    "build_preparation_free_multi_observation_case",
    "coverage_fraction",
    "fractional_error",
    "interval_contains_truth",
    "parameter_bias",
    "scatter",
    "summarize_recovery",
    "recovery_fit_from_baymorph",
    "recovery_fit_from_inference_result",
    "build_synthetic_observation",
    "ValidationWorkspace",
]
