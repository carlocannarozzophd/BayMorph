"""Scientific validation infrastructure for BayMorph."""

from baymorph.validation.benchmark import BenchmarkResult, benchmark_call
from baymorph.validation.experiment import (
    RecoveryExperiment,
    RecoveryFit,
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
from baymorph.validation.baymorph_adapter import recovery_fit_from_baymorph
from baymorph.validation.simulation import build_synthetic_observation

__all__ = [
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
    "coverage_fraction",
    "fractional_error",
    "interval_contains_truth",
    "parameter_bias",
    "scatter",
    "summarize_recovery",
    "recovery_fit_from_baymorph",
    "build_synthetic_observation",
]