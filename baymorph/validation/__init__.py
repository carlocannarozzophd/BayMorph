"""Scientific validation infrastructure for BayMorph."""

from baymorph.validation.multi_observation_sbc import (
    MultiObservationSBCTruthDraw,
    PreparationFreeMultiObservationSBCCampaign,
    build_preparation_free_sbc_case,
    draw_multi_observation_sbc_truth,
    reference_multi_observation_sbc_parameter_config,
)
from baymorph.validation.calibration import (
    CalibrationCase,
    ParameterCalibration,
    calibration_from_inference_result,
    summarize_calibration,
)
from baymorph.validation.benchmark import (
    BenchmarkResult,
    ComparisonReport,
    benchmark_call,
    compare_results,
)
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
    weighted_credible_interval,
    weighted_posterior_rank,
)
from baymorph.validation.recovery import ParameterRecovery, RecoveryResult
from baymorph.validation.repeatability import (
    RepeatabilityRecord,
    RepeatabilitySummary,
    run_repeatability,
    summarize_repeatability,
)
from baymorph.validation.run import (
    ValidationRun,
    execute_validation_case,
    with_validation_seed,
)
from baymorph.validation.reference import (
    ReferenceValidationCase,
)
from baymorph.validation.report import (
    ValidationCheck,
    ValidationReport,
    ValidationTolerances,
    summarize_recovery,
    validate_result,
)
from baymorph.validation.simulation_grid import SimulationCase, SimulationGrid
from baymorph.validation.truth import GroundTruth
from baymorph.validation.baymorph_adapter import (
    recovery_fit_from_baymorph,
    recovery_fit_from_inference_result,
)
from baymorph.validation.simulation import build_synthetic_observation
from baymorph.validation.workspace import ValidationWorkspace

__all__ = [
    "PreparationFreeMultiObservationSBCCampaign",
    "reference_multi_observation_sbc_parameter_config",
    "draw_multi_observation_sbc_truth",
    "build_preparation_free_sbc_case",
    "MultiObservationSBCTruthDraw",
    "weighted_posterior_rank",
    "weighted_credible_interval",
    "summarize_calibration",
    "calibration_from_inference_result",
    "ParameterCalibration",
    "CalibrationCase",
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
    "summarize_repeatability",
    "run_repeatability",
    "RepeatabilitySummary",
    "RepeatabilityRecord",
    "ReferenceValidationCase",
    "SimulationCase",
    "SimulationGrid",
    "benchmark_call",
    "compare_results",
    "ComparisonReport",
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
    "ValidationRun",
    "execute_validation_case",
    "with_validation_seed",
    "validate_result",
    "ValidationTolerances",
    "ValidationReport",
    "ValidationCheck",
    "ValidationWorkspace",
]
