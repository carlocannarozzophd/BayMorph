"""Compatibility aliases for legacy multi-band fitting APIs."""

from baymorph.pipeline.multi_observation_fit import (
    _MultiObservationBuildContext,
    _build_multi_observation_context,
    _build_multi_observation_problem,
    _load_multi_observation_config,
    _run_multi_observation_inference,
    build_multi_observation_problem_from_config,
    run_multi_observation_fit_from_config,
    run_multi_observation_inference_from_config,
)

MultiBandBuildContext = _MultiObservationBuildContext
_MultiBandBuildContext = _MultiObservationBuildContext

build_multiband_problem_from_config = (
    build_multi_observation_problem_from_config
)

run_multiband_fit_from_config = (
    run_multi_observation_inference_from_config
)

_build_multiband_context = (
    _build_multi_observation_context
)

_build_multiband_problem = (
    _build_multi_observation_problem
)

_load_multiband_config = (
    _load_multi_observation_config
)

_run_multiband_inference = (
    _run_multi_observation_inference
)

__all__ = [
    "build_multiband_problem_from_config",
    "run_multiband_fit_from_config",
    "run_multi_observation_fit_from_config",
]
