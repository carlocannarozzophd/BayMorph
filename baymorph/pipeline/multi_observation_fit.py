"""Canonical production workflow for joint multi-observation inference."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from baymorph.fitting.problem import InferenceProblem
from baymorph.pipeline.multi_observation_config import (
    resolve_multi_observation_observation_configs,
)
from baymorph.pipeline.multi_observation_noise import (
    estimate_cross_observation_noise_diagnostics,
    print_cross_observation_noise_diagnostics,
)
from baymorph.pipeline.multi_observation_persistence import (
    save_multi_observation_fit_products,
)
from baymorph.pipeline.multiband_fit import (
    _build_multi_observation_fit_products,
    _build_multiband_context,
    _build_multiband_problem,
    _load_multiband_config,
    _run_multiband_inference,
)
from baymorph.results import (
    InferenceResult,
    MultiObservationFitProducts,
)


def build_multi_observation_problem_from_config(
    config: str | Path | Mapping[str, Any],
) -> InferenceProblem:
    """Build the backend-neutral joint multi-observation problem."""
    raw_config, base_dir = _load_multiband_config(
        config
    )

    return _build_multiband_problem(
        raw_config,
        base_dir=base_dir,
    )


def run_multi_observation_inference_from_config(
    config: str | Path | Mapping[str, Any],
) -> InferenceResult:
    """Run joint inference with pre-fit noise diagnostics."""
    raw_config, base_dir = _load_multiband_config(
        config
    )

    fit_config = raw_config[
        "fit"
    ]
    model_name = fit_config[
        "models"
    ][0]

    context = _build_multiband_context(
        raw_config,
        base_dir=base_dir,
    )

    observation_configs = (
        resolve_multi_observation_observation_configs(
            raw_config,
            base_dir=base_dir,
        )
    )

    noise_diagnostics = (
        _estimate_pre_fit_noise_diagnostics(
            context=context,
            observation_configs=observation_configs,
        )
    )

    result = _run_multiband_inference(
        fit_config=fit_config,
        model_name=model_name,
        problem=context.problem,
    )

    result.diagnostics[
        "pre_fit_cross_observation_noise"
    ] = noise_diagnostics

    return result


def run_multi_observation_fit_from_config(
    config: str | Path | Mapping[str, Any],
) -> MultiObservationFitProducts:
    """Run joint inference and build multi-observation fit products."""
    raw_config, base_dir = _load_multiband_config(
        config
    )

    fit_config = raw_config[
        "fit"
    ]
    model_name = fit_config[
        "models"
    ][0]

    context = _build_multiband_context(
        raw_config,
        base_dir=base_dir,
    )

    observation_configs = (
        resolve_multi_observation_observation_configs(
            raw_config,
            base_dir=base_dir,
        )
    )

    noise_diagnostics = (
        _estimate_pre_fit_noise_diagnostics(
            context=context,
            observation_configs=observation_configs,
        )
    )

    inference_result = _run_multiband_inference(
        fit_config=fit_config,
        model_name=model_name,
        problem=context.problem,
    )

    inference_result.diagnostics[
        "pre_fit_cross_observation_noise"
    ] = noise_diagnostics

    products = _build_multi_observation_fit_products(
        context=context,
        inference_result=inference_result,
        model_name=model_name,
    )

    if "outputs" in raw_config:
        save_multi_observation_fit_products(
            raw_config,
            products,
        )

    return products


def _estimate_pre_fit_noise_diagnostics(
    *,
    context: Any,
    observation_configs: Mapping[
        str,
        Mapping[str, Any],
    ],
) -> dict[str, Any]:
    """Estimate blank-sky noise coupling before sampling."""
    diagnostics = (
        estimate_cross_observation_noise_diagnostics(
            image_products=context.image_products,
            observation_configs=observation_configs,
        )
    )

    print_cross_observation_noise_diagnostics(
        diagnostics
    )

    return diagnostics


__all__ = [
    "build_multi_observation_problem_from_config",
    "run_multi_observation_fit_from_config",
    "run_multi_observation_inference_from_config",
]
