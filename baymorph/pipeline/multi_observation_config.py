"""Canonical configuration API for multi-observation inference."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from baymorph.fitting.multi_observation import (
    MultiObservationParameterLayout,
)
from baymorph.pipeline.multiband_config import (
    build_multiband_parameter_layout,
    multiband_parameter_frame,
    parse_prior_config,
    resolve_multiband_config,
    resolve_multiband_observation_configs,
)


def resolve_multi_observation_observation_configs(
    config: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve all single-observation configurations."""
    return resolve_multiband_observation_configs(
        config,
        base_dir=base_dir,
    )


def build_multi_observation_parameter_layout(
    config: Mapping[str, Any],
) -> MultiObservationParameterLayout:
    """Build the canonical multi-observation parameter layout."""
    return build_multiband_parameter_layout(
        config
    )


def resolve_multi_observation_config(
    config: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> tuple[
    dict[str, dict[str, Any]],
    MultiObservationParameterLayout,
]:
    """Resolve observation configurations and parameter coupling."""
    return resolve_multiband_config(
        config,
        base_dir=base_dir,
    )


def multi_observation_parameter_frame(
    config: Mapping[str, Any],
) -> str:
    """Return the declared multi-observation parameter frame."""
    return multiband_parameter_frame(
        config
    )


__all__ = [
    "build_multi_observation_parameter_layout",
    "multi_observation_parameter_frame",
    "parse_prior_config",
    "resolve_multi_observation_config",
    "resolve_multi_observation_observation_configs",
]
