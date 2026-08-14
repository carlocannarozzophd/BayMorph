"""Compatibility aliases for legacy multi-band configuration APIs."""

from baymorph.pipeline.multi_observation_config import (
    SINGLE_SERSIC_PARAMETER_NAMES,
    build_multi_observation_parameter_layout,
    multi_observation_parameter_frame,
    parse_prior_config,
    resolve_multi_observation_config,
    resolve_multi_observation_observation_configs,
)

build_multiband_parameter_layout = (
    build_multi_observation_parameter_layout
)

multiband_parameter_frame = (
    multi_observation_parameter_frame
)

resolve_multiband_config = (
    resolve_multi_observation_config
)

resolve_multiband_observation_configs = (
    resolve_multi_observation_observation_configs
)

__all__ = [
    "SINGLE_SERSIC_PARAMETER_NAMES",
    "build_multiband_parameter_layout",
    "multiband_parameter_frame",
    "parse_prior_config",
    "resolve_multiband_config",
    "resolve_multiband_observation_configs",
]
