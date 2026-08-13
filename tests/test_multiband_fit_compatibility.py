"""Tests for the legacy multi-band fitting compatibility surface."""

from baymorph.pipeline import multiband_fit
from baymorph.pipeline.multi_observation_fit import (
    build_multi_observation_problem_from_config,
    run_multi_observation_fit_from_config,
    run_multi_observation_inference_from_config,
)


def test_legacy_problem_builder_is_canonical_alias():
    assert (
        multiband_fit.build_multiband_problem_from_config
        is build_multi_observation_problem_from_config
    )


def test_legacy_inference_entry_point_is_canonical_alias():
    assert (
        multiband_fit.run_multiband_fit_from_config
        is run_multi_observation_inference_from_config
    )


def test_canonical_fit_products_entry_point_remains_available():
    assert (
        multiband_fit.run_multi_observation_fit_from_config
        is run_multi_observation_fit_from_config
    )
