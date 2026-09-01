"""Tests for equal-weight posterior propagation into radial profiles."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from baymorph.analysis.posterior_profiles import (
    _as_samples_array,
)
from baymorph.pipeline.radial_profile import (
    _posterior_samples,
)


def test_pipeline_prefers_nested_equal_weight_samples():
    """Inference results must not propagate backend-native weighted samples."""
    raw = np.asarray(
        [
            [1.0],
            [2.0],
            [100.0],
        ]
    )

    equal_weight = np.asarray(
        [
            [1.0],
            [1.0],
            [2.0],
        ]
    )

    result = SimpleNamespace(
        samples=raw,
        posterior=SimpleNamespace(
            equal_weight_samples=equal_weight,
        ),
    )

    assert np.array_equal(
        _posterior_samples(
            result
        ),
        equal_weight,
    )


def test_analysis_prefers_nested_equal_weight_samples():
    """Direct posterior-profile analysis must also use equal-weight samples."""
    raw = np.asarray(
        [
            [1.0],
            [2.0],
            [100.0],
        ]
    )

    equal_weight = np.asarray(
        [
            [1.0],
            [1.0],
            [2.0],
        ]
    )

    result = SimpleNamespace(
        samples=raw,
        posterior=SimpleNamespace(
            equal_weight_samples=equal_weight,
        ),
    )

    assert np.array_equal(
        _as_samples_array(
            result
        ),
        equal_weight,
    )


def test_analysis_preserves_raw_equal_weight_arrays():
    """Explicit sample arrays remain supported."""
    samples = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    assert np.array_equal(
        _as_samples_array(
            samples
        ),
        samples,
    )
