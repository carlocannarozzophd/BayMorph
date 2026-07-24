"""Tests for native nested-sampling finalization."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.geometry import (
    GeometryModel,
    ParameterGeometry,
)
from baymorph.fitting.geometry.nested import (
    GeometryNestedSampler,
    NestedSamplerConfig,
)
from baymorph.fitting.parameters import (
    Parameter,
    ParameterSet,
)
from baymorph.results.posterior import PosteriorResult


def make_sampler() -> GeometryNestedSampler:
    """Create a deterministic one-dimensional sampler."""
    geometry = GeometryModel(
        [
            ParameterGeometry(
                name="x",
            )
        ]
    )
    parameter_set = ParameterSet(
        [
            Parameter(
                "x",
                0.0,
                1.0,
            )
        ]
    )

    return GeometryNestedSampler(
        geometry=geometry,
        parameter_set=parameter_set,
        likelihood=lambda values: float(values[0]),
        config=NestedSamplerConfig(
            nlive=5,
            dlogz=1.0e-12,
            max_iterations=3,
            seed=23,
        ),
    )


def test_finalize_requires_terminated_sampler():
    sampler = make_sampler()
    sampler.initialize()

    with pytest.raises(
        RuntimeError,
        match="must terminate",
    ):
        sampler.finalize()


def test_finalize_returns_posterior_result():
    sampler = make_sampler()
    sampler.run_active_phase()

    result = sampler.finalize()

    assert isinstance(
        result,
        PosteriorResult,
    )
    assert result.samples.shape == (
        sampler.state.ndead + sampler.nlive,
        sampler.parameter_set.ndim,
    )
    assert result.logl is not None
    assert result.logl.shape == (
        result.samples.shape[0],
    )
    assert result.logwt.shape == (
        result.samples.shape[0],
    )
    assert result.logz.shape == (
        result.samples.shape[0],
    )
    assert result.parameter_names == ["x"]
    assert np.sum(result.weights) == pytest.approx(
        1.0
    )
    assert result.final_logz == pytest.approx(
        sampler.state.logz
    )
    assert sampler.state.finalized is True


def test_finalize_assigns_remaining_mass_to_live_points():
    sampler = make_sampler()
    sampler.run_active_phase()

    logx_final = sampler.state.log_prior_volume
    live_logl = np.sort(
        sampler.live_points.logl
    )

    result = sampler.finalize()

    expected_live_logwt = (
        logx_final
        - np.log(sampler.nlive)
        + live_logl
    )

    np.testing.assert_allclose(
        result.logwt[-sampler.nlive :],
        expected_live_logwt,
    )


def test_finalize_preserves_dead_point_contributions():
    sampler = make_sampler()
    sampler.run_active_phase()

    expected_dead_logwt = np.asarray(
        [
            point.log_weighted_likelihood
            for point in sampler.state.dead_points
        ],
        dtype=float,
    )

    result = sampler.finalize()

    np.testing.assert_allclose(
        result.logwt[: sampler.state.ndead],
        expected_dead_logwt,
    )


def test_finalize_cannot_run_twice():
    sampler = make_sampler()
    sampler.run_active_phase()
    sampler.finalize()

    with pytest.raises(
        RuntimeError,
        match="already been finalized",
    ):
        sampler.finalize()
