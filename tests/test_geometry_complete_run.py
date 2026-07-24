"""Tests for complete native nested-sampling execution."""

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


def make_sampler(
    *,
    seed: int = 41,
) -> GeometryNestedSampler:
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
            nlive=6,
            dlogz=1.0e-12,
            max_iterations=4,
            seed=seed,
        ),
    )


def test_run_executes_complete_sampler_lifecycle():
    sampler = make_sampler()

    result = sampler.run()

    assert isinstance(
        result,
        PosteriorResult,
    )
    assert sampler.state.initialized is True
    assert sampler.state.terminated is True
    assert sampler.state.finalized is True
    assert sampler.state.active is False
    assert sampler.state.iteration == 4
    assert sampler.state.ndead == 4
    assert len(sampler.live_points) == sampler.nlive


def test_run_returns_all_dead_and_live_points():
    sampler = make_sampler()

    result = sampler.run()

    expected_samples = (
        sampler.state.ndead
        + sampler.nlive
    )

    assert result.samples.shape == (
        expected_samples,
        sampler.parameter_set.ndim,
    )
    assert result.logl is not None
    assert result.logl.shape == (
        expected_samples,
    )
    assert result.logwt.shape == (
        expected_samples,
    )
    assert result.logz.shape == (
        expected_samples,
    )
    assert result.parameter_names == ["x"]
    assert np.sum(result.weights) == pytest.approx(
        1.0
    )
    assert result.final_logz == pytest.approx(
        sampler.state.logz
    )


def test_run_is_deterministic_for_fixed_seed():
    first = make_sampler(
        seed=73
    )
    second = make_sampler(
        seed=73
    )

    first_result = first.run()
    second_result = second.run()

    np.testing.assert_allclose(
        first_result.samples,
        second_result.samples,
    )
    np.testing.assert_allclose(
        first_result.logl,
        second_result.logl,
    )
    np.testing.assert_allclose(
        first_result.logwt,
        second_result.logwt,
    )
    np.testing.assert_allclose(
        first_result.logz,
        second_result.logz,
    )
    assert first.state.ncall == second.state.ncall


def test_run_resumes_initialized_active_sampler():
    sampler = make_sampler()
    sampler.initialize()
    sampler.step()

    initial_iteration = sampler.state.iteration
    initial_ncall = sampler.state.ncall

    result = sampler.run()

    assert initial_iteration == 1
    assert sampler.state.iteration == 4
    assert sampler.state.ncall > initial_ncall
    assert sampler.state.terminated is True
    assert sampler.state.finalized is True
    assert result.final_logz == pytest.approx(
        sampler.state.logz
    )


def test_run_cannot_execute_twice():
    sampler = make_sampler()
    sampler.run()

    with pytest.raises(
        RuntimeError,
        match="already been finalized",
    ):
        sampler.run()
