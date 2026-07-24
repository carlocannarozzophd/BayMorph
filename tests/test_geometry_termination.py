"""Tests for native nested-sampling termination."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.geometry import (
    GeometryModel,
    ParameterGeometry,
)
from baymorph.fitting.geometry.nested import (
    AnyTerminationCriterion,
    EvidenceTerminationCriterion,
    GeometryNestedSampler,
    MaximumIterationTerminationCriterion,
    MaximumLikelihoodCallTerminationCriterion,
    NestedSamplerConfig,
)
from baymorph.fitting.parameters import (
    Parameter,
    ParameterSet,
)


def make_geometry() -> GeometryModel:
    """Create a one-dimensional test geometry."""
    return GeometryModel(
        [
            ParameterGeometry(
                name="x",
            )
        ]
    )


def make_parameters() -> ParameterSet:
    """Create a one-dimensional uniform prior."""
    return ParameterSet(
        [
            Parameter(
                "x",
                0.0,
                1.0,
            )
        ]
    )


def test_evidence_termination_does_not_stop_before_first_iteration():
    criterion = EvidenceTerminationCriterion(
        dlogz=0.1,
    )

    decision = criterion.evaluate(
        iteration=0,
        ncall=10,
        logz=-np.inf,
        logx=0.0,
        live_logl=np.asarray([-2.0, -1.0]),
    )

    assert decision.should_stop is False
    assert decision.delta_logz == np.inf


def test_evidence_termination_stops_below_tolerance():
    criterion = EvidenceTerminationCriterion(
        dlogz=0.2,
    )

    decision = criterion.evaluate(
        iteration=10,
        ncall=100,
        logz=0.0,
        logx=-5.0,
        live_logl=np.asarray([0.0, 1.0]),
    )

    assert decision.should_stop is True
    assert decision.reason == "evidence tolerance reached"
    assert decision.delta_logz is not None
    assert decision.delta_logz < 0.2


def test_maximum_iteration_termination():
    criterion = MaximumIterationTerminationCriterion(
        max_iterations=3,
    )

    decision = criterion.evaluate(
        iteration=3,
        ncall=20,
        logz=-1.0,
        logx=-0.5,
        live_logl=np.asarray([0.0]),
    )

    assert decision.should_stop is True
    assert decision.reason == "maximum iterations reached"


def test_maximum_likelihood_call_termination():
    criterion = MaximumLikelihoodCallTerminationCriterion(
        max_calls=25,
    )

    decision = criterion.evaluate(
        iteration=2,
        ncall=25,
        logz=-1.0,
        logx=-0.5,
        live_logl=np.asarray([0.0]),
    )

    assert decision.should_stop is True
    assert decision.reason == "maximum likelihood calls reached"


def test_any_termination_returns_first_stopping_decision():
    criterion = AnyTerminationCriterion(
        MaximumIterationTerminationCriterion(
            max_iterations=2,
        ),
        MaximumLikelihoodCallTerminationCriterion(
            max_calls=100,
        ),
    )

    decision = criterion.evaluate(
        iteration=2,
        ncall=100,
        logz=-1.0,
        logx=-0.5,
        live_logl=np.asarray([0.0]),
    )

    assert decision.should_stop is True
    assert decision.reason == "maximum iterations reached"


def test_sampler_active_phase_stops_at_maximum_iterations():
    sampler = GeometryNestedSampler(
        geometry=make_geometry(),
        parameter_set=make_parameters(),
        likelihood=lambda values: float(values[0]),
        config=NestedSamplerConfig(
            nlive=10,
            dlogz=1.0e-12,
            max_iterations=4,
            seed=17,
        ),
    )

    results = sampler.run_active_phase()

    assert len(results) == 4
    assert sampler.state.iteration == 4
    assert sampler.state.ndead == 4
    assert sampler.state.terminated is True
    assert sampler.state.finalized is False
    assert len(sampler.live_points) == sampler.nlive


def test_termination_requires_initialization():
    sampler = GeometryNestedSampler(
        geometry=make_geometry(),
        parameter_set=make_parameters(),
        likelihood=lambda values: float(values[0]),
        config=NestedSamplerConfig(
            nlive=5,
            seed=2,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="must be initialized",
    ):
        sampler.termination_decision()


def test_nested_run_finalization_requires_terminated_state():
    sampler = GeometryNestedSampler(
        geometry=make_geometry(),
        parameter_set=make_parameters(),
        likelihood=lambda values: float(values[0]),
        config=NestedSamplerConfig(
            nlive=5,
            seed=11,
        ),
    )
    sampler.initialize()

    with pytest.raises(
        RuntimeError,
        match="active NestedRun",
    ):
        sampler.state.mark_finalized(
            logz=0.0
        )


def test_nested_run_finalization_commits_final_evidence():
    sampler = GeometryNestedSampler(
        geometry=make_geometry(),
        parameter_set=make_parameters(),
        likelihood=lambda values: float(values[0]),
        config=NestedSamplerConfig(
            nlive=5,
            dlogz=1.0e-12,
            max_iterations=1,
            seed=13,
        ),
    )
    sampler.run_active_phase()

    final_logz = sampler.state.logz + 0.5

    sampler.state.mark_finalized(
        logz=final_logz
    )

    assert sampler.state.logz == pytest.approx(
        final_logz
    )
    assert sampler.state.terminated is True
    assert sampler.state.finalized is True
    assert sampler.state.active is False


def test_nested_run_cannot_be_finalized_twice():
    sampler = GeometryNestedSampler(
        geometry=make_geometry(),
        parameter_set=make_parameters(),
        likelihood=lambda values: float(values[0]),
        config=NestedSamplerConfig(
            nlive=5,
            dlogz=1.0e-12,
            max_iterations=1,
            seed=17,
        ),
    )
    sampler.run_active_phase()

    sampler.state.mark_finalized(
        logz=sampler.state.logz
    )

    with pytest.raises(
        RuntimeError,
        match="already been finalized",
    ):
        sampler.state.mark_finalized(
            logz=sampler.state.logz
        )
