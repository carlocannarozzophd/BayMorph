"""Tests for the native nested-sampling iteration engine."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.geometry import (
    GeometryModel,
    ParameterGeometry,
)
from baymorph.fitting.geometry.nested import (
    GeometryNestedSampler,
    IterationResult,
    NestedSamplerConfig,
)
from baymorph.fitting.geometry.nested.evidence import (
    update_log_evidence,
)
from baymorph.fitting.geometry.nested.shrinkage import (
    ExpectedShrinkageModel,
)
from baymorph.fitting.geometry.proposals import (
    PriorProposalEngine,
    ProposalEngine,
    ProposalFailure,
)
from baymorph.fitting.parameters import (
    Parameter,
    ParameterSet,
)


def make_geometry() -> GeometryModel:
    """Create a simple two-dimensional geometry."""
    return GeometryModel(
        [
            ParameterGeometry(
                name="x",
                block="position",
            ),
            ParameterGeometry(
                name="y",
                block="position",
            ),
        ]
    )


def make_parameters() -> ParameterSet:
    """Create two independent uniform parameters."""
    return ParameterSet(
        [
            Parameter("x", 0.0, 1.0),
            Parameter("y", 0.0, 1.0),
        ]
    )


def make_sampler(
    *,
    seed: int = 12,
    proposal_engine: ProposalEngine | None = None,
) -> GeometryNestedSampler:
    """Create one deterministic test sampler."""
    return GeometryNestedSampler(
        geometry=make_geometry(),
        parameter_set=make_parameters(),
        likelihood=lambda values: float(
            np.sum(values)
        ),
        proposal_engine=proposal_engine,
        config=NestedSamplerConfig(
            nlive=8,
            seed=seed,
        ),
    )


class FailingProposalEngine(ProposalEngine):
    """Proposal engine that always fails without evaluating likelihood."""

    identifier = "failing"

    def propose(
        self,
        *,
        live_points,
        logl_min,
        parameter_set,
        likelihood,
        rng,
    ):
        del live_points
        del logl_min
        del parameter_set
        del likelihood
        del rng

        raise ProposalFailure(
            "Deliberate proposal failure."
        )


def test_step_executes_one_complete_iteration():
    sampler = make_sampler()
    sampler.initialize()

    initial_values = sampler.live_points.values.copy()
    initial_logl = sampler.live_points.logl.copy()
    initial_worst = sampler.worst
    initial_ncall = sampler.state.ncall

    result = sampler.step()

    assert isinstance(result, IterationResult)
    assert sampler.state.iteration == 1
    assert sampler.state.ndead == 1
    assert sampler.state.ncall == (
        initial_ncall + result.proposal.ncall
    )
    assert len(sampler.live_points) == sampler.nlive

    dead_point = sampler.state.dead_points[0]

    assert np.array_equal(
        dead_point.values,
        initial_worst.values,
    )
    assert dead_point.logl == pytest.approx(
        initial_worst.logl
    )
    assert result.proposal.logl > initial_worst.logl
    assert sampler.state.log_prior_volume == pytest.approx(
        -1.0 / sampler.nlive
    )

    expected_logz = update_log_evidence(
        current_logz=-np.inf,
        log_weight=result.plan.log_weight,
        logl=initial_worst.logl,
    )

    assert sampler.state.logz == pytest.approx(
        expected_logz
    )

    assert not np.array_equal(
        sampler.live_points.values,
        initial_values,
    )
    assert not np.array_equal(
        sampler.live_points.logl,
        initial_logl,
    )


def test_step_is_deterministic_for_fixed_seed():
    first = make_sampler(seed=31)
    second = make_sampler(seed=31)

    first.initialize()
    second.initialize()

    first_result = first.step()
    second_result = second.step()

    assert np.array_equal(
        first_result.proposal.values,
        second_result.proposal.values,
    )
    assert first_result.proposal.logl == pytest.approx(
        second_result.proposal.logl
    )
    assert first.state.logz == pytest.approx(
        second.state.logz
    )
    assert np.array_equal(
        first.live_points.values,
        second.live_points.values,
    )


def test_step_preserves_run_after_proposal_failure():
    sampler = make_sampler(
        proposal_engine=FailingProposalEngine()
    )
    sampler.initialize()

    values_before = sampler.live_points.values.copy()
    logl_before = sampler.live_points.logl.copy()
    iteration_before = sampler.state.iteration
    ncall_before = sampler.state.ncall
    logx_before = sampler.state.log_prior_volume
    logz_before = sampler.state.logz
    dead_before = list(sampler.state.dead_points)
    diagnostics_before = list(
        sampler.state.diagnostics
    )

    with pytest.raises(
        ProposalFailure,
        match="Deliberate proposal failure",
    ):
        sampler.step()

    assert np.array_equal(
        sampler.live_points.values,
        values_before,
    )
    assert np.array_equal(
        sampler.live_points.logl,
        logl_before,
    )
    assert sampler.state.iteration == iteration_before
    assert sampler.state.ncall == ncall_before
    assert sampler.state.log_prior_volume == logx_before
    assert sampler.state.logz == logz_before
    assert sampler.state.dead_points == dead_before
    assert sampler.state.diagnostics == diagnostics_before


def test_multiple_steps_preserve_nested_sampling_invariants():
    sampler = make_sampler(
        seed=7,
        proposal_engine=PriorProposalEngine(
            max_attempts=10_000
        ),
    )
    sampler.initialize()

    previous_logx = sampler.state.log_prior_volume
    previous_logz = sampler.state.logz
    previous_dead_logl = -np.inf

    for expected_iteration in range(1, 6):
        result = sampler.step()

        assert sampler.state.iteration == expected_iteration
        assert len(sampler.live_points) == sampler.nlive
        assert result.plan.logx_current < previous_logx
        assert sampler.state.logz >= previous_logz
        assert result.dead_point.logl >= previous_dead_logl
        assert result.proposal.logl > result.dead_point.logl

        previous_logx = sampler.state.log_prior_volume
        previous_logz = sampler.state.logz
        previous_dead_logl = result.dead_point.logl


def test_expected_shrinkage_matches_iteration_count():
    sampler = make_sampler()
    sampler.initialize()

    model = ExpectedShrinkageModel()

    for iteration in range(1, 4):
        result = sampler.step()

        expected = model.log_prior_volume(
            iteration=iteration,
            nlive=sampler.nlive,
            current_log_prior_volume=(
                result.plan.logx_previous
            ),
            rng=sampler.rng,
        )

        assert result.plan.logx_current == pytest.approx(
            expected
        )


def test_step_requires_initialized_run():
    sampler = make_sampler()

    with pytest.raises(
        RuntimeError,
        match="must be initialized",
    ):
        sampler.step()
