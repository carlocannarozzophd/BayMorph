from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.geometry import GeometryModel, ParameterGeometry
from baymorph.fitting.geometry.nested.live_points import LivePoints
from baymorph.fitting.geometry.nested.sampler import GeometryNestedSampler
from baymorph.fitting.geometry.proposals import (
    PriorProposalEngine,
    ProposalFailure,
    ProposalResult,
)
from baymorph.fitting.parameters import Parameter, ParameterSet


def make_geometry() -> GeometryModel:
    return GeometryModel(
        [
            ParameterGeometry("x", block="position"),
            ParameterGeometry("y", block="position"),
        ]
    )


def make_parameters() -> ParameterSet:
    return ParameterSet(
        [
            Parameter("x", 0.0, 1.0),
            Parameter("y", 0.0, 1.0),
        ]
    )


def make_live_points() -> LivePoints:
    live_points = LivePoints(make_geometry())
    live_points.append([0.1, 0.1], 0.2)
    live_points.append([0.4, 0.4], 0.8)
    return live_points


def test_proposal_result_copies_and_freezes_values():
    values = np.array([0.2, 0.3])
    result = ProposalResult(
        values=values,
        logl=0.5,
        ncall=1,
        attempts=1,
        engine="test",
    )
    values[0] = 0.9

    assert result.values[0] == pytest.approx(0.2)
    assert result.values.flags.writeable is False
    with pytest.raises(ValueError):
        result.values[0] = 0.1


def test_proposal_result_rejects_invalid_accounting():
    with pytest.raises(ValueError, match="attempts cannot be smaller"):
        ProposalResult(
            values=np.array([0.2]),
            logl=0.0,
            ncall=2,
            attempts=1,
            engine="test",
        )


def test_prior_proposal_is_deterministic():
    engine = PriorProposalEngine(max_attempts=20)
    kwargs = {
        "live_points": make_live_points(),
        "logl_min": 0.5,
        "parameter_set": make_parameters(),
        "likelihood": lambda values: float(np.sum(values)),
    }

    first = engine.propose(rng=np.random.default_rng(7), **kwargs)
    second = engine.propose(rng=np.random.default_rng(7), **kwargs)

    assert np.allclose(first.values, second.values)
    assert first.logl == pytest.approx(second.logl)
    assert first.ncall == second.ncall
    assert first.engine == "prior"


def test_prior_proposal_satisfies_strict_constraint():
    engine = PriorProposalEngine(max_attempts=20)
    result = engine.propose(
        live_points=make_live_points(),
        logl_min=0.8,
        parameter_set=make_parameters(),
        likelihood=lambda values: float(np.sum(values)),
        rng=np.random.default_rng(5),
    )

    assert result.logl > 0.8
    assert result.ncall == result.attempts
    assert result.diagnostics["acceptance_fraction"] == pytest.approx(
        1.0 / result.attempts
    )


def test_prior_proposal_rejects_equality_with_threshold():
    engine = PriorProposalEngine(max_attempts=3)

    with pytest.raises(ProposalFailure, match="after 3 attempts"):
        engine.propose(
            live_points=make_live_points(),
            logl_min=1.0,
            parameter_set=make_parameters(),
            likelihood=lambda values: 1.0,
            rng=np.random.default_rng(2),
        )


def test_prior_proposal_fails_after_configured_attempts():
    engine = PriorProposalEngine(max_attempts=4)
    calls = 0

    def likelihood(values):
        nonlocal calls
        calls += 1
        return -1.0

    with pytest.raises(ProposalFailure, match="after 4 attempts"):
        engine.propose(
            live_points=make_live_points(),
            logl_min=0.0,
            parameter_set=make_parameters(),
            likelihood=likelihood,
            rng=np.random.default_rng(11),
        )

    assert calls == 4


def test_prior_proposal_rejects_nan_likelihood():
    engine = PriorProposalEngine(max_attempts=2)

    with pytest.raises(ValueError, match="Likelihood returned NaN"):
        engine.propose(
            live_points=make_live_points(),
            logl_min=0.0,
            parameter_set=make_parameters(),
            likelihood=lambda values: np.nan,
            rng=np.random.default_rng(1),
        )


def test_sampler_replacement_updates_state_and_population():
    sampler = GeometryNestedSampler(
        geometry=make_geometry(),
        parameter_set=make_parameters(),
        likelihood=lambda values: float(np.sum(values)),
        proposal_engine=PriorProposalEngine(max_attempts=100),
        nlive=5,
        seed=13,
    )
    sampler.initialize()
    initial_ncall = sampler.state.ncall
    initial_length = len(sampler.live_points)
    initial_threshold = sampler.worst.logl

    removed, proposal = sampler.replace_worst()

    assert len(sampler.live_points) == initial_length
    assert removed.logl == pytest.approx(initial_threshold)
    assert proposal.logl > removed.logl
    assert sampler.state.ncall == initial_ncall + proposal.ncall
    assert sampler.state.iteration == 1
    assert sampler.worst.logl >= removed.logl


def test_sampler_replacement_requires_initialization():
    sampler = GeometryNestedSampler(
        geometry=make_geometry(),
        parameter_set=make_parameters(),
        likelihood=lambda values: float(np.sum(values)),
        nlive=3,
        seed=4,
    )

    with pytest.raises(RuntimeError, match="must be initialized"):
        sampler.replace_worst()


def test_sampler_uses_prior_proposal_by_default():
    sampler = GeometryNestedSampler(
        geometry=make_geometry(),
        parameter_set=make_parameters(),
        likelihood=lambda values: float(np.sum(values)),
        nlive=3,
        seed=9,
    )

    assert isinstance(sampler.proposal_engine, PriorProposalEngine)
