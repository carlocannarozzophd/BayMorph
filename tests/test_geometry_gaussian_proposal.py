from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.geometry import GeometryModel, ParameterGeometry
from baymorph.fitting.geometry.nested.live_points import LivePoints
from baymorph.fitting.geometry.proposals import (
    GaussianProposalEngine,
    ProposalFailure,
)
from baymorph.fitting.geometry.transforms import LogTransform
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
    live_points.append([0.20, 0.20], 0.40)
    live_points.append([0.30, 0.40], 0.70)
    live_points.append([0.40, 0.30], 0.70)
    live_points.append([0.50, 0.50], 1.00)
    return live_points


class StubRng:
    def __init__(self, candidates: list[list[float]]) -> None:
        self._candidates = [
            np.asarray(candidate, dtype=float)
            for candidate in candidates
        ]
        self.calls = 0

    def multivariate_normal(
        self,
        *,
        mean,
        cov,
        check_valid,
    ) -> np.ndarray:
        assert np.asarray(mean).ndim == 1
        assert np.asarray(cov).ndim == 2
        assert check_valid == "raise"

        candidate = self._candidates[self.calls]
        self.calls += 1
        return candidate.copy()


def test_gaussian_proposal_is_exported():
    assert GaussianProposalEngine.identifier == "gaussian"


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("covariance_scale", 0.0, "covariance_scale"),
        ("covariance_scale", np.inf, "covariance_scale"),
        ("regularization", -1.0, "regularization"),
        (
            "absolute_regularization",
            0.0,
            "absolute_regularization",
        ),
        ("max_attempts", 0, "max_attempts"),
    ],
)
def test_gaussian_proposal_rejects_invalid_configuration(
    keyword,
    value,
    message,
):
    with pytest.raises(ValueError, match=message):
        GaussianProposalEngine(**{keyword: value})


def test_gaussian_proposal_satisfies_strict_constraint():
    engine = GaussianProposalEngine(max_attempts=10)

    result = engine.propose(
        live_points=make_live_points(),
        logl_min=0.7,
        parameter_set=make_parameters(),
        likelihood=lambda values: float(np.sum(values)),
        rng=np.random.default_rng(12),
    )

    assert result.logl > 0.7
    assert result.ncall <= result.attempts
    assert result.engine == "gaussian"
    assert result.diagnostics["acceptance_fraction"] == pytest.approx(
        1.0 / result.attempts
    )
    assert result.diagnostics["likelihood_fraction"] == pytest.approx(
        result.ncall / result.attempts
    )


def test_gaussian_proposal_rejects_out_of_bounds_without_likelihood_call():
    engine = GaussianProposalEngine(max_attempts=2)
    rng = StubRng(
        [
            [2.0, 2.0],
            [0.6, 0.6],
        ]
    )
    calls = 0

    def likelihood(values):
        nonlocal calls
        calls += 1
        return float(np.sum(values))

    result = engine.propose(
        live_points=make_live_points(),
        logl_min=1.0,
        parameter_set=make_parameters(),
        likelihood=likelihood,
        rng=rng,
    )

    assert calls == 1
    assert result.ncall == 1
    assert result.attempts == 2
    assert np.allclose(result.values, [0.6, 0.6])
    assert result.diagnostics[
        "bounds_rejection_fraction"
    ] == pytest.approx(0.5)


def test_gaussian_proposal_rejects_likelihood_equality():
    engine = GaussianProposalEngine(max_attempts=2)
    rng = StubRng(
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )

    with pytest.raises(ProposalFailure, match="after 2 attempts"):
        engine.propose(
            live_points=make_live_points(),
            logl_min=1.0,
            parameter_set=make_parameters(),
            likelihood=lambda values: 1.0,
            rng=rng,
        )


def test_gaussian_proposal_reports_accurate_likelihood_calls():
    engine = GaussianProposalEngine(max_attempts=3)
    rng = StubRng(
        [
            [-1.0, 0.5],
            [0.2, 0.2],
            [0.7, 0.7],
        ]
    )
    calls = 0

    def likelihood(values):
        nonlocal calls
        calls += 1
        return float(np.sum(values))

    result = engine.propose(
        live_points=make_live_points(),
        logl_min=1.0,
        parameter_set=make_parameters(),
        likelihood=likelihood,
        rng=rng,
    )

    assert calls == 2
    assert result.ncall == 2
    assert result.attempts == 3
    assert result.diagnostics["likelihood_fraction"] == pytest.approx(
        2.0 / 3.0
    )
    assert result.diagnostics[
        "bounds_rejection_fraction"
    ] == pytest.approx(1.0 / 3.0)


def test_gaussian_proposal_fails_after_configured_attempts():
    engine = GaussianProposalEngine(max_attempts=3)
    rng = StubRng(
        [
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
        ]
    )

    with pytest.raises(
        ProposalFailure,
        match="after 3 attempts and 3 likelihood evaluations",
    ):
        engine.propose(
            live_points=make_live_points(),
            logl_min=2.0,
            parameter_set=make_parameters(),
            likelihood=lambda values: float(np.sum(values)),
            rng=rng,
        )


def test_gaussian_proposal_rejects_nan_likelihood():
    engine = GaussianProposalEngine(max_attempts=1)
    rng = StubRng([[0.5, 0.5]])

    with pytest.raises(ValueError, match="Likelihood returned NaN"):
        engine.propose(
            live_points=make_live_points(),
            logl_min=0.0,
            parameter_set=make_parameters(),
            likelihood=lambda values: np.nan,
            rng=rng,
        )


def test_gaussian_proposal_regularizes_degenerate_covariance():
    geometry = GeometryModel(
        [
            ParameterGeometry("x", block="position"),
            ParameterGeometry("y", block="position"),
        ]
    )
    live_points = LivePoints(geometry)
    live_points.append([0.5, 0.5], 1.0)
    live_points.append([0.5, 0.5], 1.0)
    live_points.append([0.5, 0.5], 1.0)

    engine = GaussianProposalEngine(
        covariance_scale=1.0,
        max_attempts=100,
    )
    result = engine.propose(
        live_points=live_points,
        logl_min=0.0,
        parameter_set=make_parameters(),
        likelihood=lambda values: 1.0,
        rng=np.random.default_rng(5),
    )

    assert np.all(np.isfinite(result.values))
    assert result.logl == pytest.approx(1.0)


def test_gaussian_proposal_uses_proposal_space():
    geometry = GeometryModel(
        [
            ParameterGeometry(
                "scale",
                block="structure",
                transform=LogTransform(),
            )
        ]
    )
    parameters = ParameterSet(
        [
            Parameter("scale", 0.1, 10.0),
        ]
    )
    live_points = LivePoints(geometry)
    live_points.append([1.0], 0.0)
    live_points.append([2.0], 0.0)
    live_points.append([4.0], 0.0)

    proposal_value = float(np.log(3.0))
    rng = StubRng([[proposal_value]])

    result = GaussianProposalEngine(max_attempts=1).propose(
        live_points=live_points,
        logl_min=0.0,
        parameter_set=parameters,
        likelihood=lambda values: float(values[0]),
        rng=rng,
    )

    assert result.values[0] == pytest.approx(3.0)
    assert result.logl == pytest.approx(3.0)


def test_gaussian_proposal_validates_parameter_order():
    parameters = ParameterSet(
        [
            Parameter("y", 0.0, 1.0),
            Parameter("x", 0.0, 1.0),
        ]
    )

    with pytest.raises(ValueError, match="parameter ordering"):
        GaussianProposalEngine().propose(
            live_points=make_live_points(),
            logl_min=0.0,
            parameter_set=parameters,
            likelihood=lambda values: 1.0,
            rng=np.random.default_rng(1),
        )


def test_gaussian_proposal_rejects_nan_threshold():
    with pytest.raises(ValueError, match="logl_min"):
        GaussianProposalEngine().propose(
            live_points=make_live_points(),
            logl_min=np.nan,
            parameter_set=make_parameters(),
            likelihood=lambda values: 1.0,
            rng=np.random.default_rng(1),
        )
