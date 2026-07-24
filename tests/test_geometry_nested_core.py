"""Tests for nested-sampling evidence utilities and state records."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.geometry.nested.evidence import (
    expected_log_prior_volume,
    log_shell_weight,
    logdiffexp,
    update_log_evidence,
)
from baymorph.fitting.geometry.nested.state import (
    DeadPoint,
    NestedState,
)


def make_dead_point(
    *,
    iteration: int = 1,
    logl: float = -2.0,
    logx_previous: float = 0.0,
    logx_current: float = -0.1,
) -> DeadPoint:
    """Create one consistent dead-point record."""
    log_weight = logdiffexp(
        logx_previous,
        logx_current,
    )

    return DeadPoint(
        values=np.asarray([1.0, 2.0]),
        logl=logl,
        logx_previous=logx_previous,
        logx_current=logx_current,
        log_weight=log_weight,
        log_weighted_likelihood=log_weight + logl,
        iteration=iteration,
    )


def test_logdiffexp_matches_linear_difference():
    first = np.log(0.8)
    second = np.log(0.3)

    result = logdiffexp(first, second)

    assert np.exp(result) == pytest.approx(0.5)


def test_logdiffexp_rejects_invalid_ordering():
    with pytest.raises(
        ValueError,
        match="requires first > second",
    ):
        logdiffexp(-1.0, -1.0)

    with pytest.raises(
        ValueError,
        match="requires first > second",
    ):
        logdiffexp(-2.0, -1.0)


def test_expected_log_prior_volume_uses_deterministic_shrinkage():
    assert expected_log_prior_volume(
        iteration=0,
        nlive=10,
    ) == pytest.approx(0.0)

    assert expected_log_prior_volume(
        iteration=3,
        nlive=10,
    ) == pytest.approx(-0.3)


def test_expected_log_prior_volume_validates_inputs():
    with pytest.raises(
        ValueError,
        match="Iteration must be non-negative",
    ):
        expected_log_prior_volume(
            iteration=-1,
            nlive=10,
        )

    with pytest.raises(
        ValueError,
        match="nlive must be positive",
    ):
        expected_log_prior_volume(
            iteration=1,
            nlive=0,
        )


def test_log_shell_weight_matches_expected_volume_difference():
    nlive = 5
    iteration = 2

    result = log_shell_weight(
        iteration=iteration,
        nlive=nlive,
    )

    previous = np.exp(
        expected_log_prior_volume(
            iteration=iteration - 1,
            nlive=nlive,
        )
    )
    current = np.exp(
        expected_log_prior_volume(
            iteration=iteration,
            nlive=nlive,
        )
    )

    assert np.exp(result) == pytest.approx(
        previous - current
    )


def test_log_shell_weight_rejects_zero_iteration():
    with pytest.raises(
        ValueError,
        match="start at iteration 1",
    ):
        log_shell_weight(
            iteration=0,
            nlive=10,
        )


def test_update_log_evidence_accumulates_contributions():
    first = update_log_evidence(
        current_logz=-np.inf,
        log_weight=np.log(0.2),
        logl=np.log(3.0),
    )

    second = update_log_evidence(
        current_logz=first,
        log_weight=np.log(0.3),
        logl=np.log(2.0),
    )

    assert np.exp(first) == pytest.approx(0.6)
    assert np.exp(second) == pytest.approx(1.2)


def test_dead_point_freezes_values():
    values = np.asarray([1.0, 2.0])
    dead_point = make_dead_point()

    assert dead_point.values.flags.writeable is False

    values[0] = 99.0

    assert dead_point.values[0] == pytest.approx(1.0)


def test_dead_point_rejects_non_decreasing_prior_volume():
    with pytest.raises(
        ValueError,
        match="prior volume must decrease",
    ):
        DeadPoint(
            values=np.asarray([1.0, 2.0]),
            logl=-2.0,
            logx_previous=-0.2,
            logx_current=-0.1,
            log_weight=-3.0,
            log_weighted_likelihood=-5.0,
            iteration=1,
        )


def test_nested_state_defaults():
    state = NestedState()

    assert state.iteration == 0
    assert state.ncall == 0
    assert state.initialized is False
    assert state.log_prior_volume == pytest.approx(0.0)
    assert state.logz == -np.inf
    assert state.ndead == 0
    assert state.terminated is False
    assert state.finalized is False


def test_nested_state_appends_dead_points_in_order():
    state = NestedState()

    first = make_dead_point(
        iteration=1,
        logl=-3.0,
        logx_previous=0.0,
        logx_current=-0.1,
    )
    second = make_dead_point(
        iteration=2,
        logl=-2.0,
        logx_previous=-0.1,
        logx_current=-0.2,
    )

    state.append_dead_point(first)
    state.append_dead_point(second)

    assert state.ndead == 2
    assert state.dead_points == [first, second]


def test_nested_state_rejects_dead_point_iteration_gaps():
    state = NestedState()

    with pytest.raises(
        ValueError,
        match="does not match removal order",
    ):
        state.append_dead_point(
            make_dead_point(
                iteration=2,
            )
        )


def test_nested_state_rejects_decreasing_dead_likelihood():
    state = NestedState()

    state.append_dead_point(
        make_dead_point(
            iteration=1,
            logl=-1.0,
            logx_previous=0.0,
            logx_current=-0.1,
        )
    )

    with pytest.raises(
        ValueError,
        match="must be non-decreasing",
    ):
        state.append_dead_point(
            make_dead_point(
                iteration=2,
                logl=-2.0,
                logx_previous=-0.1,
                logx_current=-0.2,
            )
        )
