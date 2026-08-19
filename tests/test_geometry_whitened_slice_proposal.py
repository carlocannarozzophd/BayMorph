"""Tests for the experimental prior-cube whitened slice proposal."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting.geometry import (
    GeometryModel,
    ParameterGeometry,
)
from baymorph.fitting.geometry.nested.live_points import LivePoints
from baymorph.fitting.geometry.proposals import (
    WhitenedSliceProposalEngine,
)
from baymorph.fitting.parameters import (
    Parameter,
    ParameterSet,
)
from baymorph.fitting.priors import (
    BetaPrior,
    LogUniformPrior,
    NormalPrior,
    UniformPrior,
)


def make_geometry(
    names: tuple[str, ...] = ("x", "y"),
) -> GeometryModel:
    """Return a simple geometry with one block."""
    return GeometryModel(
        [
            ParameterGeometry(
                name,
                block="test",
            )
            for name in names
        ]
    )


def make_parameters() -> ParameterSet:
    """Return a two-dimensional uniform prior."""
    return ParameterSet(
        [
            Parameter(
                "x",
                0.0,
                1.0,
            ),
            Parameter(
                "y",
                0.0,
                1.0,
            ),
        ]
    )


def make_live_points() -> LivePoints:
    """Return live points spanning a correlated constrained region."""
    live_points = LivePoints(
        make_geometry()
    )

    for values in (
        (0.30, 0.42),
        (0.40, 0.50),
        (0.50, 0.58),
        (0.60, 0.66),
        (0.70, 0.74),
        (0.80, 0.82),
    ):
        array = np.asarray(
            values,
            dtype=float,
        )

        live_points.append(
            array,
            float(
                np.sum(array)
            ),
        )

    return live_points


def test_whitened_slice_identifier_is_stable():
    """The engine identifier is part of proposal diagnostics."""
    assert (
        WhitenedSliceProposalEngine.identifier
        == "whitened_slice"
    )


def test_prior_cube_inverse_round_trips_supported_priors():
    """All built-in sampled priors must map back to the unit cube."""
    parameter_set = ParameterSet(
        [
            Parameter(
                "uniform",
                prior=UniformPrior(
                    lower=-3.0,
                    upper=5.0,
                ),
            ),
            Parameter(
                "log_uniform",
                prior=LogUniformPrior(
                    lower=1.0e-3,
                    upper=1.0e3,
                ),
            ),
            Parameter(
                "normal",
                prior=NormalPrior(
                    mean=2.0,
                    sigma=0.7,
                ),
            ),
            Parameter(
                "beta",
                prior=BetaPrior(
                    alpha=2.0,
                    beta=4.0,
                    lower=0.2,
                    upper=0.95,
                ),
            ),
        ]
    )

    unit = np.asarray(
        [
            0.13,
            0.31,
            0.67,
            0.82,
        ],
        dtype=float,
    )

    physical = parameter_set.transform(
        unit
    )

    recovered = (
        WhitenedSliceProposalEngine._physical_to_unit_cube(
            parameter_set,
            physical,
        )
    )

    np.testing.assert_allclose(
        recovered,
        unit,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_whitened_slice_proposal_is_deterministic_for_fixed_seed():
    """A fixed RNG seed must reproduce the same constrained chain."""
    engine = WhitenedSliceProposalEngine(
        n_repeats=5
    )

    kwargs = {
        "live_points": make_live_points(),
        "logl_min": 0.80,
        "parameter_set": make_parameters(),
        "likelihood": lambda values: float(
            np.sum(values)
        ),
    }

    first = engine.propose(
        rng=np.random.default_rng(19),
        **kwargs,
    )

    second = engine.propose(
        rng=np.random.default_rng(19),
        **kwargs,
    )

    np.testing.assert_allclose(
        first.values,
        second.values,
    )

    assert first.logl == pytest.approx(
        second.logl
    )
    assert first.ncall == second.ncall
    assert first.attempts == second.attempts


def test_whitened_slice_proposal_satisfies_strict_constraint():
    """The returned physical point must satisfy the nested constraint."""
    threshold = 1.05

    result = WhitenedSliceProposalEngine(
        n_repeats=6
    ).propose(
        live_points=make_live_points(),
        logl_min=threshold,
        parameter_set=make_parameters(),
        likelihood=lambda values: float(
            np.sum(values)
        ),
        rng=np.random.default_rng(7),
    )

    assert result.logl > threshold
    assert result.engine == "whitened_slice"
    assert result.ncall > 0
    assert result.attempts >= result.ncall

    assert result.diagnostics[
        "n_repeats"
    ] == pytest.approx(
        6.0
    )

    assert result.diagnostics[
        "likelihood_calls_per_slice"
    ] > 0.0

    assert result.diagnostics[
        "whitening_condition_number"
    ] >= 1.0

    unit = (
        WhitenedSliceProposalEngine._physical_to_unit_cube(
            make_parameters(),
            result.values,
        )
    )

    assert np.all(
        unit >= 0.0
    )
    assert np.all(
        unit <= 1.0
    )


def test_whitened_slice_regularizes_degenerate_live_covariance():
    """Exactly collinear live points must still yield a valid proposal."""
    geometry = make_geometry()

    live_points = LivePoints(
        geometry
    )

    for value in (
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
    ):
        values = np.asarray(
            [
                value,
                value,
            ],
            dtype=float,
        )

        live_points.append(
            values,
            float(
                np.sum(values)
            ),
        )

    result = WhitenedSliceProposalEngine(
        n_repeats=4,
        regularization=1.0e-6,
    ).propose(
        live_points=live_points,
        logl_min=0.70,
        parameter_set=make_parameters(),
        likelihood=lambda values: float(
            np.sum(values)
        ),
        rng=np.random.default_rng(31),
    )

    assert np.all(
        np.isfinite(
            result.values
        )
    )

    assert result.logl > 0.70

    assert result.diagnostics[
        "whitening_condition_number"
    ] > 1.0


def test_whitened_slice_supports_nonuniform_physical_priors():
    """The chain must operate through the prior cube, not physical space."""
    parameter_set = ParameterSet(
        [
            Parameter(
                "scale",
                prior=LogUniformPrior(
                    lower=1.0e-2,
                    upper=1.0e2,
                ),
            )
        ]
    )

    geometry = make_geometry(
        ("scale",)
    )

    live_points = LivePoints(
        geometry
    )

    for unit_value in (
        0.20,
        0.35,
        0.50,
        0.65,
        0.80,
    ):
        physical = parameter_set.transform(
            np.asarray(
                [unit_value]
            )
        )

        live_points.append(
            physical,
            float(
                np.log(
                    physical[0]
                )
            ),
        )

    result = WhitenedSliceProposalEngine(
        n_repeats=3
    ).propose(
        live_points=live_points,
        logl_min=-1.0,
        parameter_set=parameter_set,
        likelihood=lambda values: float(
            np.log(
                values[0]
            )
        ),
        rng=np.random.default_rng(23),
    )

    assert result.logl > -1.0
    assert (
        1.0e-2
        <= result.values[0]
        <= 1.0e2
    )


def test_whitened_slice_rejects_parameter_order_mismatch():
    """Proposal geometry and prior ordering must remain identical."""
    parameter_set = ParameterSet(
        [
            Parameter(
                "y",
                0.0,
                1.0,
            ),
            Parameter(
                "x",
                0.0,
                1.0,
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="parameter ordering",
    ):
        WhitenedSliceProposalEngine().propose(
            live_points=make_live_points(),
            logl_min=0.5,
            parameter_set=parameter_set,
            likelihood=lambda values: 1.0,
            rng=np.random.default_rng(1),
        )
