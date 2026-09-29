"""Tests for typed priors and fixed parameters."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.fitting import (
    FixedPrior,
    NormalPrior,
    Parameter,
    ParameterSet,
    UniformPrior,
)


def test_legacy_parameter_constructor_creates_uniform_prior():
    parameter = Parameter(
        "x",
        1.0,
        3.0,
    )

    assert isinstance(
        parameter.prior,
        UniformPrior,
    )
    assert parameter.transform(0.25) == pytest.approx(1.5)
    assert parameter.lower == pytest.approx(1.0)
    assert parameter.upper == pytest.approx(3.0)


def test_fixed_parameter_does_not_contribute_sampled_dimension():
    parameters = ParameterSet(
        [
            Parameter(
                "flux",
                prior=UniformPrior(
                    lower=1.0,
                    upper=3.0,
                ),
            ),
            Parameter(
                "x0",
                prior=FixedPrior(
                    value=15.5,
                ),
            ),
        ]
    )

    assert parameters.ndim == 1
    assert parameters.names == ["flux"]
    assert parameters.all_names == [
        "flux",
        "x0",
    ]
    assert parameters.fixed_values == {
        "x0": 15.5,
    }

    transformed = parameters.transform(
        np.array([0.25])
    )

    assert transformed == pytest.approx(
        np.array([1.5])
    )


def test_normal_prior_maps_median_to_mean():
    prior = NormalPrior(
        mean=4.0,
        sigma=2.0,
    )

    assert prior.transform(0.5) == pytest.approx(4.0)


def test_normal_prior_produces_finite_endpoint_values():
    prior = NormalPrior(
        mean=0.0,
        sigma=1.0,
    )

    assert np.isfinite(prior.transform(0.0))
    assert np.isfinite(prior.transform(1.0))


def test_uniform_prior_rejects_invalid_bounds():
    with pytest.raises(
        ValueError,
        match="greater than",
    ):
        UniformPrior(
            lower=1.0,
            upper=1.0,
        )


def test_normal_prior_rejects_non_positive_sigma():
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        NormalPrior(
            mean=0.0,
            sigma=0.0,
        )


def test_parameter_rejects_mixed_prior_interfaces():
    with pytest.raises(
        ValueError,
        match="cannot be combined",
    ):
        Parameter(
            "x",
            0.0,
            1.0,
            prior=UniformPrior(
                lower=0.0,
                upper=1.0,
            ),
        )


def test_beta_prior_maps_symmetric_median_to_interval_midpoint():
    from baymorph.fitting import BetaPrior

    prior = BetaPrior(
        alpha=2.0,
        beta=2.0,
        lower=2.0,
        upper=8.0,
    )

    assert prior.transform(
        0.5
    ) == pytest.approx(
        5.0
    )


def test_beta_prior_respects_scaled_endpoints():
    from baymorph.fitting import BetaPrior

    prior = BetaPrior(
        alpha=2.0,
        beta=3.0,
        lower=0.2,
        upper=1.0,
    )

    assert prior.transform(
        0.0
    ) == pytest.approx(
        0.2
    )
    assert prior.transform(
        1.0
    ) == pytest.approx(
        1.0
    )


@pytest.mark.parametrize(
    (
        "alpha",
        "beta_parameter",
    ),
    [
        (
            0.0,
            2.0,
        ),
        (
            -1.0,
            2.0,
        ),
        (
            2.0,
            0.0,
        ),
        (
            2.0,
            -1.0,
        ),
    ],
)
def test_beta_prior_rejects_non_positive_shapes(
    alpha,
    beta_parameter,
):
    from baymorph.fitting import BetaPrior

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        BetaPrior(
            alpha=alpha,
            beta=beta_parameter,
        )


def test_log_uniform_prior_maps_median_to_geometric_mean():
    """The unit-cube midpoint must map to the geometric mean."""
    from baymorph.fitting import LogUniformPrior

    prior = LogUniformPrior(
        lower=1.0e2,
        upper=1.0e8,
    )

    assert prior.transform(
        0.5
    ) == pytest.approx(
        1.0e5
    )


def test_log_uniform_prior_respects_endpoints():
    """Finite log-uniform bounds must map exactly from unit-cube endpoints."""
    from baymorph.fitting import LogUniformPrior

    prior = LogUniformPrior(
        lower=2.0,
        upper=200.0,
    )

    assert prior.transform(
        0.0
    ) == pytest.approx(
        2.0
    )

    assert prior.transform(
        1.0
    ) == pytest.approx(
        200.0
    )


@pytest.mark.parametrize(
    (
        "lower",
        "upper",
    ),
    [
        (
            0.0,
            10.0,
        ),
        (
            -1.0,
            10.0,
        ),
        (
            10.0,
            10.0,
        ),
        (
            10.0,
            1.0,
        ),
    ],
)
def test_log_uniform_prior_rejects_invalid_bounds(
    lower,
    upper,
):
    """Log-uniform priors require ordered strictly positive bounds."""
    from baymorph.fitting import LogUniformPrior

    with pytest.raises(
        ValueError
    ):
        LogUniformPrior(
            lower=lower,
            upper=upper,
        )
