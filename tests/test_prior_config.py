"""Tests for shared typed-prior configuration parsing."""

from __future__ import annotations

import pytest

from baymorph.fitting.priors import (
    BetaPrior,
    FixedPrior,
    NormalPrior,
    UniformPrior,
)
from baymorph.pipeline.prior_config import parse_prior_config


def test_parse_supported_prior_types() -> None:
    """The shared parser must construct every supported prior type."""
    assert parse_prior_config(
        {
            "type": "fixed",
            "value": 2.5,
        }
    ) == FixedPrior(
        value=2.5,
    )

    assert parse_prior_config(
        {
            "type": "uniform",
            "lower": 0.1,
            "upper": 8.0,
        }
    ) == UniformPrior(
        lower=0.1,
        upper=8.0,
    )

    assert parse_prior_config(
        {
            "type": "normal",
            "mean": 0.0,
            "sigma": 2.0,
        }
    ) == NormalPrior(
        mean=0.0,
        sigma=2.0,
    )

    assert parse_prior_config(
        {
            "type": "beta",
            "alpha": 2.0,
            "beta": 5.0,
            "lower": 0.2,
            "upper": 1.0,
        }
    ) == BetaPrior(
        alpha=2.0,
        beta=5.0,
        lower=0.2,
        upper=1.0,
    )


def test_parse_relative_uniform_prior() -> None:
    """Relative uniform priors must resolve around explicit references."""
    prior = parse_prior_config(
        {
            "type": "uniform",
            "lower": -8.0,
            "upper": 8.0,
            "relative_to": "target_x",
        },
        references={
            "target_x": 174.5,
        },
    )

    assert prior == UniformPrior(
        lower=166.5,
        upper=182.5,
    )


def test_relative_prior_does_not_change_reference_mapping() -> None:
    """Resolving a relative prior must not mutate its reference context."""
    references = {
        "target_x": 100.0,
    }

    parse_prior_config(
        {
            "type": "uniform",
            "lower": -5.0,
            "upper": 5.0,
            "relative_to": "target_x",
        },
        references=references,
    )

    assert references == {
        "target_x": 100.0,
    }


def test_relative_prior_requires_references() -> None:
    """Relative uniform priors must fail without an explicit context."""
    with pytest.raises(
        ValueError,
        match="requires reference values",
    ):
        parse_prior_config(
            {
                "type": "uniform",
                "lower": -8.0,
                "upper": 8.0,
                "relative_to": "target_x",
            }
        )


def test_relative_prior_rejects_unknown_reference() -> None:
    """Relative priors must not silently resolve unknown reference names."""
    with pytest.raises(
        ValueError,
        match="Unknown prior reference",
    ):
        parse_prior_config(
            {
                "type": "uniform",
                "lower": -8.0,
                "upper": 8.0,
                "relative_to": "target_x",
            },
            references={
                "target_y": 42.0,
            },
        )


def test_parse_prior_requires_mapping() -> None:
    """Prior configuration must be mapping-like."""
    with pytest.raises(
        TypeError,
        match="must be a mapping",
    ):
        parse_prior_config(
            ["uniform"]
        )


def test_parse_prior_rejects_incomplete_scaled_beta() -> None:
    """Scaled beta priors must define both finite bounds."""
    with pytest.raises(
        ValueError,
        match="Missing",
    ):
        parse_prior_config(
            {
                "type": "beta",
                "alpha": 2.0,
                "beta": 2.0,
                "lower": 0.0,
            }
        )


def test_parse_prior_rejects_extra_keys() -> None:
    """Typed prior mappings must not silently accept unknown keys."""
    with pytest.raises(
        ValueError,
        match="Extra",
    ):
        parse_prior_config(
            {
                "type": "uniform",
                "lower": 0.0,
                "upper": 1.0,
                "unexpected": 42,
            }
        )


def test_parse_symbolic_uniform_bounds() -> None:
    """Typed priors must support canonical symbolic angular bounds."""
    prior = parse_prior_config(
        {
            "type": "uniform",
            "lower": "-pi/2",
            "upper": "pi/2",
        }
    )

    assert prior == UniformPrior(
        lower=-1.5707963267948966,
        upper=1.5707963267948966,
    )


def test_parse_log_uniform_prior() -> None:
    """The shared parser must construct a log-uniform prior."""
    from baymorph.fitting.priors import LogUniformPrior

    prior = parse_prior_config(
        {
            "type": "log_uniform",
            "lower": 1.0e2,
            "upper": 1.0e8,
        }
    )

    assert prior == LogUniformPrior(
        lower=1.0e2,
        upper=1.0e8,
    )


def test_log_uniform_prior_rejects_extra_configuration_keys() -> None:
    """Log-uniform prior configuration must use an exact key set."""
    with pytest.raises(
        ValueError,
        match="Extra",
    ):
        parse_prior_config(
            {
                "type": "log_uniform",
                "lower": 1.0,
                "upper": 100.0,
                "unexpected": 42,
            }
        )
