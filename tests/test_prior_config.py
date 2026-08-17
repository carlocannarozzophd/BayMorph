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
