"""Tests for weighted posterior contact with declared prior boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.results.prior_boundaries import (
    audit_posterior_prior_boundaries,
)
from baymorph.results.samples import PosteriorSamples


def test_flux_contact_uses_logarithmic_prior_measure() -> None:
    """A weighted posterior at a log-prior upper edge must be flagged."""
    posterior = PosteriorSamples(
        samples=np.asarray(
            [
                [
                    5_830_043.0,
                    40.0,
                ],
                [
                    5_830_196.0,
                    44.0,
                ],
                [
                    5_830_214.0,
                    48.0,
                ],
            ]
        ),
        weights=np.asarray(
            [
                0.2,
                0.3,
                0.5,
            ]
        ),
        parameter_names=[
            "flux",
            "re",
        ],
    )

    diagnostics = audit_posterior_prior_boundaries(
        posterior,
        {
            "flux": {
                "type": "log_uniform",
                "lower": 12_510.587338311978,
                "upper": 5_830_215.232335205,
            },
            "re": {
                "type": "uniform",
                "lower": 1.0,
                "upper": 120.0,
            },
        },
    )

    assert diagnostics["status"] == "boundary_contact"
    assert diagnostics["flagged_parameters"] == [
        "flux"
    ]
    assert diagnostics[
        "evidence_prior_sensitivity_warning"
    ]

    flux_diagnostics = diagnostics[
        "parameters"
    ][
        "flux"
    ]

    assert flux_diagnostics[
        "upper_edge_mass"
    ] == pytest.approx(
        1.0
    )
    assert flux_diagnostics[
        "boundary_contact"
    ]


def test_interior_posterior_is_clear() -> None:
    """Interior posterior mass must not produce an edge warning."""
    posterior = PosteriorSamples(
        samples=np.asarray(
            [
                [
                    8.0,
                ],
                [
                    10.0,
                ],
                [
                    12.0,
                ],
            ]
        ),
        weights=np.asarray(
            [
                0.25,
                0.5,
                0.25,
            ]
        ),
        parameter_names=[
            "flux",
        ],
    )

    diagnostics = audit_posterior_prior_boundaries(
        posterior,
        {
            "flux": {
                "type": "log_uniform",
                "lower": 1.0,
                "upper": 100.0,
            },
        },
    )

    assert diagnostics["status"] == "clear"
    assert diagnostics["flagged_parameters"] == []
    assert not diagnostics[
        "evidence_prior_sensitivity_warning"
    ]


def test_background_level_uses_sky_prior_alias() -> None:
    """Persisted background names must resolve the configured sky prior."""
    posterior = PosteriorSamples(
        samples=np.asarray(
            [
                [
                    49.99,
                ],
                [
                    50.0,
                ],
            ]
        ),
        weights=np.asarray(
            [
                0.4,
                0.6,
            ]
        ),
        parameter_names=[
            "background_level",
        ],
    )

    diagnostics = audit_posterior_prior_boundaries(
        posterior,
        {
            "sky": {
                "type": "uniform",
                "lower": -50.0,
                "upper": 50.0,
            },
        },
    )

    background = diagnostics[
        "parameters"
    ][
        "background_level"
    ]

    assert background["prior_name"] == "sky"
    assert background["boundary_contact"]


def test_unresolved_relative_prior_is_not_compared_to_absolute_samples() -> None:
    """Relative offsets must not be audited as absolute coordinates."""
    posterior = PosteriorSamples(
        samples=np.asarray(
            [
                [
                    345.4,
                ],
                [
                    345.5,
                ],
            ]
        ),
        weights=np.asarray(
            [
                0.5,
                0.5,
            ]
        ),
        parameter_names=[
            "x0",
        ],
    )

    diagnostics = audit_posterior_prior_boundaries(
        posterior,
        {
            "x0": {
                "type": "uniform",
                "lower": -8.0,
                "upper": 8.0,
                "relative_to": "target_x",
            },
        },
    )

    assert diagnostics["status"] == "not_evaluated"
    assert diagnostics["flagged_parameters"] == []
    assert diagnostics["parameters"] == {}
