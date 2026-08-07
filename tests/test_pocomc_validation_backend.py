"""Tests for the pocoMC scientific-validation backend."""

from __future__ import annotations

import numpy as np
import pytest

from validation import ValidationExperiment
from validation.backends import PocoMCValidationBackend
from validation.problems.mathematical import (
    make_gaussian_1d_problem,
)


pytest.importorskip(
    "pocomc"
)


def test_pocomc_backend_name():
    backend = PocoMCValidationBackend()

    assert backend.name == "pocomc"


def test_pocomc_gaussian_1d_smoke_run():
    problem = make_gaussian_1d_problem()
    backend = PocoMCValidationBackend()

    experiment = ValidationExperiment(
        problem=problem,
        backend=backend,
        seed=13,
        config={
            "n_effective": 64,
            "n_active": 32,
            "n_total": 128,
            "n_evidence": 128,
            "pytorch_threads": 1,
            "print_progress": False,
        },
    )

    result = experiment.run()

    assert result.problem_name == "gaussian_1d"
    assert result.backend_name == "pocomc"

    assert result.samples.ndim == 2
    assert result.samples.shape[1] == 1

    assert result.weights.shape == (
        result.samples.shape[0],
    )
    assert result.logl.shape == (
        result.samples.shape[0],
    )

    assert np.sum(
        result.weights
    ) == pytest.approx(
        1.0
    )

    assert np.isfinite(
        result.logz
    )
    assert result.logzerr is not None
    assert np.isfinite(
        result.logzerr
    )
    assert result.logzerr >= 0.0

    assert result.ncall > 0
    assert result.runtime_seconds > 0.0

    assert result.metadata[
        "evidence_estimator"
    ] == "importance_sampling"


def test_pocomc_gaussian_1d_recovers_reference():
    problem = make_gaussian_1d_problem()
    backend = PocoMCValidationBackend()

    result = ValidationExperiment(
        problem=problem,
        backend=backend,
        seed=42,
        config={
            "n_effective": 256,
            "n_active": 128,
            "n_total": 1024,
            "n_evidence": 1024,
            "pytorch_threads": 1,
            "print_progress": False,
        },
    ).run()

    posterior_sigma = np.sqrt(
        problem.reference_covariance[
            0,
            0,
        ]
    )

    mean_error_sigma = abs(
        result.posterior_mean[0]
        - problem.reference_mean[0]
    ) / posterior_sigma

    logz_error = abs(
        result.logz
        - problem.reference_logz
    )

    assert mean_error_sigma < 0.15
    assert logz_error < 0.15
