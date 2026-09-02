"""Tests for scientific validation comparisons."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.results import (
    EvidenceEstimate,
    InferenceResult,
    PosteriorSamples,
)
from baymorph.validation.run import ValidationRun
from baymorph.validation.benchmark import compare_results


def make_run(
    *,
    backend_name: str,
    logz: float,
    ncall: int,
    runtime_seconds: float,
    offset: float = 0.0,
) -> ValidationRun:
    """Create a small canonical validation run."""
    inference = InferenceResult(
        posterior=PosteriorSamples(
            samples=np.asarray(
                [
                    [
                        -1.0 + offset,
                    ],
                    [
                        1.0 + offset,
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
                "x",
            ],
            log_likelihood=np.asarray(
                [
                    -0.5,
                    -0.5,
                ]
            ),
        ),
        evidence=EvidenceEstimate(
            logz=logz,
            uncertainty=0.1,
            method="test",
        ),
        diagnostics={
            "backend": backend_name,
            "calls": ncall,
        },
        provenance={
            "backend": backend_name,
        },
    )

    return ValidationRun(
        case_name="quadratic",
        backend_name=backend_name,
        result=inference,
        seed=42,
        config={},
        runtime_seconds=runtime_seconds,
    )


def test_compare_results_reports_backend_differences():
    reference = make_run(
        backend_name="dynesty",
        logz=-1.1,
        ncall=200,
        runtime_seconds=4.0,
    )
    candidate = make_run(
        backend_name="nautilus",
        logz=-1.0,
        ncall=100,
        runtime_seconds=2.0,
        offset=0.2,
    )

    report = compare_results(
        reference,
        candidate,
    )

    assert report.problem_name == "quadratic"
    assert report.reference_backend == "dynesty"
    assert report.candidate_backend == "nautilus"
    assert report.logz_difference == pytest.approx(
        0.1
    )
    assert report.posterior_mean_distance == pytest.approx(
        0.2
    )
    assert report.likelihood_call_ratio == pytest.approx(
        0.5
    )
    assert report.runtime_ratio == pytest.approx(
        0.5
    )


def test_compare_results_rejects_different_problems():
    reference = make_run(
        backend_name="dynesty",
        logz=-1.0,
        ncall=100,
        runtime_seconds=1.0,
    )
    candidate = make_run(
        backend_name="nautilus",
        logz=-1.0,
        ncall=100,
        runtime_seconds=1.0,
    )

    object.__setattr__(
        candidate,
        "case_name",
        "different",
    )

    with pytest.raises(
        ValueError,
        match="same problem",
    ):
        compare_results(
            reference,
            candidate,
        )
