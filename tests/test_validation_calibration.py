"""Tests for posterior calibration diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.results import InferenceResult, PosteriorSamples
from baymorph.validation.calibration import (
    calibration_from_inference_result,
    summarize_calibration,
)


def _result() -> InferenceResult:
    samples = np.array(
        [
            [0.0, 10.0],
            [1.0, 20.0],
            [2.0, 30.0],
            [3.0, 40.0],
        ]
    )
    weights = np.full(4, 0.25)

    return InferenceResult(
        posterior=PosteriorSamples(
            samples=samples,
            weights=weights,
            parameter_names=["x", "y"],
        ),
        diagnostics={},
        provenance={"backend": "test"},
    )


def test_calibration_from_inference_result():
    case = calibration_from_inference_result(
        case_id="case-000",
        result=_result(),
        truth={
            "x": 1.5,
            "y": 25.0,
        },
    )

    assert case.case_id == "case-000"

    x = case.parameters["x"]

    assert x.truth == 1.5
    assert x.posterior_mean == pytest.approx(1.5)
    assert x.posterior_rank == pytest.approx(0.5)
    assert x.mean_error == pytest.approx(0.0)


def test_calibration_reports_credible_interval_coverage():
    case = calibration_from_inference_result(
        case_id="case-000",
        result=_result(),
        truth={
            "x": 1.5,
            "y": 25.0,
        },
    )

    assert case.parameters["x"].covered_68
    assert case.parameters["x"].covered_95
    assert case.parameters["y"].covered_68
    assert case.parameters["y"].covered_95


def test_calibration_requires_truth_parameters_in_posterior():
    with pytest.raises(
        ValueError,
        match="missing truth parameters",
    ):
        calibration_from_inference_result(
            case_id="case-000",
            result=_result(),
            truth={
                "missing": 1.0,
            },
        )


def test_summarize_calibration():
    first = calibration_from_inference_result(
        case_id="case-000",
        result=_result(),
        truth={
            "x": 1.5,
            "y": 25.0,
        },
    )
    second = calibration_from_inference_result(
        case_id="case-001",
        result=_result(),
        truth={
            "x": 1.5,
            "y": 25.0,
        },
    )

    summary = summarize_calibration(
        [first, second]
    )

    assert summary["x"]["count"] == 2
    assert summary["x"]["mean_error_mean"] == pytest.approx(0.0)
    assert summary["x"]["coverage_68"] == pytest.approx(1.0)
    assert summary["x"]["coverage_95"] == pytest.approx(1.0)
    assert summary["x"]["rank_mean"] == pytest.approx(0.5)
    assert summary["x"]["rank_min"] == pytest.approx(0.5)
    assert summary["x"]["rank_max"] == pytest.approx(0.5)


def test_summarize_calibration_requires_cases():
    with pytest.raises(
        ValueError,
        match="At least one calibration case",
    ):
        summarize_calibration([])
