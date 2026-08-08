"""Tests for the canonical backend-neutral inference result."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.results import (
    EvidenceEstimate,
    InferenceResult,
    PosteriorSamples,
)


def _make_posterior() -> PosteriorSamples:
    """Construct a small weighted posterior for result tests."""
    return PosteriorSamples(
        samples=np.asarray(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
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
        log_likelihood=np.asarray(
            [
                -5.0,
                -1.0,
                -3.0,
            ]
        ),
        log_prior=np.asarray(
            [
                -2.0,
                -2.0,
                -2.0,
            ]
        ),
    )


def test_inference_result_exposes_posterior_quantities():
    posterior = _make_posterior()

    result = InferenceResult(
        posterior=posterior
    )

    assert result.samples is posterior.samples
    assert result.weights is posterior.weights

    assert result.parameter_names == [
        "flux",
        "re",
    ]

    assert (
        result.log_likelihood
        is posterior.log_likelihood
    )

    assert result.log_prior is posterior.log_prior

    assert result.best_index == 1

    assert result.best_parameters == {
        "flux": 2.0,
        "re": 20.0,
    }


def test_inference_result_exposes_evidence_quantities():
    result = InferenceResult(
        posterior=_make_posterior(),
        evidence=EvidenceEstimate(
            logz=-12.5,
            uncertainty=0.08,
            method="pocomc",
            uncertainty_kind="importance_sampling_internal",
        ),
    )

    assert result.logz == pytest.approx(
        -12.5
    )

    assert result.logz_error == pytest.approx(
        0.08
    )

    assert result.evidence_method == "pocomc"


def test_inference_result_allows_missing_evidence():
    result = InferenceResult(
        posterior=_make_posterior()
    )

    assert result.evidence is None
    assert result.logz is None
    assert result.logz_error is None
    assert result.evidence_method is None


def test_inference_result_preserves_diagnostics_and_provenance():
    result = InferenceResult(
        posterior=_make_posterior(),
        diagnostics={
            "calls": 1234,
        },
        provenance={
            "backend": "pincopallo",
            "version": "1.0",
        },
    )

    assert result.diagnostics == {
        "calls": 1234,
    }

    assert result.provenance == {
        "backend": "pincopallo",
        "version": "1.0",
    }


def test_inference_result_rejects_nonposterior_input():
    with pytest.raises(
        TypeError,
        match="PosteriorSamples",
    ):
        InferenceResult(
            posterior=object(),
        )


def test_inference_result_rejects_invalid_evidence():
    with pytest.raises(
        TypeError,
        match="EvidenceEstimate",
    ):
        InferenceResult(
            posterior=_make_posterior(),
            evidence=object(),
        )


def test_inference_result_reports_effective_sample_size():
    result = InferenceResult(
        posterior=PosteriorSamples(
            samples=np.asarray(
                [
                    [0.0],
                    [1.0],
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
        )
    )

    assert (
        result.effective_sample_size
        == pytest.approx(2.0)
    )
