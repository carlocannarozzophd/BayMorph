"""Tests for canonical inference-result validation adaptation."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.results import InferenceResult
from baymorph.results.evidence import EvidenceEstimate
from baymorph.results.samples import PosteriorSamples
from baymorph.validation.baymorph_adapter import (
    recovery_fit_from_inference_result,
)


def make_result() -> InferenceResult:
    posterior = PosteriorSamples(
        samples=np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ]
        ),
        weights=np.array(
            [
                0.2,
                0.6,
                0.2,
            ]
        ),
        parameter_names=[
            "re_arcsec",
            "flux__g",
        ],
        log_likelihood=np.array(
            [
                -10.0,
                -5.0,
                -8.0,
            ]
        ),
    )

    return InferenceResult(
        posterior=posterior,
        evidence=EvidenceEstimate(
            logz=-123.0,
            uncertainty=0.4,
            method="test",
        ),
        diagnostics={
            "ncall": 1250,
        },
        provenance={
            "backend": "test_backend",
        },
    )


def test_inference_result_adapter_uses_posterior_summaries():
    result = recovery_fit_from_inference_result(
        make_result(),
        metadata={
            "campaign": "joint-recovery",
        },
    )

    assert set(
        result.estimates
    ) == {
        "re_arcsec",
        "flux__g",
    }

    assert result.estimates[
        "re_arcsec"
    ] == pytest.approx(
        2.0
    )

    assert result.intervals is not None
    assert result.intervals[
        "re_arcsec"
    ][0] <= result.estimates[
        "re_arcsec"
    ]
    assert result.intervals[
        "re_arcsec"
    ][1] >= result.estimates[
        "re_arcsec"
    ]

    assert result.log_evidence == pytest.approx(
        -123.0
    )
    assert result.log_evidence_error == pytest.approx(
        0.4
    )
    assert result.likelihood_evaluations == 1250

    assert result.metadata[
        "adapter"
    ] == "baymorph_inference_result"
    assert result.metadata[
        "backend"
    ] == "test_backend"
    assert result.metadata[
        "campaign"
    ] == "joint-recovery"


def test_inference_result_adapter_allows_missing_evidence():
    inference = make_result()
    inference.evidence = None

    result = recovery_fit_from_inference_result(
        inference
    )

    assert result.log_evidence is None
    assert result.log_evidence_error is None


def test_inference_result_adapter_requires_canonical_result():
    with pytest.raises(
        TypeError,
        match="InferenceResult",
    ):
        recovery_fit_from_inference_result(
            object()
        )
