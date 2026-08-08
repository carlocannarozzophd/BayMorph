"""Tests for canonical and legacy inference-result persistence."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.results import (
    EvidenceEstimate,
    InferenceResult,
    PosteriorResult,
    PosteriorSamples,
)


def test_inference_result_save_load_round_trip(
    tmp_path,
):
    result = InferenceResult(
        posterior=PosteriorSamples(
            samples=np.asarray(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ),
            weights=np.asarray(
                [
                    0.25,
                    0.75,
                ]
            ),
            parameter_names=[
                "x",
                "y",
            ],
            log_likelihood=np.asarray(
                [
                    -2.0,
                    -1.0,
                ]
            ),
            log_prior=np.asarray(
                [
                    -0.5,
                    -0.4,
                ]
            ),
        ),
        evidence=EvidenceEstimate(
            logz=-10.5,
            uncertainty=0.2,
            method="pocomc",
            uncertainty_kind="importance_sampling_internal",
            diagnostics={
                "n_evidence": 4096,
            },
            provenance={
                "backend": "pocomc",
            },
        ),
        diagnostics={
            "calls": 1234,
        },
        provenance={
            "backend": "pocomc",
            "seed": 42,
        },
    )

    path = result.save(
        tmp_path / "result.npz"
    )

    loaded = InferenceResult.load(
        path
    )

    assert np.array_equal(
        loaded.samples,
        result.samples,
    )

    assert loaded.weights == pytest.approx(
        result.weights
    )

    assert loaded.parameter_names == [
        "x",
        "y",
    ]

    assert loaded.log_likelihood == pytest.approx(
        result.log_likelihood
    )

    assert loaded.log_prior == pytest.approx(
        result.log_prior
    )

    assert loaded.logz == pytest.approx(
        -10.5
    )

    assert loaded.logz_error == pytest.approx(
        0.2
    )

    assert loaded.evidence_method == "pocomc"

    assert loaded.diagnostics == {
        "calls": 1234,
    }

    assert loaded.provenance == {
        "backend": "pocomc",
        "seed": 42,
    }


def test_inference_result_save_load_without_evidence(
    tmp_path,
):
    result = InferenceResult(
        posterior=PosteriorSamples(
            samples=np.asarray(
                [
                    [1.0],
                    [2.0],
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

    loaded = InferenceResult.load(
        result.save(
            tmp_path / "result.npz"
        )
    )

    assert loaded.evidence is None
    assert loaded.logz is None
    assert loaded.logz_error is None


def test_inference_result_loads_legacy_posterior_archive(
    tmp_path,
):
    legacy = PosteriorResult(
        samples=np.asarray(
            [
                [1.0],
                [2.0],
            ]
        ),
        logwt=np.log(
            np.asarray(
                [
                    0.4,
                    0.6,
                ]
            )
        ),
        logz=np.asarray(
            [
                0.0,
            ]
        ),
        parameter_names=[
            "x",
        ],
        logl=np.asarray(
            [
                -2.0,
                -1.0,
            ]
        ),
        logzerr=np.asarray(
            [
                0.3,
            ]
        ),
    )

    path = legacy.save(
        tmp_path / "legacy.npz"
    )

    loaded = InferenceResult.load(
        path
    )

    assert loaded.weights == pytest.approx(
        np.asarray(
            [
                0.4,
                0.6,
            ]
        )
    )

    assert loaded.logz == pytest.approx(
        0.0
    )

    assert loaded.logz_error == pytest.approx(
        0.3
    )

    assert (
        loaded.evidence_method
        == "legacy_baymorph"
    )

    assert loaded.provenance == {
        "source_format": "posterior_result_v1",
    }
