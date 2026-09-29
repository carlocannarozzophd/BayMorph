"""Tests for canonical and legacy inference-result persistence."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.results import (
    EvidenceEstimate,
    InferenceResult,
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


def test_emcee_result_preserves_mcmc_ess_without_evidence(
    tmp_path,
):
    """Posterior-only emcee metadata must survive serialization."""
    result = InferenceResult(
        posterior=PosteriorSamples(
            samples=np.asarray(
                [
                    [-0.2],
                    [0.0],
                    [0.1],
                    [0.3],
                ],
                dtype=float,
            ),
            weights=np.full(
                4,
                0.25,
                dtype=float,
            ),
            parameter_names=[
                "x",
            ],
            log_likelihood=np.asarray(
                [
                    -0.02,
                    0.0,
                    -0.005,
                    -0.045,
                ],
                dtype=float,
            ),
        ),
        evidence=None,
        diagnostics={
            "backend": "emcee",
            "converged": True,
            "mcmc_effective_sample_size": 2.5,
            "mcmc_effective_sample_size_by_parameter": {
                "x": 2.5,
            },
            "evidence_available": False,
        },
        provenance={
            "backend": "emcee",
            "inference_kind": "posterior_only",
            "sampler": {
                "name": "emcee",
                "init": {
                    "nwalkers": 12,
                    "seed": 42,
                },
                "run": {
                    "burn_in": 100,
                    "max_steps": 1000,
                },
                "execution": {
                    "requested_processes": 1,
                    "processes": 1,
                    "progress": False,
                },
            },
        },
    )

    path = result.save(
        tmp_path / "emcee_result.npz"
    )
    loaded = InferenceResult.load(
        path
    )

    assert loaded.evidence is None
    assert loaded.logz is None
    assert loaded.logz_error is None
    assert loaded.evidence_method is None

    assert loaded.samples == pytest.approx(
        result.samples
    )
    assert loaded.weights == pytest.approx(
        result.weights
    )
    assert loaded.log_likelihood == pytest.approx(
        result.log_likelihood
    )

    assert loaded.diagnostics == result.diagnostics
    assert loaded.provenance == result.provenance

    assert (
        loaded.posterior.effective_sample_size
        == pytest.approx(
            4.0
        )
    )
    assert (
        loaded.effective_sample_size
        == pytest.approx(
            2.5
        )
    )


def test_inference_result_rejects_legacy_posterior_archive(
    tmp_path,
):
    path = tmp_path / "legacy.npz"

    np.savez_compressed(
        path,
        samples=np.asarray(
            [
                [1.0],
                [2.0],
            ],
            dtype=float,
        ),
        logwt=np.log(
            np.asarray(
                [
                    0.4,
                    0.6,
                ],
                dtype=float,
            )
        ),
        logz=np.asarray(
            [
                0.0,
            ],
            dtype=float,
        ),
        parameter_names=np.asarray(
            [
                "x",
            ],
            dtype=str,
        ),
        logl=np.asarray(
            [
                -2.0,
                -1.0,
            ],
            dtype=float,
        ),
        logzerr=np.asarray(
            [
                0.3,
            ],
            dtype=float,
        ),
    )

    with pytest.raises(
        ValueError,
        match="format_version",
    ):
        InferenceResult.load(
            path
        )


def test_inference_result_serializes_runtime_sampler_objects(
    tmp_path,
):
    """Runtime sampler objects should have stable provenance metadata."""
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
        ),
        provenance={
            "backend": "dynesty",
            "sampler": {
                "name": "dynesty",
                "mode": "static",
                "init": {
                    "nlive": 500,
                    "rstate": np.random.default_rng(
                        42
                    ),
                },
                "run": {
                    "dlogz": 0.05,
                },
            },
        },
    )

    path = result.save(
        tmp_path / "result.npz"
    )

    loaded = InferenceResult.load(
        path
    )

    assert loaded.provenance[
        "sampler"
    ]["init"]["nlive"] == 500

    assert loaded.provenance[
        "sampler"
    ]["init"]["rstate"] == {
        "__runtime_object__": True,
        "type": "numpy.random.Generator",
        "bit_generator": "PCG64",
    }

    assert loaded.provenance[
        "sampler"
    ]["run"] == {
        "dlogz": 0.05,
    }
