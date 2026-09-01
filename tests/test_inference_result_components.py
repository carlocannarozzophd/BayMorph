"""Tests for backend-neutral inference-result components."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.results import (
    EvidenceEstimate,
    PosteriorSamples,
)


def test_posterior_samples_normalize_weights():
    posterior = PosteriorSamples(
        samples=np.asarray(
            [
                [0.0, 1.0],
                [2.0, 3.0],
            ]
        ),
        weights=np.asarray(
            [
                2.0,
                6.0,
            ]
        ),
        parameter_names=[
            "x",
            "y",
        ],
    )

    assert posterior.weights == pytest.approx(
        np.asarray(
            [
                0.25,
                0.75,
            ]
        )
    )


def test_posterior_samples_reports_effective_sample_size():
    posterior = PosteriorSamples(
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

    assert posterior.effective_sample_size == pytest.approx(
        2.0
    )


def test_posterior_samples_preserves_sample_metadata():
    posterior = PosteriorSamples(
        samples=np.asarray(
            [
                [0.0],
                [1.0],
            ]
        ),
        weights=np.asarray(
            [
                0.4,
                0.6,
            ]
        ),
        parameter_names=[
            "x",
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
                -0.5,
            ]
        ),
    )

    assert posterior.log_likelihood == pytest.approx(
        np.asarray(
            [
                -2.0,
                -1.0,
            ]
        )
    )

    assert posterior.log_prior == pytest.approx(
        np.asarray(
            [
                -0.5,
                -0.5,
            ]
        )
    )

    assert posterior.best_parameters == {
        "x": 1.0,
    }


def test_posterior_samples_rejects_invalid_weight_shape():
    with pytest.raises(
        ValueError,
        match="one value per sample",
    ):
        PosteriorSamples(
            samples=np.zeros(
                (3, 1)
            ),
            weights=np.ones(
                2
            ),
            parameter_names=[
                "x",
            ],
        )


def test_posterior_samples_rejects_parameter_name_mismatch():
    with pytest.raises(
        ValueError,
        match="parameter names",
    ):
        PosteriorSamples(
            samples=np.zeros(
                (3, 2)
            ),
            weights=np.ones(
                3
            ),
            parameter_names=[
                "x",
            ],
        )


def test_evidence_estimate_preserves_method_semantics():
    evidence = EvidenceEstimate(
        logz=-12.5,
        uncertainty=0.15,
        method="dynesty",
        uncertainty_kind=(
            "nested_sampling_internal"
        ),
        diagnostics={
            "nlive": 500,
        },
        provenance={
            "backend": "dynesty",
        },
    )

    assert evidence.logz == pytest.approx(
        -12.5
    )
    assert evidence.uncertainty == pytest.approx(
        0.15
    )
    assert evidence.method == "dynesty"
    assert (
        evidence.uncertainty_kind
        == "nested_sampling_internal"
    )
    assert evidence.diagnostics == {
        "nlive": 500,
    }
    assert evidence.provenance == {
        "backend": "dynesty",
    }


def test_evidence_estimate_rejects_negative_uncertainty():
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        EvidenceEstimate(
            logz=-3.0,
            uncertainty=-0.1,
            method="test",
        )


def test_evidence_estimate_requires_method():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        EvidenceEstimate(
            logz=-3.0,
            method=" ",
        )


def test_equal_weight_samples_preserve_equal_weights():
    posterior = PosteriorSamples(
        samples=np.asarray(
            [
                [0.0],
                [1.0],
                [2.0],
                [3.0],
            ]
        ),
        weights=np.ones(4),
        parameter_names=[
            "x",
        ],
    )

    assert np.array_equal(
        posterior.equal_weight_samples,
        posterior.samples,
    )


def test_equal_weight_samples_are_deterministic():
    posterior = PosteriorSamples(
        samples=np.asarray(
            [
                [0.0],
                [1.0],
                [2.0],
            ]
        ),
        weights=np.asarray(
            [
                0.1,
                0.2,
                0.7,
            ]
        ),
        parameter_names=[
            "x",
        ],
    )

    first = posterior.equal_weight_samples
    second = posterior.equal_weight_samples

    assert np.array_equal(
        first,
        second,
    )


def test_equal_weight_samples_follow_weighted_mass():
    posterior = PosteriorSamples(
        samples=np.arange(
            10,
            dtype=float,
        ).reshape(
            10,
            1,
        ),
        weights=np.asarray(
            [
                0.6,
                0.3,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        parameter_names=[
            "x",
        ],
    )

    values = posterior.equal_weight_samples[
        :,
        0,
    ]

    assert np.count_nonzero(
        values == 0.0
    ) == 6
    assert np.count_nonzero(
        values == 1.0
    ) == 3
    assert np.count_nonzero(
        values == 2.0
    ) == 1

    assert not np.any(
        values >= 3.0
    )


def test_equal_weight_samples_exclude_zero_weight_samples():
    posterior = PosteriorSamples(
        samples=np.asarray(
            [
                [0.0],
                [1.0],
                [2.0],
                [3.0],
            ]
        ),
        weights=np.asarray(
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        ),
        parameter_names=[
            "x",
        ],
    )

    assert np.array_equal(
        posterior.equal_weight_samples,
        np.full(
            (
                4,
                1,
            ),
            3.0,
        ),
    )
