"""Tests for heterogeneous multi-observation validation truth."""

from __future__ import annotations

import pytest

from baymorph.validation import (
    MultiObservationGroundTruth,
    ObservationTruth,
)


def make_truth() -> MultiObservationGroundTruth:
    return MultiObservationGroundTruth(
        model_name="single_sersic",
        parameters={
            "flux__g": 900.0,
            "flux__r": 1200.0,
            "re_arcsec": 1.2,
            "n": 2.0,
            "q": 0.65,
            "position_angle": 0.35,
            "ra": 150.0,
            "dec": 30.0,
            "sky__g": 0.0,
            "sky__r": 0.0,
        },
        observations={
            "g": ObservationTruth(
                image_shape=(61, 61),
                pixel_scale=0.20,
                psf_metadata={
                    "fwhm_pixels": 2.0,
                },
                noise_metadata={
                    "sigma": 8.0,
                },
            ),
            "r": ObservationTruth(
                image_shape=(47, 73),
                pixel_scale=0.35,
                psf_metadata={
                    "fwhm_pixels": 2.8,
                },
                noise_metadata={
                    "sigma": 11.0,
                },
            ),
        },
        random_seed=42,
    )


def test_multi_observation_truth_describes_one_joint_state():
    truth = make_truth()

    assert truth.model_name == "single_sersic"
    assert truth.parameter(
        "re_arcsec"
    ) == pytest.approx(
        1.2
    )

    assert tuple(
        truth.observations
    ) == (
        "g",
        "r",
    )

    assert truth.observations[
        "g"
    ].image_shape == (
        61,
        61,
    )

    assert truth.observations[
        "r"
    ].pixel_scale == pytest.approx(
        0.35
    )


def test_multi_observation_truth_is_immutable():
    truth = make_truth()

    with pytest.raises(
        TypeError
    ):
        truth.parameters[
            "re_arcsec"
        ] = 2.0

    with pytest.raises(
        TypeError
    ):
        truth.observations[
            "g"
        ] = truth.observations[
            "r"
        ]


def test_observation_truth_rejects_invalid_pixel_scale():
    with pytest.raises(
        ValueError,
        match="pixel scale",
    ):
        ObservationTruth(
            image_shape=(31, 31),
            pixel_scale=0.0,
        )


def test_multi_observation_truth_requires_observations():
    with pytest.raises(
        ValueError,
        match="at least one observation",
    ):
        MultiObservationGroundTruth(
            model_name="single_sersic",
            parameters={
                "re_arcsec": 1.2,
            },
            observations={},
        )


def test_recovery_experiment_accepts_multi_observation_truth():
    from baymorph.validation import (
        RecoveryExperiment,
        RecoveryFit,
    )

    truth = make_truth()

    experiment = RecoveryExperiment(
        case_id="multi-observation-smoke",
        truth=truth,
        simulator=lambda input_truth: {
            "truth": input_truth,
        },
        fitter=lambda observation: RecoveryFit(
            estimates=dict(
                observation["truth"].parameters
            )
        ),
    )

    result = experiment.run()

    assert result.truth == dict(
        truth.parameters
    )
    assert result.estimates == dict(
        truth.parameters
    )
    assert result.metadata[
        "model_name"
    ] == "single_sersic"
