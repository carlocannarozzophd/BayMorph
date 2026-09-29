"""Tests for repeated-noise multi-observation campaigns."""

from __future__ import annotations

import pytest

from baymorph.validation import (
    MultiObservationGroundTruth,
    MultiObservationNoiseCampaign,
    ObservationTruth,
)


def _truth():
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
            "background_level__g": 0.0,
            "background_level__r": 0.0,
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
                metadata={
                    "rotation_deg": 0.0,
                    "crpix": (
                        31.0,
                        31.0,
                    ),
                    "raw_background": 100.0,
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
                metadata={
                    "rotation_deg": 32.0,
                    "crpix": (
                        38.0,
                        24.0,
                    ),
                    "raw_background": 100.0,
                },
            ),
        },
        random_seed=42,
    )


def _campaign(tmp_path):
    return MultiObservationNoiseCampaign(
        root=tmp_path,
        truth=_truth(),
        sampler={
            "name": "pocomc",
            "init": {
                "n_effective": 512,
                "n_active": 256,
                "sample": "tpcn",
                "random_state": 42,
            },
            "run": {
                "n_total": 4096,
                "n_evidence": 4096,
            },
        },
        n_cases=10,
    )


def test_campaign_changes_only_noise_seeds(
    tmp_path,
):
    campaign = _campaign(
        tmp_path
    )

    first = campaign._truth_for_case(
        0
    )
    second = campaign._truth_for_case(
        1
    )

    assert (
        first.parameters
        == second.parameters
    )

    assert (
        first.observations[
            "g"
        ].image_shape
        == second.observations[
            "g"
        ].image_shape
    )

    assert (
        first.observations[
            "g"
        ].metadata["seed"]
        != second.observations[
            "g"
        ].metadata["seed"]
    )

    assert (
        first.observations[
            "r"
        ].metadata["seed"]
        != second.observations[
            "r"
        ].metadata["seed"]
    )


def test_campaign_noise_seeds_are_distinct_between_observations(
    tmp_path,
):
    campaign = _campaign(
        tmp_path
    )

    truth = campaign._truth_for_case(
        3
    )

    assert (
        truth.observations[
            "g"
        ].metadata["seed"]
        != truth.observations[
            "r"
        ].metadata["seed"]
    )


def test_campaign_rejects_invalid_size(
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        MultiObservationNoiseCampaign(
            root=tmp_path,
            truth=_truth(),
            sampler={},
            n_cases=0,
        )


def test_campaign_resume_reuses_completed_case(
    tmp_path,
    monkeypatch,
):
    campaign = _campaign(
        tmp_path
    )

    expected = object()

    monkeypatch.setattr(
        MultiObservationNoiseCampaign,
        "_result_path",
        lambda self, index: (
            tmp_path
            / "existing.npz"
        ),
    )

    (
        tmp_path
        / "existing.npz"
    ).touch()

    monkeypatch.setattr(
        MultiObservationNoiseCampaign,
        "load_case",
        lambda self, index: expected,
    )

    def fail_run_case(
        self,
        index,
    ):
        raise AssertionError(
            "Completed case must not be rerun."
        )

    monkeypatch.setattr(
        MultiObservationNoiseCampaign,
        "run_case",
        fail_run_case,
    )

    object.__setattr__(
        campaign,
        "n_cases",
        1,
    )

    assert campaign.run() == [
        expected
    ]


def test_campaign_records_failure_metadata(
    tmp_path,
):
    campaign = _campaign(
        tmp_path
    )

    campaign._write_failure(
        index=0,
        stage="preparation",
        exc=ValueError(
            "No stable radial-profile plateau was detected."
        ),
    )

    path = (
        tmp_path
        / "noise-000"
        / "failure.yaml"
    )

    assert path.exists()

    import yaml

    with path.open() as stream:
        record = yaml.safe_load(
            stream
        )

    assert record == {
        "case_id": "noise-000",
        "status": "failed",
        "stage": "preparation",
        "exception_type": "ValueError",
        "message": (
            "No stable radial-profile plateau was detected."
        ),
    }
