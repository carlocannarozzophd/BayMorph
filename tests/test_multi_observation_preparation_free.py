"""Tests for preparation-free multi-observation recovery."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from baymorph.results import InferenceResult, PosteriorSamples
from baymorph.validation import (
    MultiObservationGroundTruth,
    MultiObservationNoiseCampaign,
    ObservationTruth,
    PreparationFreeMultiObservationNoiseCampaign,
    build_preparation_free_multi_observation_case,
)
from baymorph.validation import multi_observation_preparation_free


def _truth() -> MultiObservationGroundTruth:
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
                psf_metadata={"fwhm_pixels": 2.0},
                noise_metadata={"sigma": 8.0},
                metadata={
                    "rotation_deg": 0.0,
                    "crpix": (31.0, 31.0),
                    "seed": 11,
                    "raw_background": 100.0,
                },
            ),
            "r": ObservationTruth(
                image_shape=(47, 73),
                pixel_scale=0.35,
                psf_metadata={"fwhm_pixels": 2.8},
                noise_metadata={"sigma": 11.0},
                metadata={
                    "rotation_deg": 32.0,
                    "crpix": (38.0, 24.0),
                    "seed": 22,
                    "raw_background": 100.0,
                },
            ),
        },
        random_seed=42,
    )


def _campaign(tmp_path) -> PreparationFreeMultiObservationNoiseCampaign:
    return PreparationFreeMultiObservationNoiseCampaign(
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
        n_cases=3,
    )


def test_preparation_free_case_uses_native_grids_and_known_noise():
    truth = _truth()
    case = build_preparation_free_multi_observation_case(truth)

    assert case.problem.parameter_layout.observations == ("g", "r")
    assert case.problem.likelihood.image_shapes == {
        "g": (61, 61),
        "r": (47, 73),
    }

    for observation, sigma in {"g": 8.0, "r": 11.0}.items():
        shape = truth.observations[observation].image_shape
        assert case.noiseless_images[observation].shape == shape
        assert case.observed_images[observation].shape == shape
        assert case.noise_maps[observation].shape == shape
        assert np.all(case.noise_maps[observation] == sigma)

        likelihood = case.problem.likelihood._likelihoods[observation]
        assert likelihood.mask is None
        assert likelihood.n_valid_pixels == int(np.prod(shape))


def test_preparation_free_data_are_exact_model_plus_seeded_noise():
    truth = _truth()
    case = build_preparation_free_multi_observation_case(truth)

    for observation in case.problem.parameter_layout.observations:
        sigma = truth.observations[observation].noise_metadata["sigma"]
        shape = truth.observations[observation].image_shape
        expected_noise = np.random.default_rng(
            case.noise_seeds[observation]
        ).normal(loc=0.0, scale=sigma, size=shape)
        actual_noise = (
            case.observed_images[observation]
            - case.noiseless_images[observation]
        )
        np.testing.assert_allclose(
            actual_noise,
            expected_noise,
            rtol=0.0,
            atol=1.0e-10,
        )


def test_preparation_free_problem_closes_at_injected_truth():
    case = build_preparation_free_multi_observation_case(_truth())
    rendered = case.problem.render_models(case.truth_vector)

    for observation in case.problem.parameter_layout.observations:
        np.testing.assert_allclose(
            rendered[observation],
            case.noiseless_images[observation],
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    assert np.isfinite(case.problem.log_likelihood(case.truth_vector))


def test_preparation_free_case_preserves_end_to_end_prior_contract():
    case = build_preparation_free_multi_observation_case(_truth())

    assert case.problem.parameter_names == [
        "flux__g",
        "flux__r",
        "re_arcsec",
        "n",
        "q",
        "position_angle",
        "ra",
        "dec",
        "sky__g",
        "sky__r",
    ]
    expected_bounds = {
        "flux__g": (450.0, 1350.0),
        "flux__r": (600.0, 1800.0),
        "re_arcsec": (0.6, 1.8),
        "n": (0.8, 4.0),
        "q": (0.3, 1.0),
        "position_angle": (-1.5, 1.5),
        "ra": (150.0 - 1.0 / 3600.0, 150.0 + 1.0 / 3600.0),
        "dec": (30.0 - 1.0 / 3600.0, 30.0 + 1.0 / 3600.0),
        "sky__g": (-20.0, 20.0),
        "sky__r": (-20.0, 20.0),
    }
    actual_bounds = case.problem.parameter_layout.parameter_set.bounds

    assert actual_bounds.keys() == expected_bounds.keys()
    for parameter_name, expected in expected_bounds.items():
        assert actual_bounds[parameter_name] == pytest.approx(expected)


def test_preparation_free_case_rejects_nonpositive_known_noise():
    truth = _truth()
    observations = dict(truth.observations)
    observations["g"] = replace(
        observations["g"],
        noise_metadata={"sigma": 0.0},
    )
    invalid = replace(truth, observations=observations)

    with pytest.raises(ValueError, match="sigma must be finite and positive"):
        build_preparation_free_multi_observation_case(invalid)


def test_preparation_free_case_rejects_duplicate_noise_seeds():
    truth = _truth()
    observations = {
        name: replace(
            observation,
            metadata={**observation.metadata, "seed": 17},
        )
        for name, observation in truth.observations.items()
    }
    invalid = replace(truth, observations=observations)

    with pytest.raises(ValueError, match="distinct noise seeds"):
        build_preparation_free_multi_observation_case(invalid)


def test_preparation_free_case_persists_exact_inputs(tmp_path):
    case = build_preparation_free_multi_observation_case(_truth())
    path = case.save_inputs(tmp_path / "inputs.npz")

    with np.load(path, allow_pickle=False) as archive:
        assert archive["parameter_names"].tolist() == case.problem.parameter_names
        np.testing.assert_array_equal(
            archive["observed__r"],
            case.observed_images["r"],
        )
        np.testing.assert_array_equal(
            archive["noise__g"],
            case.noise_maps["g"],
        )


def test_preparation_free_campaign_builds_public_direct_case(tmp_path):
    case = _campaign(tmp_path).build_case(0)

    assert case.noise_seeds == {"g": 10000, "r": 10001}
    assert case.problem.likelihood.image_shapes == {
        "g": (61, 61),
        "r": (47, 73),
    }


def test_preparation_free_campaign_reuses_end_to_end_seed_schedule(tmp_path):
    direct = _campaign(tmp_path / "direct")
    end_to_end = MultiObservationNoiseCampaign(
        root=tmp_path / "end_to_end",
        truth=_truth(),
        sampler=direct.sampler,
        n_cases=direct.n_cases,
    )

    for index in range(direct.n_cases):
        direct_truth = direct._truth_for_case(index)
        end_to_end_truth = end_to_end._truth_for_case(index)
        direct_seeds = {
            name: observation.metadata["seed"]
            for name, observation in direct_truth.observations.items()
        }
        end_to_end_seeds = {
            name: observation.metadata["seed"]
            for name, observation in end_to_end_truth.observations.items()
        }
        assert direct_seeds == end_to_end_seeds


def test_preparation_free_campaign_uses_direct_problem_and_pocomc(
    tmp_path,
    monkeypatch,
):
    campaign = _campaign(tmp_path)
    problem = object()
    captured = {}

    class FakeCase:
        def __init__(self):
            self.problem = problem

        def save_inputs(self, path):
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.touch()
            return output

    class FakeInferenceResult:
        diagnostics = {}

        def save(self, path):
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.touch()
            return output

    fake_result = FakeInferenceResult()

    def fake_build(truth):
        captured["truth"] = truth
        return FakeCase()

    def fake_make_sampler(config, *, problem):
        captured["sampler"] = config
        captured["problem"] = problem
        return SimpleNamespace(fit=lambda **kwargs: fake_result)

    expected = object()
    monkeypatch.setattr(
        multi_observation_preparation_free,
        "build_preparation_free_multi_observation_case",
        fake_build,
    )
    monkeypatch.setattr(
        multi_observation_preparation_free,
        "make_sampler",
        fake_make_sampler,
    )
    monkeypatch.setattr(
        PreparationFreeMultiObservationNoiseCampaign,
        "_recovery_result",
        lambda self, **kwargs: expected,
    )

    result = campaign.run_case(0)

    assert result is expected
    assert captured["sampler"] == campaign.sampler
    assert captured["problem"] is problem
    assert (
        captured["truth"].observations["g"].metadata["seed"]
        != captured["truth"].observations["r"].metadata["seed"]
    )
    case_dir = tmp_path / "preparation_free" / "noise-000"
    assert (case_dir / "preparation_free_inputs.npz").exists()
    assert (case_dir / "joint_inference_result.npz").exists()


def test_preparation_free_campaign_records_construction_failures(
    tmp_path,
    monkeypatch,
):
    campaign = _campaign(tmp_path)

    def fail_build(truth):
        raise ValueError("invalid direct problem")

    monkeypatch.setattr(
        multi_observation_preparation_free,
        "build_preparation_free_multi_observation_case",
        fail_build,
    )

    with pytest.raises(ValueError, match="invalid direct problem"):
        campaign.run_case(0)

    failure_path = (
        tmp_path
        / "preparation_free"
        / "noise-000"
        / "failure.yaml"
    )
    with failure_path.open() as stream:
        record = yaml.safe_load(stream)

    assert record["stage"] == "problem_construction"


def test_preparation_free_recovery_metadata_identifies_native_grid():
    campaign = PreparationFreeMultiObservationNoiseCampaign(
        root="unused",
        truth=_truth(),
        sampler={"name": "pocomc"},
        n_cases=1,
    )
    truth = campaign._truth_for_case(0)
    names = list(truth.parameters)
    center = np.asarray(
        [truth.parameter(name) for name in names],
        dtype=float,
    )
    posterior = PosteriorSamples(
        samples=np.vstack(
            [
                center - 1.0e-6,
                center,
                center + 1.0e-6,
            ]
        ),
        weights=np.ones(3),
        parameter_names=names,
    )
    inference_result = InferenceResult(
        posterior=posterior,
        diagnostics={"calls": 123},
        provenance={"backend": "pocomc"},
    )

    recovery = campaign._recovery_result(
        index=0,
        inference_result=inference_result,
        runtime_seconds=1.5,
    )

    assert recovery.metadata["campaign"] == (
        "preparation_free_multi_observation_noise"
    )
    assert recovery.metadata["preparation_free"] is True
    assert recovery.metadata["likelihood_grid"] == "native_detector"
    assert recovery.likelihood_evaluations == 123
