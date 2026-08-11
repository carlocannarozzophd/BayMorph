"""Tests for prior-predictive multi-observation SBC construction."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.validation import (
    MultiObservationGroundTruth,
    ObservationTruth,
)
from baymorph.validation.multi_observation_sbc import (
    build_preparation_free_sbc_case,
    draw_multi_observation_sbc_truth,
    reference_multi_observation_sbc_parameter_config,
)


def _reference_truth() -> MultiObservationGroundTruth:
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
                },
            ),
        },
        random_seed=42,
    )


def test_sbc_truth_draw_is_reproducible():
    first = draw_multi_observation_sbc_truth(
        _reference_truth(),
        index=3,
    )
    second = draw_multi_observation_sbc_truth(
        _reference_truth(),
        index=3,
    )

    np.testing.assert_array_equal(
        first.unit_cube,
        second.unit_cube,
    )
    assert first.truth.parameters == second.truth.parameters

    first_noise_seeds = {
        name: observation.metadata["seed"]
        for name, observation in first.truth.observations.items()
    }
    second_noise_seeds = {
        name: observation.metadata["seed"]
        for name, observation in second.truth.observations.items()
    }

    assert first_noise_seeds == second_noise_seeds


def test_sbc_case_indices_produce_distinct_truth_draws():
    first = draw_multi_observation_sbc_truth(
        _reference_truth(),
        index=0,
    )
    second = draw_multi_observation_sbc_truth(
        _reference_truth(),
        index=1,
    )

    assert not np.array_equal(
        first.unit_cube,
        second.unit_cube,
    )
    assert first.truth.parameters != second.truth.parameters


def test_sbc_truth_is_exact_parameter_set_transform():
    draw = draw_multi_observation_sbc_truth(
        _reference_truth(),
        index=0,
    )

    expected = draw.parameter_layout.parameter_set.transform(
        draw.unit_cube
    )

    actual = np.asarray(
        [
            draw.truth.parameter(name)
            for name in draw.parameter_layout.parameter_set.names
        ],
        dtype=float,
    )

    np.testing.assert_array_equal(actual, expected)


def test_sbc_prior_contract_has_expected_global_support():
    draw = draw_multi_observation_sbc_truth(
        _reference_truth(),
        index=0,
    )

    bounds = draw.parameter_layout.parameter_set.bounds

    assert bounds["flux__g"] == pytest.approx((450.0, 1350.0))
    assert bounds["flux__r"] == pytest.approx((600.0, 1800.0))
    assert bounds["re_arcsec"] == pytest.approx((0.6, 1.8))
    assert bounds["n"] == pytest.approx((0.8, 4.0))
    assert bounds["q"] == pytest.approx((0.3, 1.0))
    assert bounds["sky__g"] == pytest.approx((-20.0, 20.0))
    assert bounds["sky__r"] == pytest.approx((-20.0, 20.0))


def test_sbc_routing_preserves_shared_and_independent_parameters():
    draw = draw_multi_observation_sbc_truth(
        _reference_truth(),
        index=0,
    )

    routed = draw.parameter_layout.route(
        dict(draw.truth.parameters)
    )

    assert routed["g"]["re_arcsec"] == routed["r"]["re_arcsec"]
    assert routed["g"]["n"] == routed["r"]["n"]
    assert routed["g"]["q"] == routed["r"]["q"]

    assert routed["g"]["flux"] == draw.truth.parameter("flux__g")
    assert routed["r"]["flux"] == draw.truth.parameter("flux__r")
    assert routed["g"]["sky"] == draw.truth.parameter("sky__g")
    assert routed["r"]["sky"] == draw.truth.parameter("sky__r")


def test_sbc_truth_and_noise_randomness_are_distinct():
    draw = draw_multi_observation_sbc_truth(
        _reference_truth(),
        index=7,
    )

    noise_seeds = [
        observation.metadata["seed"]
        for observation in draw.truth.observations.values()
    ]

    assert draw.truth_seed == 30007
    assert noise_seeds == [40014, 40015]
    assert draw.truth_seed not in noise_seeds
    assert len(set(noise_seeds)) == 2


def test_preparation_free_sbc_case_closes_at_drawn_truth():
    draw, case = build_preparation_free_sbc_case(
        _reference_truth(),
        index=0,
    )

    assert case.problem.parameter_names == (
        draw.parameter_layout.parameter_set.names
    )

    np.testing.assert_array_equal(
        case.truth_vector,
        draw.parameter_layout.parameter_set.transform(
            draw.unit_cube
        ),
    )

    rendered = case.problem.render_models(
        case.truth_vector
    )

    for observation in case.problem.parameter_layout.bands:
        np.testing.assert_allclose(
            rendered[observation],
            case.noiseless_images[observation],
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    assert np.isfinite(
        case.problem.log_likelihood(
            case.truth_vector
        )
    )


def test_reference_sbc_prior_is_truth_independent():
    first = reference_multi_observation_sbc_parameter_config()
    second = reference_multi_observation_sbc_parameter_config()

    assert first == second


def _sbc_campaign(
    tmp_path,
):
    from baymorph.validation.multi_observation_sbc import (
        PreparationFreeMultiObservationSBCCampaign,
    )

    return PreparationFreeMultiObservationSBCCampaign(
        root=tmp_path,
        truth=_reference_truth(),
        sampler={
            "name": "pocomc",
            "init": {
                "n_effective": 512,
                "n_active": 256,
                "sample": "tpcn",
                "random_state": 42,
                "pool": 10,
            },
            "run": {
                "n_total": 4096,
                "n_evidence": 4096,
            },
        },
        n_cases=3,
    )


def test_sbc_campaign_uses_prior_predictive_truth(tmp_path):
    campaign = _sbc_campaign(tmp_path)

    first = campaign._truth_for_case(0)
    second = campaign._truth_for_case(1)

    expected_first = draw_multi_observation_sbc_truth(
        _reference_truth(),
        index=0,
    ).truth

    assert first.parameters == expected_first.parameters
    assert first.parameters != second.parameters


def test_sbc_campaign_builds_problem_with_fixed_sbc_prior(tmp_path):
    campaign = _sbc_campaign(tmp_path)

    case = campaign.build_case(0)

    assert case.problem.parameter_layout.parameter_set.bounds == pytest.approx(
        {
            "flux__g": (450.0, 1350.0),
            "flux__r": (600.0, 1800.0),
            "re_arcsec": (0.6, 1.8),
            "n": (0.8, 4.0),
            "q": (0.3, 1.0),
            "position_angle": (
                -0.5 * np.pi,
                0.5 * np.pi,
            ),
            "ra": (
                150.0 - 1.0 / 3600.0,
                150.0 + 1.0 / 3600.0,
            ),
            "dec": (
                30.0 - 1.0 / 3600.0,
                30.0 + 1.0 / 3600.0,
            ),
            "sky__g": (-20.0, 20.0),
            "sky__r": (-20.0, 20.0),
        }
    )


def test_sbc_campaign_uses_isolated_case_paths(tmp_path):
    campaign = _sbc_campaign(tmp_path)

    assert campaign._case_id(2) == "sbc-002"
    assert campaign._case_dir(2) == (
        tmp_path
        / "preparation_free_sbc"
        / "sbc-002"
    )


def test_sbc_campaign_persists_exact_truth_draw(tmp_path):
    campaign = _sbc_campaign(tmp_path)
    draw = campaign.draw_case_truth(0)

    path = campaign._save_truth_draw(draw)

    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["unit_cube"],
            draw.unit_cube,
        )
        np.testing.assert_array_equal(
            archive["truth_vector"],
            np.asarray(
                [
                    draw.truth.parameter(name)
                    for name
                    in draw.parameter_layout.parameter_set.names
                ],
                dtype=float,
            ),
        )

        assert int(archive["truth_seed"]) == 30000
        assert int(archive["noise_seed__g"]) == 40000
        assert int(archive["noise_seed__r"]) == 40001


def test_sbc_campaign_resume_reuses_completed_case(
    tmp_path,
    monkeypatch,
):
    campaign = _sbc_campaign(tmp_path)
    object.__setattr__(campaign, "n_cases", 1)

    result_path = campaign._result_path(0)
    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    result_path.touch()

    sentinel = object()

    monkeypatch.setattr(
        type(campaign),
        "load_case",
        lambda self, index: sentinel,
    )

    def fail_run_case(self, index):
        raise AssertionError(
            "Completed SBC case should not be rerun."
        )

    monkeypatch.setattr(
        type(campaign),
        "run_case",
        fail_run_case,
    )

    results = campaign.run(
        resume=True,
        continue_on_failure=False,
    )

    assert results == [sentinel]


def test_sbc_position_angle_prior_spans_one_axial_period():
    draw = draw_multi_observation_sbc_truth(
        _reference_truth(),
        index=0,
    )

    lower, upper = (
        draw.parameter_layout.parameter_set.bounds[
            "position_angle"
        ]
    )

    assert lower == pytest.approx(-0.5 * np.pi)
    assert upper == pytest.approx(0.5 * np.pi)
    assert upper - lower == pytest.approx(np.pi)
