"""End-to-end smoke test for heterogeneous WCS-aware validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from baymorph.pipeline.image_preparation import (
    prepare_real_image_from_config,
)
from baymorph.pipeline.multi_observation_fit import (
    build_multi_observation_problem_from_config,
)
from baymorph.pipeline.noise_calibration import (
    build_correlated_noise_calibration,
)
from baymorph.pipeline.vecchia_factorization import (
    ensure_vecchia_factorization_artifact,
)
from baymorph.validation import (
    MultiObservationGroundTruth,
    MultiObservationValidationWorkspace,
    ObservationTruth,
)


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
            "background_level__g": 0.0,
            "background_level__r": 0.0,
        },
        observations={
            # Large enough for the sky correlation of AD-047, which needs
            # 1000 sky pixel pairs at every lag up to 16.
            "g": ObservationTruth(
                image_shape=(121, 121),
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
                        61.0,
                        61.0,
                    ),
                    "seed": 11,
                    "raw_background": 100.0,
                },
            ),
            "r": ObservationTruth(
                image_shape=(95, 147),
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
                        74.0,
                        48.0,
                    ),
                    "seed": 22,
                    "raw_background": 100.0,
                },
            ),
        },
        random_seed=42,
    )


def _prepare_observations(
    joint_path: Path,
) -> None:
    with joint_path.open() as stream:
        joint = yaml.safe_load(
            stream
        )

    for observation in joint["observations"].values():
        config_path = (
            joint_path.parent
            / observation["config_path"]
        )

        prepare_real_image_from_config(
            config_path,
            save=True,
        )
        # The noise artifacts of AD-047, which `baymorph run` builds.
        build_correlated_noise_calibration(
            config_path,
            save=True,
        )
        ensure_vecchia_factorization_artifact(
            config_path,
        )


def _truth_vector(
    problem,
    truth: MultiObservationGroundTruth,
) -> np.ndarray:
    return np.asarray(
        [
            truth.parameters[name]
            for name in problem.parameter_names
        ],
        dtype=float,
    )


def test_multi_observation_validation_case_builds_and_scores_truth(
    tmp_path,
):
    truth = _truth()

    workspace = MultiObservationValidationWorkspace(
        tmp_path
    )

    joint_path = workspace.write_case(
        "case-0001",
        truth,
    )

    _prepare_observations(
        joint_path
    )

    problem = build_multi_observation_problem_from_config(
        joint_path
    )

    theta = _truth_vector(
        problem,
        truth,
    )

    assert problem.parameter_layout.observations == (
        "g",
        "r",
    )

    assert problem.likelihood.image_shapes[
        "g"
    ] != problem.likelihood.image_shapes[
        "r"
    ]

    rendered = problem.render_models(
        theta
    )

    assert rendered["g"].shape == (
        problem.likelihood.image_shapes[
            "g"
        ]
    )
    assert rendered["r"].shape == (
        problem.likelihood.image_shapes[
            "r"
        ]
    )

    log_likelihood = problem.log_likelihood(
        theta
    )

    assert np.isfinite(
        log_likelihood
    )


def test_multi_observation_validation_truth_beats_perturbed_geometry(
    tmp_path,
):
    truth = _truth()

    workspace = MultiObservationValidationWorkspace(
        tmp_path
    )

    joint_path = workspace.write_case(
        "case-0001",
        truth,
    )

    _prepare_observations(
        joint_path
    )

    problem = build_multi_observation_problem_from_config(
        joint_path
    )

    theta = _truth_vector(
        problem,
        truth,
    )

    truth_logl = problem.log_likelihood(
        theta
    )

    perturbed = theta.copy()

    re_index = problem.parameter_names.index(
        "re_arcsec"
    )
    ra_index = problem.parameter_names.index(
        "ra"
    )

    perturbed[
        re_index
    ] = 1.6

    perturbed[
        ra_index
    ] += (
        0.3
        / 3600.0
    )

    perturbed_logl = problem.log_likelihood(
        perturbed
    )

    assert truth_logl > perturbed_logl
