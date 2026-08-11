"""Tests for heterogeneous multi-observation validation workspaces."""

from __future__ import annotations

from pathlib import Path

import yaml

from baymorph.validation import (
    MultiObservationGroundTruth,
    MultiObservationValidationWorkspace,
    ObservationTruth,
)


def make_truth():
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
                metadata={
                    "rotation_deg": 0.0,
                    "crpix": (
                        31.0,
                        31.0,
                    ),
                    "seed": 11,
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
                    "seed": 22,
                    "raw_background": 100.0,
                },
            ),
        },
        random_seed=42,
    )


def test_workspace_writes_heterogeneous_joint_case(
    tmp_path,
):
    workspace = MultiObservationValidationWorkspace(
        tmp_path
    )

    joint_path = workspace.write_case(
        "case-0001",
        make_truth(),
    )

    assert joint_path.exists()

    case_dir = (
        tmp_path
        / "case-0001"
    )

    assert (
        case_dir
        / "input"
        / "g.fits"
    ).exists()

    assert (
        case_dir
        / "input"
        / "r.fits"
    ).exists()

    assert (
        case_dir
        / "configs"
        / "g.yaml"
    ).exists()

    assert (
        case_dir
        / "configs"
        / "r.yaml"
    ).exists()

    with joint_path.open() as stream:
        joint = yaml.safe_load(
            stream
        )

    assert joint[
        "fit"
    ][
        "multiband"
    ][
        "parameter_frame"
    ] == "sky"

    assert joint[
        "observations"
    ] == {
        "g": {
            "config_path": (
                "configs/g.yaml"
            ),
        },
        "r": {
            "config_path": (
                "configs/r.yaml"
            ),
        },
    }


def test_workspace_preserves_native_observation_shapes(
    tmp_path,
):
    from astropy.io import fits

    workspace = MultiObservationValidationWorkspace(
        tmp_path
    )

    workspace.write_case(
        "case-0001",
        make_truth(),
    )

    case_dir = (
        Path(tmp_path)
        / "case-0001"
        / "input"
    )

    assert fits.getdata(
        case_dir / "g.fits"
    ).shape == (
        61,
        61,
    )

    assert fits.getdata(
        case_dir / "r.fits"
    ).shape == (
        47,
        73,
    )


def test_workspace_preserves_explicit_sampler_contract(
    tmp_path,
):
    workspace = MultiObservationValidationWorkspace(
        tmp_path
    )

    joint_path = workspace.write_case(
        "case-pocomc",
        make_truth(),
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
    )

    with joint_path.open() as stream:
        joint = yaml.safe_load(
            stream
        )

    assert joint["fit"]["sampler"] == {
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
    }

    assert (
        "mode"
        not in joint["fit"]["sampler"]
    )
