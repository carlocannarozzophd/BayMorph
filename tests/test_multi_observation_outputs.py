"""Tests for multi-observation output-path management."""

from pathlib import Path

import pytest

from baymorph.pipeline.multi_observation_outputs import (
    make_multi_observation_output_tree,
    multi_observation_output_dir,
    multi_observation_paths,
)


def _config(
    tmp_path: Path,
) -> dict:
    return {
        "observations": {
            "observation_a": {
                "config_path": "a.yaml",
            },
            "visit_2024_01": {
                "config_path": "b.yaml",
            },
        },
        "outputs": {
            "root": str(
                tmp_path
                / "runs"
            ),
            "analysis_name": "target_joint",
        },
        "fit": {
            "models": [
                "single_sersic",
            ],
        },
    }


def test_multi_observation_output_dir_is_analysis_specific(
    tmp_path,
):
    config = _config(
        tmp_path
    )

    assert multi_observation_output_dir(
        config
    ) == (
        tmp_path
        / "runs"
        / "target_joint"
    )


def test_multi_observation_paths_preserve_observation_identifiers(
    tmp_path,
):
    paths = multi_observation_paths(
        _config(
            tmp_path
        )
    )

    observations = paths[
        "fits"
    ][
        "single_sersic"
    ][
        "observations"
    ]

    assert tuple(
        observations
    ) == (
        "observation_a",
        "visit_2024_01",
    )

    assert observations[
        "observation_a"
    ][
        "residual_plot"
    ].name == "data_model_residual.pdf"


def test_multi_observation_paths_support_configured_plot_format(
    tmp_path,
):
    config = _config(
        tmp_path
    )

    config[
        "diagnostics"
    ] = {
        "format": "png",
    }

    paths = multi_observation_paths(
        config
    )

    fit_paths = paths[
        "fits"
    ][
        "single_sersic"
    ]

    assert fit_paths[
        "corner"
    ].name == "corner.png"

    assert fit_paths[
        "observations"
    ][
        "visit_2024_01"
    ][
        "radial_profile"
    ].name == "radial_profile.png"


def test_make_multi_observation_output_tree_creates_joint_directories(
    tmp_path,
):
    config = _config(
        tmp_path
    )

    base = (
        make_multi_observation_output_tree(
            config
        )
    )

    assert base.is_dir()

    paths = multi_observation_paths(
        config
    )

    assert paths[
        "fits"
    ][
        "single_sersic"
    ][
        "observations"
    ][
        "observation_a"
    ][
        "dir"
    ].is_dir()

    assert paths[
        "fits"
    ][
        "single_sersic"
    ][
        "observations"
    ][
        "visit_2024_01"
    ][
        "dir"
    ].is_dir()


def test_multi_observation_outputs_require_analysis_name(
    tmp_path,
):
    config = _config(
        tmp_path
    )

    del config[
        "outputs"
    ][
        "analysis_name"
    ]

    with pytest.raises(
        ValueError,
        match="outputs.analysis_name",
    ):
        multi_observation_paths(
            config
        )


@pytest.mark.parametrize(
    "identifier",
    [
        "../escape",
        "nested/name",
        ".",
        "..",
    ],
)
def test_multi_observation_output_paths_reject_unsafe_identifiers(
    tmp_path,
    identifier,
):
    config = _config(
        tmp_path
    )

    config[
        "observations"
    ] = {
        identifier: {
            "config_path": "a.yaml",
        },
    }

    with pytest.raises(
        ValueError,
        match="Observation identifier",
    ):
        multi_observation_paths(
            config
        )
