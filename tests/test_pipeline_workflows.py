from pathlib import Path

import pytest

from baymorph import (
    initialize_pipeline_run,
    run_image_preparation,
    run_psf_candidate_selection,
    run_psf_construction,
    run_real_fit,
)


def minimal_config(tmp_path):
    return {
        "object": {"name": "NGC1234"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path / "outputs")},
    }


def test_pipeline_entrypoints_exist():
    assert callable(initialize_pipeline_run)
    assert callable(run_image_preparation)
    assert callable(run_psf_candidate_selection)
    assert callable(run_psf_construction)
    assert callable(run_real_fit)


def test_initialize_pipeline_run_creates_output_tree(tmp_path):
    config = minimal_config(tmp_path)

    resolved = initialize_pipeline_run(config)

    base = Path(resolved["outputs"]["root"]) / resolved["object"]["name"]

    assert base.is_dir()
    assert (base / "config_resolved.yaml").exists()
    assert (base / "psf").is_dir()
    assert (base / "fits" / "single_sersic").is_dir()
    assert (base / "fits" / "double_sersic").is_dir()


def test_workflow_entrypoints_accept_dict_config(tmp_path):
    config = minimal_config(tmp_path)

    for runner in (
        run_psf_candidate_selection,
        run_psf_construction,
        run_real_fit,
    ):
        resolved = runner(config)
        assert resolved["object"]["name"] == "NGC1234"


def test_run_image_preparation_requires_existing_fits_file(tmp_path):
    config = minimal_config(tmp_path)

    with pytest.raises(FileNotFoundError):
        run_image_preparation(config)