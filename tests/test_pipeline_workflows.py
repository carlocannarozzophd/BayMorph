from pathlib import Path

import pytest

from baymorph.pipeline import (
    initialize_pipeline_run,
    load_empirical_psf,
    resolve_config,
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
    assert (base / "fits").is_dir()

    for model_name in resolved["fit"]["models"]:
        assert (base / "fits" / model_name).is_dir()

def test_run_real_fit_accepts_dict_config_and_requires_preparation(
    tmp_path,
):
    config = minimal_config(tmp_path)
    config.setdefault(
        "fit",
        {},
    ).setdefault(
        "priors",
        {},
    )[
        "double_sersic"
    ] = {
        "flux_bulge": {
            "type": "uniform",
            "lower": 1.0,
            "upper": 1000.0,
        },
        "flux_disk": {
            "type": "uniform",
            "lower": 1.0,
            "upper": 1000.0,
        },
    }


    with pytest.raises(
        FileNotFoundError,
        match="Image-preparation products are required before fitting",
    ):
        run_real_fit(config)


def test_run_image_preparation_requires_existing_fits_file(tmp_path):
    config = minimal_config(tmp_path)

    with pytest.raises(FileNotFoundError):
        run_image_preparation(config)


def test_run_psf_candidate_selection_requires_existing_fits_file(tmp_path):
    config = minimal_config(tmp_path)

    with pytest.raises(FileNotFoundError):
        run_psf_candidate_selection(config)


def test_run_psf_construction_requires_psf_candidates(tmp_path):
    config = minimal_config(tmp_path)

    with pytest.raises(FileNotFoundError):
        run_psf_construction(config)


def test_run_psf_construction_requires_candidate_file(tmp_path):
    config = {
        "object": {"name": "NGC1234"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path / "outputs")},
    }

    resolved = resolve_config(config)

    with pytest.raises(FileNotFoundError):
        load_empirical_psf(resolved)