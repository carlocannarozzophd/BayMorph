from pathlib import Path

import pytest
import yaml
from astropy.io import fits

from baymorph.validation import (
    GroundTruth,
    ValidationWorkspace,
)


def make_truth(**overrides):
    values = {
        "model_name": "single_sersic",
        "parameters": {
            "flux": 1000.0,
            "re": 5.0,
            "n": 2.0,
            "q": 0.8,
            "theta": 0.3,
            "x0": 15.5,
            "y0": 15.5,
        },
        "image_shape": (32, 32),
        "pixel_scale": 0.2,
        "sky_level": 10.0,
        "random_seed": 42,
        "psf_metadata": {
            "fwhm": 2.0,
        },
        "noise_metadata": {
            "poisson_noise": False,
            "read_noise_sigma": None,
        },
    }
    values.update(overrides)
    return GroundTruth(**values)


def test_validation_workspace_writes_pipeline_inputs(tmp_path):
    workspace = ValidationWorkspace(tmp_path)

    config_path = workspace.write_case(
        "case-0001",
        make_truth(),
    )

    assert config_path == tmp_path / "case-0001" / "validation.yaml"
    assert config_path.exists()

    with config_path.open() as stream:
        config = yaml.safe_load(stream)

    image_path = Path(config["data"]["image_path"])

    assert image_path.exists()
    assert config["object"]["name"] == "case-0001"
    assert config["data"]["target_position"] == {
        "x": 15.5,
        "y": 15.5,
    }
    assert config["data"]["cutout"] == {
        "mode": "fixed",
        "size": 32,
    }
    assert config["psf"]["type"] == "gaussian"
    assert config["psf"]["gaussian"]["fwhm"] == 2.0
    assert config["fit"]["models"] == ["single_sersic"]
    assert config["fit"]["sampler"]["seed"] == 42
    assert "masking" not in config

    with fits.open(image_path) as hdul:
        assert hdul[0].data.shape == (32, 32)


def test_validation_workspace_accepts_sampler_overrides(tmp_path):
    workspace = ValidationWorkspace(tmp_path)

    config_path = workspace.write_case(
        "case-0001",
        make_truth(),
        sampler={
            "nlive": 250,
            "dlogz": 0.1,
        },
    )

    with config_path.open() as stream:
        config = yaml.safe_load(stream)

    assert config["fit"]["sampler"]["nlive"] == 250
    assert config["fit"]["sampler"]["dlogz"] == 0.1
    assert config["fit"]["sampler"]["seed"] == 42


def test_validation_workspace_enables_model_comparison(tmp_path):
    workspace = ValidationWorkspace(tmp_path)

    config_path = workspace.write_case(
        "case-0001",
        make_truth(),
        models=(
            "single_sersic",
            "double_sersic",
        ),
    )

    with config_path.open() as stream:
        config = yaml.safe_load(stream)

    assert config["fit"]["models"] == [
        "single_sersic",
        "double_sersic",
    ]
    assert config["fit"]["comparison"]["enabled"]


def test_validation_workspace_rejects_invalid_case_id(tmp_path):
    workspace = ValidationWorkspace(tmp_path)

    with pytest.raises(ValueError, match="Case ID"):
        workspace.write_case(
            "../case-0001",
            make_truth(),
        )


def test_validation_workspace_requires_square_images(tmp_path):
    workspace = ValidationWorkspace(tmp_path)

    with pytest.raises(ValueError, match="square images"):
        workspace.write_case(
            "case-0001",
            make_truth(image_shape=(32, 48)),
        )


def test_validation_workspace_requires_gaussian_psf(tmp_path):
    workspace = ValidationWorkspace(tmp_path)

    with pytest.raises(ValueError, match="PSF metadata"):
        workspace.write_case(
            "case-0001",
            make_truth(psf_metadata={}),
        )