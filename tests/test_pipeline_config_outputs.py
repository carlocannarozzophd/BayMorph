import pytest
import yaml

from baymorph import load_resolved_config, resolve_config
from baymorph import make_output_tree, save_resolved_config
from baymorph import pipeline_paths


def test_resolve_minimal_config():
    config = {
        "object": {"name": "NGC1234"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": "outputs"},
    }

    resolved = resolve_config(config)

    assert resolved["object"]["name"] == "NGC1234"
    assert resolved["data"]["image_path"] == "data/image.fits"
    assert resolved["outputs"]["root"] == "outputs"
    assert "psf" in resolved
    assert "fit" in resolved


def test_missing_required_section_raises():
    config = {
        "object": {"name": "NGC1234"},
        "outputs": {"root": "outputs"},
    }

    with pytest.raises(ValueError, match="Missing required config sections"):
        resolve_config(config)


def test_output_tree_creation(tmp_path):
    config = {
        "object": {"name": "NGC1234"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path / "outputs")},
        "fit": {"models": ["single_sersic", "double_sersic"]},
    }

    resolved = resolve_config(config)
    base = make_output_tree(resolved)

    assert base == tmp_path / "outputs" / "NGC1234"
    assert (base / "psf").is_dir()
    assert (base / "fits").is_dir()
    assert (base / "model_comparison").is_dir()

    for model_name in resolved["fit"]["models"]:
        assert (base / "fits" / model_name).is_dir()


def test_output_tree_creation_uses_only_requested_models(tmp_path):
    config = {
        "object": {"name": "NGC1234"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path / "outputs")},
        "fit": {"models": ["single_sersic"]},
    }

    resolved = resolve_config(config)
    base = make_output_tree(resolved)

    assert (base / "fits" / "single_sersic").is_dir()
    assert not (base / "fits" / "double_sersic").exists()


def test_save_resolved_config(tmp_path):
    config = {
        "object": {"name": "NGC1234"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path / "outputs")},
    }

    resolved = resolve_config(config)
    path = save_resolved_config(resolved)

    assert path.exists()

    with path.open("r") as f:
        loaded = yaml.safe_load(f)

    assert loaded["object"]["name"] == "NGC1234"


def test_load_resolved_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
object:
  name: NGC1234
data:
  image_path: data/image.fits
outputs:
  root: outputs
"""
    )

    config = load_resolved_config(path)

    assert config["object"]["name"] == "NGC1234"
    assert "diagnostics" in config


def test_example_config_loads():
    config = load_resolved_config("configs/real_pipeline.yaml")

    assert "name" in config["object"]
    assert isinstance(config["object"]["name"], str)
    assert len(config["object"]["name"]) > 0
    assert config["psf"]["type"] == "empirical"
    assert "single_sersic" in config["fit"]["models"]
    assert "double_sersic" in config["fit"]["models"]


def test_pipeline_paths_are_canonical(tmp_path):
    config = {
        "object": {"name": "NGC1234"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path / "outputs")},
        "fit": {"models": ["single_sersic", "double_sersic"]},
    }

    resolved = resolve_config(config)
    paths = pipeline_paths(resolved)

    base = tmp_path / "outputs" / "NGC1234"

    assert paths["base"] == base
    assert paths["config_resolved"] == base / "config_resolved.yaml"
    assert paths["psf"]["psf_fits"] == base / "psf" / "psf.fits"
    assert paths["psf"]["metadata"] == base / "psf" / "psf_metadata.json"

    for model_name in resolved["fit"]["models"]:
        fit_dir = base / "fits" / model_name

        assert model_name in paths["fits"]
        assert paths["fits"][model_name]["dir"] == fit_dir
        assert paths["fits"][model_name]["fit_results"] == fit_dir / "fit_results.json"
        assert paths["fits"][model_name]["corner"] == fit_dir / "corner.pdf"
        assert paths["fits"][model_name]["residual"] == fit_dir / "data_model_residual.pdf"
        assert paths["fits"][model_name]["radial_profile"] == fit_dir / "radial_profile.pdf"

    assert paths["model_comparison"]["bayes_factor"] == base / "model_comparison" / "bayes_factor.json"


def test_pipeline_paths_use_only_requested_models(tmp_path):
    config = {
        "object": {"name": "NGC1234"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path / "outputs")},
        "fit": {"models": ["single_sersic"]},
    }

    resolved = resolve_config(config)
    paths = pipeline_paths(resolved)

    assert "single_sersic" in paths["fits"]
    assert "double_sersic" not in paths["fits"]


def test_resolve_config_applies_nested_defaults():
    config = {
        "object": {"name": "NGC1234"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": "outputs"},
    }

    resolved = resolve_config(config)

    assert resolved["data"]["mask_path"] is None
    assert resolved["outputs"]["overwrite"] is False
    assert resolved["preparation"]["detection"]["n_sigma"] == 3.0
    assert resolved["preparation"]["detection"]["n_levels"] == 32
    assert resolved["psf"]["type"] == "empirical"
    assert resolved["psf"]["empirical"]["construction"]["auto_size"] is True
    assert resolved["fit"]["models"] == ["double_sersic"]
    assert resolved["fit"]["sampler"]["name"] == "dynesty"
    assert resolved["diagnostics"]["radial_profile"] is True

    candidate_cfg = resolved["psf"]["empirical"]["candidate_selection"]

    assert candidate_cfg["target_n_candidates"] == 30
    assert candidate_cfg["sigma_factor"] == 4.0
    assert candidate_cfg["min_half_size"] == 12
    assert candidate_cfg["max_half_size"] == 30


def test_resolve_config_preserves_user_nested_overrides():
    config = {
        "object": {"name": "NGC1234"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": "outputs"},
        "preparation": {
            "detection": {
                "n_sigma": 5.0,
            },
        },
        "fit": {
            "models": ["single_sersic", "double_sersic"],
            "sampler": {
                "nlive": 1000,
            },
        },
    }

    resolved = resolve_config(config)

    assert resolved["preparation"]["detection"]["n_sigma"] == 5.0
    assert resolved["preparation"]["detection"]["n_levels"] == 32
    assert resolved["fit"]["models"] == ["single_sersic", "double_sersic"]
    assert resolved["fit"]["sampler"]["name"] == "dynesty"
    assert resolved["fit"]["sampler"]["nlive"] == 1000


def test_resolve_config_preserves_psf_candidate_selection_overrides():
    config = {
        "object": {"name": "NGC1234"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": "outputs"},
        "psf": {
            "empirical": {
                "candidate_selection": {
                    "target_n_candidates": 8,
                    "sigma_factor": 5.0,
                    "min_half_size": 10,
                    "max_half_size": 50,
                },
            },
        },
    }

    resolved = resolve_config(config)
    candidate_cfg = resolved["psf"]["empirical"]["candidate_selection"]

    assert candidate_cfg["auto"] is True
    assert candidate_cfg["target_n_candidates"] == 8
    assert candidate_cfg["sigma_factor"] == 5.0
    assert candidate_cfg["min_half_size"] == 10
    assert candidate_cfg["max_half_size"] == 50