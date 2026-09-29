from pathlib import Path
import numpy as np
import pytest
import yaml
from astropy.io import fits

from baymorph.pipeline import (
    load_resolved_config,
    make_output_tree,
    pipeline_paths,
    resolve_config,
    resolve_runtime_config,
    save_fit_resolved_config,
    save_resolved_config,
)

from baymorph.core.image import ImageData
from baymorph.core.observation import Observation
from baymorph.likelihood import make_likelihood

def test_resolve_minimal_config():
    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": "outputs"},
    }

    resolved = resolve_config(config)

    assert resolved["object"]["name"] == "example_galaxy"
    assert resolved["data"]["image_path"] == "data/image.fits"
    assert resolved["outputs"]["root"] == "outputs"
    assert "psf" in resolved
    assert "fit" in resolved


def test_missing_required_section_raises():
    config = {
        "object": {"name": "example_galaxy"},
        "outputs": {"root": "outputs"},
    }

    with pytest.raises(ValueError, match="Missing required config sections"):
        resolve_config(config)


def test_output_tree_creation(tmp_path):
    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path / "outputs")},
        "fit": {"models": ["single_sersic", "double_sersic"]},
    }

    resolved = resolve_config(config)
    base = make_output_tree(resolved)

    assert base == tmp_path / "outputs" / "example_galaxy"
    assert (base / "psf").is_dir()
    assert (base / "fits").is_dir()
    assert (base / "model_comparison").is_dir()

    for model_name in resolved["fit"]["models"]:
        assert (base / "fits" / model_name).is_dir()


def test_output_tree_creation_uses_only_requested_models(tmp_path):
    config = {
        "object": {"name": "example_galaxy"},
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
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path / "outputs")},
    }

    resolved = resolve_config(config)
    path = save_resolved_config(resolved)

    assert path.exists()

    with path.open("r") as f:
        loaded = yaml.safe_load(f)

    assert loaded["object"]["name"] == "example_galaxy"


def test_save_fit_resolved_config_is_model_local(tmp_path):
    """Inference snapshots must not overwrite the preparation-root config."""
    config = resolve_config(
        {
            "object": {
                "name": "example_galaxy",
            },
            "data": {
                "image_path": "data/image.fits",
            },
            "outputs": {
                "root": str(
                    tmp_path
                    / "outputs"
                ),
            },
            "fit": {
                "models": [
                    "single_sersic",
                ],
            },
        }
    )

    path = save_fit_resolved_config(
        config,
        "single_sersic",
    )
    base = (
        tmp_path
        / "outputs"
        / "example_galaxy"
    )

    assert path == (
        base
        / "fits"
        / "single_sersic"
        / "config_resolved.yaml"
    )
    assert path.is_file()
    assert not (
        base
        / "config_resolved.yaml"
    ).exists()


def test_load_resolved_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
object:
  name: example_galaxy
data:
  image_path: data/image.fits
outputs:
  root: outputs
"""
    )

    config = load_resolved_config(path)

    assert config["object"]["name"] == "example_galaxy"
    assert "diagnostics" in config


def test_official_learn_config_loads():
    config = load_resolved_config(
        "configs/simulated_galaxy.yaml"
    )

    assert config["object"]["name"] == "simulated_galaxy"
    assert config["psf"]["type"] == "gaussian"
    assert config["fit"]["models"] == [
        "single_sersic",
    ]
    assert (
        config["fit"]["sampler"]["name"]
        == "nautilus"
    )


def test_pipeline_paths_are_canonical(tmp_path):
    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path / "outputs")},
        "fit": {"models": ["single_sersic", "double_sersic"]},
    }

    resolved = resolve_config(config)
    paths = pipeline_paths(resolved)

    base = tmp_path / "outputs" / "example_galaxy"

    assert paths["base"] == base
    assert paths["config_resolved"] == base / "config_resolved.yaml"
    assert paths["psf"]["psf_fits"] == base / "psf" / "psf.fits"
    assert paths["psf"]["metadata"] == base / "psf" / "psf_metadata.json"

    for model_name in resolved["fit"]["models"]:
        fit_dir = base / "fits" / model_name

        assert model_name in paths["fits"]
        assert paths["fits"][model_name]["dir"] == fit_dir
        assert paths["fits"][model_name]["config_resolved"] == (
            fit_dir / "config_resolved.yaml"
        )
        assert paths["fits"][model_name]["complete"] == (
            fit_dir / "COMPLETE.json"
        )
        assert paths["fits"][model_name]["fit_summary"] == fit_dir / "fit_summary.json"
        assert (
            paths["fits"][model_name]["fit_summary_text"]
            == fit_dir / "fit_summary.txt"
        )
        assert paths["fits"][model_name]["posterior"] == fit_dir / "posterior.npz"
        assert paths["fits"][model_name]["corner"] == fit_dir / "corner.pdf"
        assert paths["fits"][model_name]["residual"] == fit_dir / "data_model_residual.pdf"
        assert paths["fits"][model_name]["radial_profile"] == fit_dir / "radial_profile.pdf"

    assert paths["model_comparison"]["bayes_factor"] == base / "model_comparison" / "bayes_factor.json"


def test_pipeline_paths_use_only_requested_models(tmp_path):
    config = {
        "object": {"name": "example_galaxy"},
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
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": "outputs"},
    }

    resolved = resolve_config(config)

    assert resolved["data"]["mask_path"] is None
    assert resolved["outputs"]["overwrite"] is False
    assert resolved["preparation"]["detection"]["n_sigma"] == 3.0
    assert resolved["preparation"]["detection"]["n_levels"] == 32
    assert resolved["psf"]["type"] == "empirical"
    assert resolved["fit"]["models"] == ["single_sersic"]
    assert resolved["fit"]["sampler"]["name"] == "nautilus"
    assert resolved["diagnostics"]["radial_profile"] is True

    candidate_cfg = resolved["psf"]["empirical"]["candidate_selection"]

    assert candidate_cfg["target_n_candidates"] == 30
    assert candidate_cfg["sigma_factor"] == 4.0
    assert candidate_cfg["min_half_size"] == 12
    assert candidate_cfg["max_half_size"] == 30


def test_resolve_config_preserves_user_nested_overrides():
    config = {
        "object": {"name": "example_galaxy"},
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
    assert resolved["fit"]["sampler"]["name"] == "nautilus"
    assert resolved["fit"]["sampler"]["nlive"] == 1000


def test_resolve_config_preserves_psf_candidate_selection_overrides():
    config = {
        "object": {"name": "example_galaxy"},
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


def test_resolve_config_defaults_production_mask_and_psf_contract():
    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": "outputs"},
    }

    resolved = resolve_config(config)

    assert resolved["data"]["mask_mode"] == "union"
    assert (
        resolved["psf"]["empirical"]["construction"]
        ["centroid_tolerance_pixels"]
        == pytest.approx(0.05)
    )


@pytest.mark.parametrize("mode", ["unknown", "", None])
def test_resolve_config_rejects_invalid_mask_mode(mode):
    config = {
        "object": {"name": "example_galaxy"},
        "data": {
            "image_path": "data/image.fits",
            "mask_mode": mode,
        },
        "outputs": {"root": "outputs"},
    }

    with pytest.raises(ValueError, match="data.mask_mode"):
        resolve_config(config)


@pytest.mark.parametrize("mode", ["exact", "replace"])
def test_resolve_config_requires_mask_path_for_replacement(mode):
    config = {
        "object": {"name": "example_galaxy"},
        "data": {
            "image_path": "data/image.fits",
            "mask_mode": mode,
        },
        "outputs": {"root": "outputs"},
    }

    with pytest.raises(ValueError, match="data.mask_path is required"):
        resolve_config(config)


def test_resolve_config_rejects_auto_psf_max_peak():
    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": "outputs"},
        "psf": {
            "empirical": {
                "candidate_selection": {"max_peak": "auto"},
            },
        },
    }

    with pytest.raises(ValueError, match="saturation-keyword units"):
        resolve_config(config)


@pytest.mark.parametrize(
    "value",
    [None, True, "50000", 0.0, -1.0, float("nan")],
)
def test_resolve_config_requires_finite_positive_numeric_psf_max_peak(
    value,
):
    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": "outputs"},
        "psf": {
            "empirical": {
                "candidate_selection": {"max_peak": value},
            },
        },
    }

    with pytest.raises(
        ValueError,
        match="max_peak must be a finite positive number",
    ):
        resolve_config(config)


def test_resolve_runtime_config_preserves_numeric_target_position():
    image = ImageData(
        data=np.zeros((20, 30), dtype=float),
    )
    observation = Observation.from_image(image)

    config = {
        "data": {
            "target_position": {
                "x": 12.5,
                "y": 8.0,
            },
        },
    }

    resolved = resolve_runtime_config(
        config,
        observation,
    )

    assert resolved["data"]["target_position"] == {
        "x": 12.5,
        "y": 8.0,
    }
    assert resolved is not config
    assert resolved["data"] is not config["data"]


def test_resolve_runtime_config_resolves_center_target_position():
    image = ImageData(
        data=np.zeros((21, 31), dtype=float),
    )
    observation = Observation.from_image(image)

    config = {
        "data": {
            "target_position": "center",
        },
    }

    resolved = resolve_runtime_config(
        config,
        observation,
    )

    assert resolved["data"]["target_position"] == {
        "x": 15.0,
        "y": 10.0,
    }
    assert config["data"]["target_position"] == "center"


def test_resolve_runtime_config_resolves_auto_pixel_scale():
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 16.0
    header["CRPIX2"] = 11.0
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 2.0
    header["CDELT1"] = -0.21 / 3600.0
    header["CDELT2"] = 0.21 / 3600.0

    image = ImageData(
        data=np.zeros((21, 31), dtype=float),
        header=header,
    )
    observation = Observation.from_image(image)

    config = {
        "data": {
            "pixel_scale": "auto",
        },
    }

    resolved = resolve_runtime_config(
        config,
        observation,
    )

    assert resolved["data"]["pixel_scale"] == pytest.approx(0.21)
    assert config["data"]["pixel_scale"] == "auto"


def test_resolve_runtime_config_resolves_missing_pixel_scale():
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 16.0
    header["CRPIX2"] = 11.0
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 2.0
    header["CDELT1"] = -0.21 / 3600.0
    header["CDELT2"] = 0.21 / 3600.0

    image = ImageData(
        data=np.zeros((21, 31), dtype=float),
        header=header,
    )
    observation = Observation.from_image(image)

    config = {
        "data": {
            "pixel_scale": None,
        },
    }

    resolved = resolve_runtime_config(
        config,
        observation,
    )

    assert resolved["data"]["pixel_scale"] == pytest.approx(0.21)


def test_resolve_runtime_config_preserves_explicit_pixel_scale():
    image = ImageData(
        data=np.zeros((21, 31), dtype=float),
    )
    observation = Observation.from_image(image)

    config = {
        "data": {
            "pixel_scale": 0.4,
        },
    }

    resolved = resolve_runtime_config(
        config,
        observation,
    )

    assert resolved["data"]["pixel_scale"] == 0.4


def test_resolve_runtime_config_keeps_auto_without_metadata_pixel_scale():
    image = ImageData(
        data=np.zeros((21, 31), dtype=float),
    )
    observation = Observation.from_image(image)

    config = {
        "data": {
            "pixel_scale": "auto",
        },
    }

    resolved = resolve_runtime_config(
        config,
        observation,
    )

    assert resolved["data"]["pixel_scale"] == "auto"


def test_resolve_runtime_config_rejects_auto_max_peak_with_satlevel():
    image = ImageData(
        data=np.zeros((21, 31), dtype=float),
        header=fits.Header({"SATLEVEL": 54321.0}),
    )
    observation = Observation.from_image(image)
    config = {
        "psf": {
            "empirical": {
                "candidate_selection": {"max_peak": "auto"},
            },
        },
    }

    with pytest.raises(ValueError, match="saturation-keyword units"):
        resolve_runtime_config(config, observation)


def test_resolve_runtime_config_rejects_auto_max_peak_without_metadata():
    observation = Observation.from_image(
        ImageData(data=np.zeros((21, 31), dtype=float))
    )
    config = {
        "psf": {
            "empirical": {
                "candidate_selection": {"max_peak": "auto"},
            },
        },
    }

    with pytest.raises(
        ValueError,
        match="Supply an explicit finite numerical max_peak",
    ):
        resolve_runtime_config(config, observation)


def test_resolve_runtime_config_preserves_explicit_max_peak():
    image = ImageData(
        data=np.zeros((21, 31), dtype=float),
    )

    observation = Observation.from_image(image)

    config = {
        "psf": {
            "empirical": {
                "candidate_selection": {
                    "max_peak": 12000,
                }
            }
        }
    }

    resolved = resolve_runtime_config(
        config,
        observation,
    )

    assert (
        resolved["psf"]["empirical"]
        ["candidate_selection"]["max_peak"]
        == 12000
    )


def test_simulated_galaxy_config_is_mirrored_and_resolves_diagnostics():
    public_path = Path(
        "configs/simulated_galaxy.yaml"
    )
    resource_path = Path(
        "baymorph/resources/configs/simulated_galaxy.yaml"
    )

    assert public_path.read_bytes() == resource_path.read_bytes()

    resolved = load_resolved_config(
        public_path
    )
    diagnostics = resolved["diagnostics"]

    assert diagnostics["posterior_profile_bands"] is True
    assert diagnostics["radial_profile"] is True

def test_resolve_config_rejects_archived_native_sampler():
    config = {
        "object": {
            "name": "example_galaxy",
        },
        "data": {
            "image_path": "data/image.fits",
        },
        "outputs": {
            "root": "outputs",
        },
        "fit": {
            "sampler": {
                "name": "native",
            },
        },
    }

    with pytest.raises(
        ValueError,
        match="Unknown sampler 'native'",
    ):
        resolve_config(config)


def test_runtime_report_path_is_canonical(tmp_path):
    from baymorph.pipeline import pipeline_paths

    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path)},
    }

    path = pipeline_paths(config)["runtime_report"]

    assert path == tmp_path / "example_galaxy" / "runtime_report.yaml"


def test_build_runtime_report_returns_independent_data(tmp_path):
    from baymorph.pipeline import build_runtime_report

    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path)},
    }
    runtime = {
        "image_preparation": {
            "background": 1.5,
        },
    }

    report = build_runtime_report(
        config,
        runtime=runtime,
    )

    config["object"]["name"] = "modified"
    runtime["image_preparation"]["background"] = 9.0

    assert report["config"]["object"]["name"] == "example_galaxy"
    assert report["runtime"]["image_preparation"]["background"] == 1.5
    assert report["config"]["fit"]["sampler"]["name"] == "nautilus"


def test_runtime_report_roundtrip(tmp_path):
    from baymorph.pipeline import (
        build_runtime_report,
        load_runtime_report,
        save_runtime_report,
    )

    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path)},
    }

    report = build_runtime_report(
        config,
        runtime={
            "image_preparation": {
                "cutout_shape": [51, 51],
                "masked_fraction": 0.12,
            },
        },
    )

    path = save_runtime_report(
        config,
        report,
    )
    loaded = load_runtime_report(config)

    assert path == tmp_path / "example_galaxy" / "runtime_report.yaml"
    assert loaded == report


@pytest.mark.parametrize(
    (
        "seeing_fwhm",
        "unit",
        "pixel_scale",
        "configured",
        "expected",
    ),
    [
        (0.8, "arcsec", 0.2, ("auto", "auto"), (16, 40)),
        (4.0, "pixels", None, ("auto", "auto"), (16, 40)),
        (0.8, None, 0.2, ("auto", "auto"), (12, 30)),
        (0.8, "arcsec", 0.2, (23, "auto"), (23, 40)),
    ],
    ids=(
        "arcsec-conversion",
        "pixel-value",
        "unresolved-unit-defaults",
        "explicit-value-preserved",
    ),
)
def test_resolve_runtime_config_resolves_psf_candidate_auto_sizes(
    seeing_fwhm,
    unit,
    pixel_scale,
    configured,
    expected,
):
    """Empirical PSF auto sizes must use only explicit FWHM units."""
    header = fits.Header()
    header["FWHM"] = (
        seeing_fwhm,
        "FWHM median",
    )

    if unit is not None:
        header["FWHMUNIT"] = unit

    observation = Observation.from_image(
        ImageData(
            data=np.zeros((21, 31), dtype=float),
            header=header,
        )
    )

    config = {
        "psf": {
            "empirical": {
                "candidate_selection": {
                    "min_half_size": configured[0],
                    "max_half_size": configured[1],
                }
            }
        }
    }

    if pixel_scale is not None:
        config["data"] = {
            "pixel_scale": pixel_scale,
        }

    resolved = resolve_runtime_config(
        config,
        observation,
    )
    candidate = resolved["psf"]["empirical"][
        "candidate_selection"
    ]

    assert (
        candidate["min_half_size"],
        candidate["max_half_size"],
    ) == expected


def test_resolve_config_default_likelihood_omits_normalization_default():
    """The generic merge must not inject a Gaussian-only option."""
    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": "outputs"},
    }

    resolved = resolve_config(config)

    assert resolved["fit"]["likelihood"] == {"type": "gaussian"}


@pytest.mark.parametrize(
    "likelihood_type",
    ["cash", "poisson_gaussian"],
)
def test_resolve_config_does_not_force_normalization_on_other_likelihoods(
    likelihood_type,
):
    """A declared non-Gaussian type must not inherit the Gaussian default."""
    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": "outputs"},
        "fit": {"likelihood": {"type": likelihood_type}},
    }

    resolved = resolve_config(config)

    assert "include_normalization" not in resolved["fit"]["likelihood"]


def test_resolve_config_gaussian_likelihood_builds_from_make_likelihood():
    """The default Gaussian type must still normalize by default downstream."""
    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": "outputs"},
    }

    resolved = resolve_config(config)

    likelihood = make_likelihood(
        resolved["fit"]["likelihood"],
        data=np.zeros((3, 3)),
        noise=1.0,
    )

    assert likelihood.include_normalization is True


@pytest.mark.parametrize(
    "likelihood_type",
    ["cash", "poisson_gaussian"],
)
def test_resolve_config_non_gaussian_likelihood_builds_from_make_likelihood(
    likelihood_type,
):
    """A resolved non-Gaussian config must construct without rejection."""
    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": "outputs"},
        "fit": {"likelihood": {"type": likelihood_type}},
    }

    resolved = resolve_config(config)

    likelihood = make_likelihood(
        resolved["fit"]["likelihood"],
        data=np.ones((3, 3)),
        noise=1.0,
    )

    assert likelihood is not None


def test_resolve_config_correlated_gaussian_likelihood_defaults_unnormalized():
    """The resolved config must preserve the correlated-Gaussian default."""
    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": "outputs"},
        "fit": {"likelihood": {"type": "correlated_gaussian"}},
    }

    resolved = resolve_config(config)

    likelihood = make_likelihood(
        resolved["fit"]["likelihood"],
        data=np.zeros((3, 3)),
        noise=1.0,
        correlation_kernel=np.ones((1, 1)),
        correlation_logdet=0.0,
    )

    assert likelihood.include_normalization is False
