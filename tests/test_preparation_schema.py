"""The configuration table of the new preparation.

Every key of the preparation sections is checked when a configuration is
read: unknown and retired keys are refused with the valid names or the
remedy, values outside their domain are refused with their key, defaults are
filled and their source recorded, and a valid 0.6 configuration keeps the
values it had.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from baymorph.pipeline.config import resolve_config
from baymorph.preparation.schema import (
    DEFAULT,
    USER,
    describe_keys,
    read_preparation_settings,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "simulated_galaxy.yaml"


def _config(**sections) -> dict:
    config = {
        "object": {"name": "galaxy"},
        "data": {"image_path": "data/galaxy.fits", "cutout": {"size": 201}},
        "outputs": {"root": "runs"},
    }

    for name, value in sections.items():
        config[name] = value

    return config


def _with(path: str, value) -> dict:
    """Return the minimal configuration with one entry set by dotted path."""
    config = _config()
    parts = path.split(".")
    target = config

    for part in parts[:-1]:
        target = target.setdefault(part, {})

    target[parts[-1]] = value
    return config


def test_a_minimal_configuration_gets_every_default_with_its_source():
    settings = read_preparation_settings(_config())

    assert settings.get("data.cutout.mode") == "fixed"
    assert settings.get("data.mask_mode") == "union"
    assert settings.get("preparation.masking.contaminant_growth_pixels") == 20
    assert settings.get("preparation.detection.n_pixels") == 20
    assert settings.get("psf.type") == "empirical"
    assert settings.get("outputs.overwrite") is False
    # The two thresholds of the cutout stage, in sky scales.
    assert settings.get("data.cutout.light_threshold") == 1.0
    assert settings.get("preparation.noise.light_threshold") == 0.3

    assert settings.sources["data.image_path"] == USER
    assert settings.sources["data.cutout.size"] == USER
    assert settings.sources["data.cutout.mode"] == DEFAULT
    assert settings.sources["psf.empirical.construction.support_half_size"] == DEFAULT


def test_every_key_of_the_table_is_resolved_and_has_a_source():
    settings = read_preparation_settings(_config())
    keys = [row["key"] for row in describe_keys()]

    assert len(keys) == len(set(keys))
    assert sorted(settings.sources) == sorted(keys)

    for key in keys:
        settings.get(key)


def test_the_cutout_is_automatic_when_no_size_is_declared():
    config = _config()
    config["data"]["cutout"] = {}

    assert read_preparation_settings(config).get("data.cutout.mode") == "auto"


def test_a_fixed_cutout_without_a_size_is_refused():
    config = _config()
    config["data"]["cutout"] = {"mode": "fixed"}

    with pytest.raises(ValueError, match="data.cutout.size"):
        read_preparation_settings(config)


def test_a_curated_mask_used_as_it_is_is_not_grown_by_default():
    config = _config()
    config["data"].update({"mask_path": "data/mask.fits", "mask_mode": "exact"})

    settings = read_preparation_settings(config)

    assert settings.get("preparation.masking.contaminant_growth_pixels") == 0


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("data.cutout.sizes", 201, "Available entries"),
        ("data.target", {"x": 1, "y": 2}, "Available entries"),
        ("preparation.detection.n_sgima", 3.0, "Available entries"),
        ("preparation.masking.growth", 20, "Available entries"),
        ("psf.empirical.candidate", {}, "Available entries"),
        ("psf.gaussian.sigma", 1.0, "Available entries"),
        ("outputs.namespace", "x", "Available entries"),
        ("object.ra", 10.0, "Available entries"),
    ],
)
def test_unknown_keys_are_refused_with_the_valid_names(path, value, message):
    with pytest.raises(ValueError, match=message):
        read_preparation_settings(_with(path, value))


def test_unknown_and_retired_sections_are_refused():
    with pytest.raises(ValueError, match="Available sections"):
        read_preparation_settings(_config(preparations={}))

    with pytest.raises(ValueError, match="preparation.masking"):
        read_preparation_settings(_config(masking={}))

    with pytest.raises(ValueError, match="joint path"):
        read_preparation_settings(_config(observations=[]))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("preparation.detection.npixels", 5, "n_pixels"),
        ("data.noise_map_path", "noise.fits", "weight map"),
        ("data.external_mask_path", "mask.fits", "data.mask_path"),
        ("preparation.background.method", "mesh", "empirically"),
        ("preparation.noise.method", "mad", "AD-041"),
        ("psf.empirical.construction.normalize", True, "unit sum"),
        ("psf.empirical.construction.auto_size", True, "cutout limits"),
    ],
)
def test_retired_keys_are_refused_with_their_remedy(path, value, message):
    with pytest.raises(ValueError, match=message):
        read_preparation_settings(_with(path, value))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("data.pixel_scale", -0.2),
        ("data.pixel_scale", 0.0),
        ("data.pixel_scale", "0.2"),
        ("data.hdu", -1),
        ("data.hdu", True),
        ("data.target_position", "middle"),
        ("data.target_position", [1.0]),
        ("data.target_position", {"x": 1.0}),
        ("data.target_position", {"x": float("nan"), "y": 1.0}),
        ("data.target_position", {"ra": 10.0, "dec": 1.0}),
        ("data.region", {"size": -3}),
        ("data.mask_mode", "intersection"),
        ("data.cutout.size", 0),
        ("data.cutout.size", 201.5),
        ("data.cutout.initial_factor", 1.0),
        ("data.cutout.sky_margin", -1.0),
        ("data.cutout.light_threshold", 0.0),
        ("data.cutout.light_threshold", float("nan")),
        ("data.cutout.light_threshold", True),
        ("preparation.noise.light_threshold", -0.3),
        ("preparation.noise.light_threshold", float("inf")),
        ("outputs.overwrite", "yes please"),
        ("preparation.detection.n_sigma", 0.0),
        ("preparation.detection.n_levels", True),
        ("preparation.masking.negative_threshold_sigma", -5.0),
        ("preparation.masking.contaminant_rule", "every_segment"),
        ("preparation.masking.contaminant_growth_pixels", 2.5),
        ("preparation.noise.sky_correlation", "high_pass"),
        ("psf.type", "moffat"),
        ("psf.empirical.candidate_selection.max_peak", "auto"),
        ("psf.empirical.candidate_selection.max_eccentricity", 1.5),
        ("psf.empirical.candidate_selection.point_source_screen", "stars"),
        ("psf.empirical.candidate_selection.galaxy_exclusion_box", {"xmin": 1}),
        ("psf.empirical.construction.centroid_tolerance_pixels", 0.0),
    ],
)
def test_values_outside_their_domain_are_refused_with_their_key(path, value):
    with pytest.raises(ValueError) as error:
        read_preparation_settings(_with(path, value))

    assert path in str(error.value)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"data.mask_mode": "exact"}, "mask_path"),
        (
            {
                "data.mask_path": "mask.fits",
                "data.mask_mode": "replace",
                "preparation.masking.contaminant_growth_pixels": 20,
            },
            "set it to 0",
        ),
        (
            {
                "preparation.noise.sky_correlation": "field_separated",
                "preparation.masking.contaminant_growth_pixels": 10,
            },
            "field_separated",
        ),
        ({"psf.type": "gaussian"}, "psf.gaussian.fwhm"),
        (
            {
                "psf.empirical.candidate_selection.min_peak": 100.0,
                "psf.empirical.candidate_selection.max_peak": 50.0,
            },
            "no star can be selected",
        ),
        ({"psf.empirical.construction.sky_inner_radius": 50.0}, "support_half_size"),
        ({"data.cutout.min_size": 300, "data.cutout.max_size": 200}, "min_size"),
        (
            {
                "psf.empirical.candidate_selection.min_half_size": 40,
                "psf.empirical.candidate_selection.max_half_size": 20,
            },
            "min_half_size",
        ),
    ],
)
def test_inconsistent_keys_are_refused(changes, message):
    config = _config()

    for path, value in changes.items():
        parts = path.split(".")
        target = config

        for part in parts[:-1]:
            target = target.setdefault(part, {})

        target[parts[-1]] = value

    with pytest.raises(ValueError, match=message):
        read_preparation_settings(config)


def test_values_are_returned_in_canonical_form():
    config = _config()
    config["data"].update(
        {
            "target_position": [100, 120.5],
            "mask_mode": " Union ",
            "region": "whole",
            "pixel_scale": "auto",
            "hdu": "SCI",
        }
    )
    config["psf"] = {"type": "Gaussian", "gaussian": {"fwhm": 2}}

    settings = read_preparation_settings(config)

    assert settings.get("data.target_position") == {"x": 100.0, "y": 120.5}
    assert settings.get("data.mask_mode") == "union"
    assert settings.get("data.region") == "whole"
    assert settings.get("data.pixel_scale") == "auto"
    assert settings.get("data.hdu") == "SCI"
    assert settings.get("psf.type") == "gaussian"
    assert settings.get("psf.gaussian.fwhm") == 2.0


def test_null_disables_the_negative_pixel_criterion():
    settings = read_preparation_settings(
        _with("preparation.masking.negative_threshold_sigma", None)
    )

    assert settings.get("preparation.masking.negative_threshold_sigma") is None
    assert settings.sources["preparation.masking.negative_threshold_sigma"] == USER


def test_missing_required_entries_are_refused():
    config = _config()
    del config["outputs"]

    with pytest.raises(ValueError, match="Missing required config sections"):
        read_preparation_settings(config)

    config = _config()
    del config["data"]["image_path"]

    with pytest.raises(ValueError, match="image_path"):
        read_preparation_settings(config)


def test_the_resolved_values_are_read_only_and_copies_are_mutable():
    settings = read_preparation_settings(_config())

    with pytest.raises(TypeError):
        settings.values["data"]["mask_mode"] = "exact"

    copied = settings.as_dict()
    copied["data"]["mask_mode"] = "exact"

    assert settings.get("data.mask_mode") == "union"


def test_the_input_configuration_is_not_modified():
    config = _config()
    before = copy.deepcopy(config)

    read_preparation_settings(config)

    assert config == before


def test_the_example_configuration_keeps_its_0_6_values():
    raw = yaml.safe_load(EXAMPLE.read_text())
    old = resolve_config(copy.deepcopy(raw))
    new = read_preparation_settings(raw).as_dict()

    def leaves(mapping, prefix=""):
        for key, value in mapping.items():
            name = f"{prefix}{key}"
            if isinstance(value, dict):
                yield from leaves(value, f"{name}.")
            else:
                yield name, value

    for section in ("object", "data", "outputs", "preparation", "psf"):
        old_leaves = dict(leaves(old.get(section, {}), f"{section}."))
        new_leaves = dict(leaves(new[section], f"{section}."))

        assert set(old_leaves) <= set(new_leaves)

        for name, value in old_leaves.items():
            assert new_leaves[name] == value, name
