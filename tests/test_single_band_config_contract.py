"""Tests for the canonical single-band configuration contract."""

from __future__ import annotations

import pytest

from baymorph.pipeline import resolve_config


def _minimal_config() -> dict:
    """Return the minimum valid single-band configuration."""
    return {
        "object": {
            "name": "test_object",
        },
        "data": {
            "image_path": "data/image.fits",
        },
        "outputs": {
            "root": "runs",
        },
    }


def test_detection_default_uses_canonical_n_pixels() -> None:
    """Resolved configurations must use only the canonical key."""
    resolved = resolve_config(_minimal_config())

    detection = resolved["preparation"]["detection"]

    assert detection["n_pixels"] == 20
    assert "npixels" not in detection


def test_explicit_n_pixels_is_preserved() -> None:
    """User-defined canonical detection settings must be preserved."""
    config = _minimal_config()
    config["preparation"] = {
        "detection": {
            "n_pixels": 11,
        },
    }

    resolved = resolve_config(config)

    assert resolved["preparation"]["detection"]["n_pixels"] == 11
    assert "npixels" not in resolved["preparation"]["detection"]


def test_legacy_npixels_is_rejected() -> None:
    """The obsolete detection alias must fail explicitly."""
    config = _minimal_config()
    config["preparation"] = {
        "detection": {
            "npixels": 11,
        },
    }

    with pytest.raises(
        ValueError,
        match=r"preparation\.detection\.n_pixels",
    ):
        resolve_config(config)


@pytest.mark.parametrize(
    "unsupported_key",
    [
        "noise_map_path",
        "psf_path",
    ],
)
def test_unsupported_external_data_paths_are_rejected(
    unsupported_key,
) -> None:
    """Unimplemented external products must not be silently accepted."""
    config = _minimal_config()
    config["data"][unsupported_key] = "external.fits"

    with pytest.raises(
        ValueError,
        match="not yet supported",
    ):
        resolve_config(config)


def test_declared_calibration_is_checked() -> None:
    """data.calibration accepts the declarable quantities and refuses the rest."""
    config = _minimal_config()
    config["data"]["calibration"] = {
        "image_unit": "electron/s",
        "zeropoint": 25.94,
        "photometric_system": "AB",
        "saturation": 80.0,
    }

    assert resolve_config(config)["data"]["calibration"]["zeropoint"] == 25.94

    config["data"]["calibration"] = {"photometric_system": "AB"}

    with pytest.raises(ValueError, match="data.calibration.photometric_system"):
        resolve_config(config)

    config["data"]["calibration"] = {"seeing": 0.8}

    with pytest.raises(ValueError, match="cannot declare"):
        resolve_config(config)


def test_the_retired_external_mask_path_is_rejected() -> None:
    """data.external_mask_path is refused for every use of a configuration."""
    config = _minimal_config()
    config["data"]["external_mask_path"] = "mask.fits"

    with pytest.raises(ValueError, match="data.mask_path"):
        resolve_config(config)


def test_external_mask_path_remains_supported() -> None:
    """The implemented external-mask configuration must remain valid."""
    config = _minimal_config()
    config["data"]["mask_path"] = "mask.fits"

    resolved = resolve_config(config)

    assert resolved["data"]["mask_path"] == "mask.fits"


@pytest.mark.parametrize(
    ("section", "value"),
    [
        ("background", "empirical"),
        ("noise", "empirical"),
    ],
)
def test_fixed_preparation_method_selectors_are_rejected(
    section,
    value,
) -> None:
    """Fixed preparation algorithms must not appear configurable."""
    config = _minimal_config()
    config["preparation"] = {
        section: {
            "method": value,
        },
    }

    with pytest.raises(
        ValueError,
        match=r"Unsupported fixed configuration entries",
    ):
        resolve_config(config)


def test_resolved_config_omits_fixed_preparation_methods() -> None:
    """Resolved configurations must expose only effective choices."""
    resolved = resolve_config(_minimal_config())

    assert "method" not in resolved["preparation"]["background"]
    assert "method" not in resolved["preparation"]["noise"]


def test_unsupported_masking_section_is_rejected() -> None:
    """Unimplemented masking controls must not be silently accepted."""
    config = _minimal_config()
    config["masking"] = {
        "connected_source": {
            "enabled": False,
        },
    }

    with pytest.raises(
        ValueError,
        match="Unsupported top-level configuration section 'masking'",
    ):
        resolve_config(config)


def test_resolved_config_omits_masking_strategy_controls() -> None:
    """Resolved configurations must contain no ineffective masking section."""
    resolved = resolve_config(_minimal_config())

    assert "masking" not in resolved


def test_unknown_fit_key_is_rejected() -> None:
    """A typo'd fit.* key must not be silently dropped alongside the default."""
    config = _minimal_config()
    config["fit"] = {
        "modles": [
            "single_sersic",
        ],
    }

    with pytest.raises(
        ValueError,
        match=r"Unsupported configuration entries in section 'fit'.*modles",
    ):
        resolve_config(config)


def test_every_declared_fit_key_is_accepted() -> None:
    """Every documented fit.* key must resolve cleanly, individually."""
    for key, value in (
        (
            "models",
            [
                "single_sersic",
            ],
        ),
        (
            "likelihood",
            {
                "type": "gaussian",
            },
        ),
        (
            "sampler",
            {
                "name": "nautilus",
            },
        ),
        (
            "comparison",
            {
                "enabled": False,
            },
        ),
        (
            "background",
            {
                "model": "constant",
            },
        ),
        (
            "priors",
            {},
        ),
    ):
        config = _minimal_config()
        config["fit"] = {
            key: value,
        }

        resolve_config(config)


def test_empirical_psf_auto_size_is_rejected() -> None:
    """Unused empirical PSF sizing controls must not be accepted."""
    config = _minimal_config()
    config["psf"] = {
        "empirical": {
            "construction": {
                "auto_size": True,
            },
        },
    }

    with pytest.raises(
        ValueError,
        match=r"psf\.empirical\.construction\.auto_size",
    ):
        resolve_config(config)


def test_resolved_config_freezes_the_empirical_psf_construction() -> None:
    """Resolved configs freeze the centering gate and the builder radii.

    The radii of the builder of AD-046 enter the PSF identity, so their
    validated values are written into every resolved configuration.
    """
    resolved = resolve_config(_minimal_config())

    empirical = resolved["psf"]["empirical"]

    assert empirical["construction"] == {
        "centroid_tolerance_pixels": 0.05,
        "sky_inner_radius": 80.0,
        "sky_outer_radius": 160.0,
        "mask_growth_pixels": 5,
        "support_half_size": 64,
    }


@pytest.mark.parametrize(
    "construction",
    [
        {"sky_inner_radius": 50.0},
        {"sky_outer_radius": 70.0},
        {"support_half_size": 2.5},
        {"mask_growth_pixels": -1},
        {"sky_inner_radius": float("nan")},
    ],
)
def test_invalid_empirical_psf_builder_radii_are_rejected(construction) -> None:
    config = _minimal_config()
    config["psf"] = {"empirical": {"construction": construction}}

    with pytest.raises(ValueError, match=r"psf\.empirical\.construction"):
        resolve_config(config)


def test_empirical_psf_normalize_option_is_rejected() -> None:
    """PSF normalization must remain an invariant, not a user option."""
    config = _minimal_config()
    config["psf"] = {
        "empirical": {
            "construction": {
                "normalize": False,
            },
        },
    }

    with pytest.raises(
        ValueError,
        match=r"psf\.empirical\.construction\.normalize",
    ):
        resolve_config(config)


def test_resolved_config_omits_empirical_psf_normalize_option() -> None:
    """Resolved configurations must not expose PSF normalization."""
    resolved = resolve_config(_minimal_config())

    empirical = resolved["psf"]["empirical"]

    assert "normalize" not in empirical["construction"]
