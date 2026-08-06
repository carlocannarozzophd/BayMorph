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


def test_resolved_config_omits_empirical_psf_construction_options() -> None:
    """Resolved configurations must omit obsolete PSF construction options."""
    resolved = resolve_config(_minimal_config())

    empirical = resolved["psf"]["empirical"]

    assert "construction" not in empirical


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

    assert "construction" not in empirical
