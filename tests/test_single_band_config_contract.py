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
