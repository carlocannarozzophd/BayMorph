"""Tests for immutable image-preparation input identity."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import baymorph.core.correlated_noise as correlated_noise
import baymorph.pipeline.noise_calibration as noise_calibration
from baymorph.core.image import ImageData
from baymorph.core.observation import Observation
from baymorph.pipeline.config import resolve_config
from baymorph.pipeline.noise_calibration import (
    _correlated_noise_calibration_generator_identity,
)
from baymorph.pipeline.prepared_identity import (
    PREPARED_INPUT_IDENTITY_VERSION,
    build_prepared_input_identity,
    sha256_file,
    validate_generator_identity,
    validate_prepared_input_identity,
)


def _science_header() -> fits.Header:
    """Return a small header with a valid celestial WCS."""
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRPIX1"] = 5.5
    header["CRPIX2"] = 5.5
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 2.0
    header["CD1_1"] = -0.21 / 3600.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 0.21 / 3600.0
    header["FILTER"] = "g_SDSS"
    header["BUNIT"] = "ADU/s"
    return header


def _write_science_fits(path, *, value_offset: float = 0.0) -> None:
    """Write deterministic science pixels and WCS metadata."""
    data = np.arange(100, dtype=float).reshape(10, 10) + value_offset
    fits.writeto(
        path,
        data,
        header=_science_header(),
        overwrite=True,
    )


def _write_mask(path, *, excluded_pixel: tuple[int, int]) -> None:
    """Write one deterministic curated exclusion mask."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[excluded_pixel] = 1
    fits.writeto(path, mask, overwrite=True)


def _config(tmp_path, image_path, *, mask_path=None):
    """Return a resolved configuration with stable preparation settings."""
    data = {
        "image_path": str(image_path),
        "target_position": {"x": 5, "y": 5},
        "cutout": {"mode": "fixed", "size": 10},
    }

    if mask_path is not None:
        data["mask_path"] = str(mask_path)
        data["mask_mode"] = "exact"

    return resolve_config(
        {
            "object": {"name": "identity_target"},
            "data": data,
            "preparation": {
                "background": {
                    "box_size": 10,
                    "filter_size": 3,
                },
                "detection": {
                    "n_sigma": 3.0,
                    "n_pixels": 5,
                    "n_levels": 32,
                },
            },
            "outputs": {
                "root": str(tmp_path / "outputs"),
                "overwrite": True,
            },
        }
    )


def test_prepared_input_identity_is_stable_and_validates(tmp_path):
    """Unchanged FITS, WCS, mask, and preparation config must validate."""
    image_path = tmp_path / "source.fits"
    _write_science_fits(image_path)
    config = _config(tmp_path, image_path)
    observation = Observation.from_fits(image_path)

    first = build_prepared_input_identity(
        config,
        observation=observation,
    )
    second = build_prepared_input_identity(config)

    assert first == second
    assert first["schema_version"] == PREPARED_INPUT_IDENTITY_VERSION
    assert len(first["fits"]["sha256"]) == 64
    assert len(first["fits"]["primary_header_sha256"]) == 64
    assert len(first["fits"]["celestial_wcs_sha256"]) == 64
    assert len(first["preparation_config_sha256"]) == 64
    assert first["curated_mask"] is None
    assert first["generator"]

    assert (
        validate_prepared_input_identity(
            first,
            config,
            observation=observation,
        )
        is None
    )


def test_validation_rejects_fits_mutation(tmp_path):
    """A pixel change after identity capture must invalidate the source."""
    image_path = tmp_path / "source.fits"
    _write_science_fits(image_path)
    config = _config(tmp_path, image_path)
    identity = build_prepared_input_identity(config)

    _write_science_fits(image_path, value_offset=42.0)

    with pytest.raises(RuntimeError, match=r"immutable inputs .*fits"):
        validate_prepared_input_identity(identity, config)


def test_validation_rejects_changed_preparation_configuration(tmp_path):
    """Preparation settings are part of the immutable scientific input."""
    image_path = tmp_path / "source.fits"
    _write_science_fits(image_path)
    config = _config(tmp_path, image_path)
    identity = build_prepared_input_identity(config)
    changed = copy.deepcopy(config)
    changed["preparation"]["detection"]["n_sigma"] = 4.0

    with pytest.raises(RuntimeError, match="preparation_config_sha256"):
        validate_prepared_input_identity(identity, changed)


def test_validation_rejects_curated_mask_mutation(tmp_path):
    """A curated-mask pixel change must invalidate the preparation input."""
    image_path = tmp_path / "source.fits"
    mask_path = tmp_path / "curated_mask.fits"
    _write_science_fits(image_path)
    _write_mask(mask_path, excluded_pixel=(2, 3))
    config = _config(tmp_path, image_path, mask_path=mask_path)
    identity = build_prepared_input_identity(config)

    _write_mask(mask_path, excluded_pixel=(4, 5))

    with pytest.raises(RuntimeError, match="curated_mask"):
        validate_prepared_input_identity(identity, config)


def test_fit_only_configuration_change_preserves_preparation_identity(
    tmp_path,
):
    """Sampler settings must not invalidate unchanged prepared inputs."""
    image_path = tmp_path / "source.fits"
    _write_science_fits(image_path)
    config = _config(tmp_path, image_path)
    identity = build_prepared_input_identity(config)
    changed = copy.deepcopy(config)
    changed["fit"]["sampler"]["run"]["n_eff"] = 1234

    assert validate_prepared_input_identity(identity, changed) is None


def test_fit_only_source_change_preserves_producer_provenance(
    tmp_path,
    monkeypatch,
):
    """Inference changes must not invalidate immutable prepared products."""
    image_path = tmp_path / "source.fits"
    _write_science_fits(image_path)
    config = _config(tmp_path, image_path)
    identity = build_prepared_input_identity(config)

    monkeypatch.setattr(
        "baymorph.pipeline.prepared_identity._package_generator_identity",
        lambda: {
            "package": "baymorph",
            "package_version": "0.3.1",
            "python_source_count": 999,
            "python_source_manifest_sha256": "different-consumer-source",
        },
    )

    assert validate_prepared_input_identity(identity, config) is None


def test_validation_requires_saved_generator_provenance(tmp_path):
    """A preparation bundle must retain its producer identity."""
    image_path = tmp_path / "source.fits"
    _write_science_fits(image_path)
    config = _config(tmp_path, image_path)
    identity = build_prepared_input_identity(config)
    del identity["generator"]

    with pytest.raises(RuntimeError, match="generator provenance"):
        validate_prepared_input_identity(identity, config)


def test_build_rejects_supplied_observation_without_source_path(tmp_path):
    """An in-memory Observation cannot attest which FITS bytes it represents."""
    image_path = tmp_path / "source.fits"
    _write_science_fits(image_path)
    config = _config(tmp_path, image_path)
    disk_observation = Observation.from_fits(image_path)
    detached = Observation.from_image(
        ImageData(
            data=disk_observation.image.data.copy(),
            header=disk_observation.image.header.copy(),
            path=None,
        )
    )

    with pytest.raises(ValueError, match="source FITS path"):
        build_prepared_input_identity(
            config,
            observation=detached,
        )


def test_build_rejects_supplied_observation_from_different_path(tmp_path):
    """An Observation loaded from another FITS must fail even if shapes match."""
    image_path = tmp_path / "source.fits"
    other_path = tmp_path / "other.fits"
    _write_science_fits(image_path)
    _write_science_fits(other_path)
    config = _config(tmp_path, image_path)
    other_observation = Observation.from_fits(other_path)

    with pytest.raises(
        ValueError,
        match="does not identify the configured FITS input",
    ):
        build_prepared_input_identity(
            config,
            observation=other_observation,
        )


def test_build_rejects_the_retired_external_mask_path(tmp_path):
    """The retired key is refused, never read as a second curated mask."""
    image_path = tmp_path / "source.fits"
    primary_mask_path = tmp_path / "primary_mask.fits"
    legacy_mask_path = tmp_path / "legacy_mask.fits"
    _write_science_fits(image_path)
    _write_mask(primary_mask_path, excluded_pixel=(2, 3))
    _write_mask(legacy_mask_path, excluded_pixel=(4, 5))
    config = _config(
        tmp_path,
        image_path,
        mask_path=primary_mask_path,
    )
    config["data"]["external_mask_path"] = str(legacy_mask_path)

    with pytest.raises(
        ValueError,
        match=r"external_mask_path is no longer supported",
    ):
        build_prepared_input_identity(config)


def test_file_digests_are_the_digests_of_the_file_bytes(tmp_path):
    """Saved products compare these digests, so their meaning must not change.

    The correlated-noise calibration and the Vecchia factorization record the
    SHA-256 of the bytes of their source and input files; a file larger than
    one read block checks that the digest covers every byte.
    """
    path = tmp_path / "blocks.bin"
    payload = np.random.default_rng(20261019).bytes(5 * 1024**2 + 17)
    path.write_bytes(payload)

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()

    identity = _correlated_noise_calibration_generator_identity()

    assert identity["core_source_sha256"] == hashlib.sha256(
        Path(correlated_noise.__file__).read_bytes()
    ).hexdigest()
    assert identity["pipeline_source_sha256"] == hashlib.sha256(
        Path(noise_calibration.__file__).read_bytes()
    ).hexdigest()


_CURRENT_GENERATOR = {
    "core_source_sha256": "a" * 64,
    "package_version": "0.5.0",
}


def test_generator_identity_ignores_only_its_provenance_fields():
    """A new version label alone must not invalidate a derived product."""
    validate_generator_identity(
        {**_CURRENT_GENERATOR, "package_version": "0.3.0"},
        _CURRENT_GENERATOR,
        provenance_fields={"package_version"},
        product="Test product",
    )

    with pytest.raises(
        ValueError,
        match=r"Test product was made by .*\(different core_source_sha256\)",
    ):
        validate_generator_identity(
            {**_CURRENT_GENERATOR, "core_source_sha256": "b" * 64},
            _CURRENT_GENERATOR,
            provenance_fields={"package_version"},
            product="Test product",
        )


@pytest.mark.parametrize(
    "saved",
    [
        None,
        ["core_source_sha256", "package_version"],
        {"core_source_sha256": "a" * 64},
        {**_CURRENT_GENERATOR, "extra": 1},
    ],
)
def test_generator_identity_must_hold_exactly_the_current_fields(saved):
    with pytest.raises(ValueError, match="incomplete or unknown generator"):
        validate_generator_identity(
            saved,
            _CURRENT_GENERATOR,
            provenance_fields={"package_version"},
            product="Test product",
        )


def test_generator_identity_rejects_unknown_provenance_fields():
    with pytest.raises(ValueError, match="not fields of its generator"):
        validate_generator_identity(
            _CURRENT_GENERATOR,
            _CURRENT_GENERATOR,
            provenance_fields={"pakage_version"},
            product="Test product",
        )
