"""Tests for the opening of the image of an observation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import baymorph.pipeline.image_preparation as image_preparation_module
import baymorph.pipeline.opening as opening
import baymorph.pipeline.prepared_identity as prepared_identity
from baymorph.core.observation import Observation
from baymorph.pipeline import pipeline_paths, resolve_config
from baymorph.pipeline.config import resolve_runtime_config
from baymorph.pipeline.prepared_identity import (
    _preparation_config_identity,
    build_prepared_input_identity,
    file_content_identity,
)
from baymorph.pipeline.psf_candidates import (
    build_psf_candidates_provenance,
    save_psf_candidates_provenance,
    validate_psf_candidates_provenance,
)
from baymorph.pipeline.psf_construction import (
    build_empirical_psf_provenance,
    validate_empirical_psf_provenance,
)
from baymorph.runner import BayMorphRunner


@pytest.fixture(autouse=True)
def _fresh_openings():
    """Start and end every test without observations kept for reuse."""
    opening.forget_opened_observations()
    yield
    opening.forget_opened_observations()


def _coadd_header() -> fits.Header:
    """Return the header of a small coadd in ADU/s with a 0.21 arcsec WCS."""
    header = fits.Header()

    for key, value in (
        ("CTYPE1", "RA---TAN"),
        ("CTYPE2", "DEC--TAN"),
        ("CUNIT1", "deg"),
        ("CUNIT2", "deg"),
        ("CRPIX1", 32.5),
        ("CRPIX2", 32.5),
        ("CRVAL1", 201.0),
        ("CRVAL2", -31.5),
        ("CD1_1", -0.21 / 3600.0),
        ("CD1_2", 0.0),
        ("CD2_1", 0.0),
        ("CD2_2", 0.21 / 3600.0),
        ("BUNIT", "ADU/s"),
        ("PHOTZP", 30.0),
        ("GAIN", 34.9),
        ("EXPTIME", 1.0),
    ):
        header[key] = value

    return header


def _write_coadd(path: Path, *, shape: tuple[int, int] = (64, 64)) -> None:
    """Write a deterministic coadd of Gaussian noise."""
    data = np.random.default_rng(20261019).normal(0.0, 1.0, shape)
    fits.writeto(path, data, header=_coadd_header(), overwrite=True)


def _reuse_every_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat every file as large, so that it is read and hashed once."""
    monkeypatch.setattr(opening, "REUSE_MIN_BYTES", 0)
    monkeypatch.setattr(prepared_identity, "REUSE_MIN_BYTES", 0)


def test_identity_of_an_image_without_new_features_is_unchanged(tmp_path):
    """The preparation identity equals the one of the 0.5.1 release.

    The digest was computed with the code of the 0.5.1 release on this
    configuration and this image, so a change of the opening that alters the
    identity of an image using none of its new features fails here.
    """
    image_path = tmp_path / "coadd.fits"
    _write_coadd(image_path)
    config = resolve_config(
        {
            "object": {"name": "identity_golden"},
            "data": {
                "image_path": str(image_path),
                "target_position": {"x": 32, "y": 32},
                "cutout": {"mode": "fixed", "size": 48},
            },
            "outputs": {"root": str(tmp_path / "outputs"), "overwrite": False},
        }
    )
    observation = Observation.from_fits(image_path)
    runtime = resolve_runtime_config(config, observation)

    _, digest = _preparation_config_identity(
        runtime,
        image_path=Path("/synthetic/galaxy.fits"),
        mask_identity=None,
    )

    assert digest == (
        "82aac9b10113b11bdbf3748d2ff1d7013402c226ba75b5bb67953a43f65fa77b"
    )

    identity = build_prepared_input_identity(config)
    fits_identity = identity["fits"]
    contiguous = np.ascontiguousarray(observation.image.data)
    array_digest = hashlib.sha256()
    array_digest.update(
        json.dumps(
            {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    array_digest.update(b"\0")
    array_digest.update(contiguous.tobytes(order="C"))

    assert fits_identity["sha256"] == hashlib.sha256(
        image_path.read_bytes()
    ).hexdigest()
    assert fits_identity["loaded_data_sha256"] == array_digest.hexdigest()
    assert fits_identity["primary_header_sha256"] == hashlib.sha256(
        observation.image.header.tostring(
            sep="\n",
            endcard=False,
            padding=False,
        ).encode("ascii")
    ).hexdigest()
    assert "hdu" not in fits_identity


def test_opened_arrays_are_read_only(tmp_path):
    """A stage cannot change the pixels that other stages read."""
    image_path = tmp_path / "coadd.fits"
    _write_coadd(image_path)

    observation = opening.open_observation(image_path)

    with pytest.raises(ValueError, match="read-only"):
        observation.image.data[0, 0] = 1.0


def test_a_small_file_is_read_at_every_opening(tmp_path):
    """A file below the reuse size is read again at every opening."""
    image_path = tmp_path / "coadd.fits"
    _write_coadd(image_path)

    first = opening.open_observation(image_path)
    second = opening.open_observation(image_path)

    assert first is not second
    assert np.array_equal(first.image.data, second.image.data)


def test_a_large_file_is_read_once_until_it_changes(tmp_path, monkeypatch):
    """A large file is reused while unchanged and read again once changed."""
    _reuse_every_file(monkeypatch)
    image_path = tmp_path / "coadd.fits"
    _write_coadd(image_path)

    first = opening.open_observation(image_path)

    assert opening.open_observation(image_path) is first

    _write_coadd(image_path, shape=(64, 72))
    changed = opening.open_observation(image_path)

    assert changed is not first
    assert changed.image.shape == (64, 72)


def test_an_unchanged_large_file_is_hashed_once(tmp_path, monkeypatch):
    """The digest of a large file is computed once while it is unchanged."""
    _reuse_every_file(monkeypatch)
    image_path = tmp_path / "coadd.fits"
    _write_coadd(image_path)
    hashed = []
    sha256_file = prepared_identity.sha256_file

    def counting_sha256_file(path):
        hashed.append(Path(path))
        return sha256_file(path)

    monkeypatch.setattr(prepared_identity, "sha256_file", counting_sha256_file)

    first = file_content_identity(image_path)
    second = file_content_identity(image_path)

    assert first == second
    assert hashed == [image_path]


def test_preparation_opens_and_hashes_the_fits_once(tmp_path, monkeypatch):
    """One saved preparation reads and hashes a large science file once."""
    from baymorph.pipeline import prepare_real_image_from_config

    _reuse_every_file(monkeypatch)
    yy, xx = np.indices((80, 80))
    image = 100.0 * np.exp(
        -0.5 * ((xx - 40.0) ** 2 + (yy - 40.0) ** 2) / 4.0**2
    )
    image_path = tmp_path / "image.fits"
    fits.writeto(image_path, image)
    config = {
        "object": {"name": "synthetic"},
        "data": {
            "image_path": str(image_path),
            "target_position": {"x": 40, "y": 40},
            "cutout": {"mode": "fixed", "size": 70},
        },
        "preparation": {
            "background": {"box_size": 10, "filter_size": 3},
            "detection": {"n_sigma": 2.0, "n_pixels": 5, "n_levels": 32},
        },
        "outputs": {"root": str(tmp_path / "outputs"), "overwrite": True},
    }
    opened = []
    hashed = []
    fits_open = fits.open
    sha256_file = prepared_identity.sha256_file

    def counting_open(name, *args, **kwargs):
        if Path(name).resolve() == image_path.resolve():
            opened.append(name)
        return fits_open(name, *args, **kwargs)

    def counting_sha256_file(path):
        if Path(path).resolve() == image_path.resolve():
            hashed.append(path)
        return sha256_file(path)

    monkeypatch.setattr(fits, "open", counting_open)
    monkeypatch.setattr(prepared_identity, "sha256_file", counting_sha256_file)
    monkeypatch.setattr(
        image_preparation_module,
        "validate_plateau_cutout_containment",
        lambda **kwargs: None,
    )

    prepare_real_image_from_config(config, save=True)

    assert len(opened) == 1
    assert len(hashed) == 1


def _psf_config(tmp_path: Path, image_path: Path, **data) -> dict:
    """Return a resolved configuration whose PSF products can be fingerprinted."""
    return resolve_config(
        {
            "object": {"name": "psf_target"},
            "data": {
                "image_path": str(image_path),
                "target_position": {"x": 32, "y": 32},
                **data,
            },
            "outputs": {"root": str(tmp_path / "outputs"), "overwrite": False},
        }
    )


def _write_psf_products(config: dict) -> None:
    """Write placeholder candidate and PSF files that provenance hashes."""
    paths = pipeline_paths(config)["psf"]
    paths["dir"].mkdir(parents=True, exist_ok=True)
    paths["candidates"].write_text("[]\n", encoding="utf-8")
    paths["psf_fits"].write_bytes(b"psf")


def test_a_psf_from_another_hdu_is_refused(tmp_path):
    """A PSF built from the default HDU is not reused for a declared one."""
    image_path = tmp_path / "coadd.fits"
    _write_coadd(image_path)
    default = _psf_config(tmp_path, image_path)
    declared = _psf_config(tmp_path, image_path, hdu=0)
    _write_psf_products(default)

    provenance = build_empirical_psf_provenance(default)

    assert "hdu" not in provenance["input_fits"]

    validate_empirical_psf_provenance(default, {"provenance": provenance})

    with pytest.raises(ValueError):
        validate_empirical_psf_provenance(declared, {"provenance": provenance})


def test_candidates_are_refused_after_their_inputs_change(tmp_path):
    """Candidates selected under other settings are not reused."""
    image_path = tmp_path / "coadd.fits"
    _write_coadd(image_path)
    config = _psf_config(tmp_path, image_path)
    _write_psf_products(config)
    save_psf_candidates_provenance(config)

    validate_psf_candidates_provenance(config)

    changed = _psf_config(tmp_path, image_path)
    changed["psf"]["empirical"]["candidate_selection"]["max_area"] = 1500

    with pytest.raises(ValueError, match="selected from another"):
        validate_psf_candidates_provenance(changed)

    other_hdu = _psf_config(tmp_path, image_path, hdu=0)

    assert (
        build_psf_candidates_provenance(other_hdu)
        != build_psf_candidates_provenance(config)
    )


def test_candidates_of_an_earlier_release_need_their_psf(tmp_path):
    """Candidates without provenance are refused when no PSF binds them."""
    image_path = tmp_path / "coadd.fits"
    _write_coadd(image_path)
    config = _psf_config(tmp_path, image_path)
    paths = pipeline_paths(config)["psf"]
    paths["dir"].mkdir(parents=True, exist_ok=True)
    paths["candidates"].write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="record no provenance"):
        BayMorphRunner._validate_candidates_without_provenance(config)
