"""Tests for FITS-to-YAML BayMorph configuration generation."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from astropy.io import fits

from baymorph.pipeline.config import (
    resolve_config,
)
from baymorph.pipeline.config_maker import (
    make_config_from_fits,
)


def _write_test_fits(
    path,
    *,
    with_metadata: bool = True,
):
    data = np.zeros(
        (
            21,
            31,
        ),
        dtype=float,
    )

    header = fits.Header()

    if with_metadata:
        header[
            "CTYPE1"
        ] = "RA---TAN"
        header[
            "CTYPE2"
        ] = "DEC--TAN"
        header[
            "CRPIX1"
        ] = 16.0
        header[
            "CRPIX2"
        ] = 11.0
        header[
            "CRVAL1"
        ] = 150.0
        header[
            "CRVAL2"
        ] = 2.0
        header[
            "CDELT1"
        ] = -0.2 / 3600.0
        header[
            "CDELT2"
        ] = 0.2 / 3600.0
        header[
            "FWHM"
        ] = 0.8
        header[
            "SATLEVEL"
        ] = 50000.0

    fits.writeto(
        path,
        data,
        header,
    )


def test_config_maker_builds_minimal_runnable_config(
    tmp_path,
):
    fits_path = (
        tmp_path
        / "galaxy_A.fits"
    )

    yaml_path = (
        tmp_path
        / "configs"
        / "galaxy_A.yaml"
    )

    _write_test_fits(
        fits_path
    )

    config = make_config_from_fits(
        fits_path,
        target_position=(
            15.0,
            10.0,
        ),
        output_path=yaml_path,
    )

    assert config[
        "object"
    ][
        "name"
    ] == "galaxy_A"

    assert config[
        "data"
    ][
        "image_path"
    ] == str(
        fits_path
    )

    assert config[
        "data"
    ][
        "pixel_scale"
    ] == pytest.approx(
        0.2
    )

    assert config[
        "data"
    ][
        "target_position"
    ] == {
        "x": 15.0,
        "y": 10.0,
    }

    assert config[
        "data"
    ][
        "cutout"
    ][
        "mode"
    ] == "auto"

    assert config[
        "psf"
    ][
        "type"
    ] == "empirical"

    candidate = config[
        "psf"
    ][
        "empirical"
    ][
        "candidate_selection"
    ]

    assert candidate[
        "auto"
    ] is True

    assert candidate[
        "min_half_size"
    ] == "auto"

    assert candidate[
        "max_half_size"
    ] == "auto"

    assert candidate[
        "max_peak"
    ] == "auto"

    resolved = resolve_config(
        config
    )

    assert resolved[
        "fit"
    ][
        "models"
    ]

    assert resolved[
        "diagnostics"
    ][
        "corner"
    ] is True

    assert yaml_path.exists()

    with yaml_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        loaded = yaml.safe_load(
            stream
        )

    assert loaded == config


def test_config_maker_center_is_explicit(
    tmp_path,
):
    fits_path = (
        tmp_path
        / "centered.fits"
    )

    _write_test_fits(
        fits_path
    )

    config = make_config_from_fits(
        fits_path,
        target_position="center",
    )

    assert config[
        "data"
    ][
        "target_position"
    ] == "center"


def test_config_maker_requires_target_seed(
    tmp_path,
):
    fits_path = (
        tmp_path
        / "galaxy.fits"
    )

    _write_test_fits(
        fits_path
    )

    with pytest.raises(
        ValueError,
        match="target_position is required",
    ):
        make_config_from_fits(
            fits_path
        )


def test_config_maker_fixed_cutout_requires_size(
    tmp_path,
):
    fits_path = (
        tmp_path
        / "galaxy.fits"
    )

    _write_test_fits(
        fits_path
    )

    with pytest.raises(
        ValueError,
        match="fixed_size is required",
    ):
        make_config_from_fits(
            fits_path,
            target_position="center",
            cutout_mode="fixed",
        )


def test_config_maker_builds_fixed_cutout(
    tmp_path,
):
    fits_path = (
        tmp_path
        / "galaxy.fits"
    )

    _write_test_fits(
        fits_path
    )

    config = make_config_from_fits(
        fits_path,
        target_position="center",
        cutout_mode="fixed",
        fixed_size=17,
    )

    assert config[
        "data"
    ][
        "cutout"
    ] == {
        "mode": "fixed",
        "size": 17,
    }


def test_config_maker_can_derive_gaussian_psf_pixels_from_header(
    tmp_path,
):
    fits_path = (
        tmp_path
        / "galaxy.fits"
    )

    _write_test_fits(
        fits_path
    )

    config = make_config_from_fits(
        fits_path,
        target_position="center",
        psf_type="gaussian",
    )

    assert config[
        "psf"
    ][
        "gaussian"
    ][
        "fwhm"
    ] == pytest.approx(
        4.0
    )


def test_config_maker_gaussian_psf_requires_scale_information(
    tmp_path,
):
    fits_path = (
        tmp_path
        / "galaxy.fits"
    )

    _write_test_fits(
        fits_path,
        with_metadata=False,
    )

    with pytest.raises(
        ValueError,
        match="Gaussian PSF generation requires",
    ):
        make_config_from_fits(
            fits_path,
            target_position="center",
            psf_type="gaussian",
        )


def test_config_maker_accepts_explicit_gaussian_fwhm(
    tmp_path,
):
    fits_path = (
        tmp_path
        / "galaxy.fits"
    )

    _write_test_fits(
        fits_path,
        with_metadata=False,
    )

    config = make_config_from_fits(
        fits_path,
        target_position="center",
        psf_type="gaussian",
        gaussian_fwhm_pixels=2.75,
    )

    assert config[
        "psf"
    ][
        "gaussian"
    ][
        "fwhm"
    ] == pytest.approx(
        2.75
    )
