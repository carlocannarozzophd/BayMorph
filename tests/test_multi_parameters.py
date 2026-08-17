"""Tests for model-agnostic multi-observation parameter construction."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from baymorph.pipeline.multi_parameters import (
    make_multi_parameters,
)


def _write_wcs_fits(
    path,
    *,
    pixel_scale: float,
    ra: float = 150.0,
    dec: float = 2.0,
):
    data = np.zeros(
        (32, 32),
        dtype=float,
    )

    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 16.0
    header["CRPIX2"] = 16.0
    header["CRVAL1"] = ra
    header["CRVAL2"] = dec
    header["CDELT1"] = -pixel_scale / 3600.0
    header["CDELT2"] = pixel_scale / 3600.0

    fits.writeto(
        path,
        data,
        header,
    )


def test_single_sersic_parameters_are_built_from_registry(
    tmp_path,
):
    first = tmp_path / "g.fits"
    second = tmp_path / "r.fits"

    _write_wcs_fits(
        first,
        pixel_scale=0.2,
    )

    _write_wcs_fits(
        second,
        pixel_scale=0.4,
        ra=150.00005,
    )

    parameters = make_multi_parameters(
        model="single_sersic",
        observations={
            "g": first,
            "r": second,
        },
        target_centers={
            "g": (15.0, 15.0),
            "r": (15.0, 15.0),
        },
        prior_bounds={
            "flux": {
                "g": (0.0, 1000.0),
                "r": (0.0, 2000.0),
            },
        },
        shared=(
            "q",
            "position_angle",
            "ra",
            "dec",
        ),
    )

    assert tuple(parameters) == (
        "flux",
        "re_arcsec",
        "n",
        "q",
        "position_angle",
        "ra",
        "dec",
        "sky",
    )

    assert parameters["flux"]["coupling"] == "independent"
    assert parameters["re_arcsec"]["coupling"] == "independent"
    assert parameters["n"]["coupling"] == "independent"
    assert parameters["q"]["coupling"] == "shared"
    assert parameters["position_angle"]["coupling"] == "shared"
    assert parameters["ra"]["coupling"] == "shared"
    assert parameters["dec"]["coupling"] == "shared"
    assert parameters["sky"]["coupling"] == "independent"

    assert parameters[
        "re_arcsec"
    ][
        "priors"
    ][
        "g"
    ][
        "lower"
    ] == pytest.approx(
        0.2
    )

    assert parameters[
        "re_arcsec"
    ][
        "priors"
    ][
        "r"
    ][
        "lower"
    ] == pytest.approx(
        0.4
    )


def test_double_sersic_uses_same_generic_builder(
    tmp_path,
):
    first = tmp_path / "g.fits"
    second = tmp_path / "r.fits"

    _write_wcs_fits(
        first,
        pixel_scale=0.2,
    )

    _write_wcs_fits(
        second,
        pixel_scale=0.3,
    )

    parameters = make_multi_parameters(
        model="double_sersic",
        observations={
            "g": first,
            "r": second,
        },
        target_centers={
            "g": (15.0, 15.0),
            "r": (15.0, 15.0),
        },
        prior_bounds={
            "flux_bulge": {
                "g": (0.0, 500.0),
                "r": (0.0, 600.0),
            },
            "flux_disk": {
                "g": (0.0, 1000.0),
                "r": (0.0, 1200.0),
            },
        },
        shared=(
            "q_bulge",
            "q_disk",
            "position_angle_bulge",
            "position_angle_disk",
            "ra",
            "dec",
        ),
    )

    assert set(parameters) == {
        "flux_bulge",
        "re_arcsec_bulge",
        "n_bulge",
        "q_bulge",
        "position_angle_bulge",
        "flux_disk",
        "re_arcsec_disk",
        "n_disk",
        "q_disk",
        "position_angle_disk",
        "ra",
        "dec",
        "sky",
    }

    assert parameters["q_bulge"]["coupling"] == "shared"
    assert parameters["q_disk"]["coupling"] == "shared"

    assert (
        parameters[
            "position_angle_bulge"
        ][
            "coupling"
        ]
        == "shared"
    )

    assert (
        parameters[
            "position_angle_disk"
        ][
            "coupling"
        ]
        == "shared"
    )

    assert parameters[
        "re_arcsec_bulge"
    ][
        "priors"
    ][
        "g"
    ][
        "lower"
    ] == pytest.approx(
        0.2
    )

    assert parameters[
        "re_arcsec_disk"
    ][
        "priors"
    ][
        "r"
    ][
        "lower"
    ] == pytest.approx(
        3.0
    )


def test_missing_standard_prior_requires_explicit_bounds(
    tmp_path,
):
    fits_path = tmp_path / "image.fits"

    _write_wcs_fits(
        fits_path,
        pixel_scale=0.2,
    )

    with pytest.raises(
        ValueError,
        match=r"No standard prior.*flux.*explicit prior_bounds",
    ):
        make_multi_parameters(
            model="single_sersic",
            observations={
                "g": fits_path,
            },
            target_centers={
                "g": (15.0, 15.0),
            },
            shared=(
                "q",
                "position_angle",
                "ra",
                "dec",
            ),
        )


def test_shared_sky_radius_requires_explicit_physical_bounds(
    tmp_path,
):
    fits_path = tmp_path / "image.fits"

    _write_wcs_fits(
        fits_path,
        pixel_scale=0.2,
    )

    with pytest.raises(
        ValueError,
        match="Shared sky-frame radius",
    ):
        make_multi_parameters(
            model="single_sersic",
            observations={
                "g": fits_path,
            },
            target_centers={
                "g": (15.0, 15.0),
            },
            prior_bounds={
                "flux": (0.0, 1000.0),
            },
            shared=(
                "re_arcsec",
                "q",
                "position_angle",
                "ra",
                "dec",
            ),
        )


def test_shared_sky_radius_accepts_explicit_physical_bounds(
    tmp_path,
):
    fits_path = tmp_path / "image.fits"

    _write_wcs_fits(
        fits_path,
        pixel_scale=0.2,
    )

    parameters = make_multi_parameters(
        model="single_sersic",
        observations={
            "g": fits_path,
        },
        target_centers={
            "g": (15.0, 15.0),
        },
        prior_bounds={
            "flux": (0.0, 1000.0),
            "re_arcsec": (0.2, 20.0),
        },
        shared=(
            "re_arcsec",
            "q",
            "position_angle",
            "ra",
            "dec",
        ),
    )

    assert parameters["re_arcsec"] == {
        "coupling": "shared",
        "prior": {
            "type": "uniform",
            "lower": 0.2,
            "upper": 20.0,
        },
    }


def test_unknown_shared_parameter_is_rejected(
    tmp_path,
):
    fits_path = tmp_path / "image.fits"

    _write_wcs_fits(
        fits_path,
        pixel_scale=0.2,
    )

    with pytest.raises(
        ValueError,
        match="outside the model contract",
    ):
        make_multi_parameters(
            model="single_sersic",
            observations={
                "g": fits_path,
            },
            target_centers={
                "g": (15.0, 15.0),
            },
            prior_bounds={
                "flux": (0.0, 1000.0),
            },
            shared=(
                "not_a_parameter",
            ),
        )
