"""Tests for model-driven sky-to-detector parameter transforms."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.wcs import WCS

from baymorph.models.geometry import get_model_geometry
from baymorph.observation.model_parameter_transforms import (
    SkyModelParameterTransform,
)


def _wcs() -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [101.0, 101.0]
    wcs.wcs.crval = [150.0, 30.0]
    wcs.wcs.ctype = [
        "RA---TAN",
        "DEC--TAN",
    ]
    wcs.wcs.cdelt = np.array(
        [
            -0.2 / 3600.0,
            0.2 / 3600.0,
        ]
    )
    return wcs


def test_single_sersic_transform_is_geometry_driven():
    transform = SkyModelParameterTransform(
        wcs=_wcs(),
        geometry=get_model_geometry(
            "single_sersic"
        ),
    )

    result = transform(
        {
            "ra": 150.0,
            "dec": 30.0,
            "re_arcsec": 1.0,
            "q": 0.7,
            "position_angle": 0.2,
            "flux": 500.0,
            "n": 2.5,
            "background_level": 1.0,
        }
    )

    assert result["re"] == pytest.approx(
        5.0,
        rel=1.0e-7,
    )
    assert result["x0"] == pytest.approx(
        100.0,
        abs=1.0e-8,
    )
    assert result["y0"] == pytest.approx(
        100.0,
        abs=1.0e-8,
    )
    assert result["flux"] == pytest.approx(
        500.0
    )
    assert result["n"] == pytest.approx(
        2.5
    )


def test_bulge_disk_transform_handles_two_ellipses():
    transform = SkyModelParameterTransform(
        wcs=_wcs(),
        geometry=get_model_geometry(
            "bulge_disk"
        ),
    )

    result = transform(
        {
            "ra": 150.0,
            "dec": 30.0,
            "re_arcsec_bulge": 0.6,
            "q_bulge": 0.8,
            "position_angle_bulge": 0.1,
            "re_arcsec_disk": 1.8,
            "q_disk": 0.5,
            "position_angle_disk": 0.4,
            "flux_bulge": 300.0,
            "n_bulge": 3.0,
            "flux_disk": 700.0,
            "background_level": 0.5,
        }
    )

    assert result["re_bulge"] == pytest.approx(
        3.0,
        rel=1.0e-7,
    )
    assert result["re_disk"] == pytest.approx(
        9.0,
        rel=1.0e-7,
    )

    assert result["x0"] == pytest.approx(
        100.0,
        abs=1.0e-8,
    )
    assert result["y0"] == pytest.approx(
        100.0,
        abs=1.0e-8,
    )

    assert result["flux_bulge"] == pytest.approx(
        300.0
    )
    assert result["n_bulge"] == pytest.approx(
        3.0
    )
    assert result["flux_disk"] == pytest.approx(
        700.0
    )
    assert result["background_level"] == pytest.approx(
        0.5
    )

    assert "ra" not in result
    assert "dec" not in result
    assert "re_arcsec_bulge" not in result
    assert "re_arcsec_disk" not in result
    assert "position_angle_bulge" not in result
    assert "position_angle_disk" not in result


def test_sersic_point_source_uses_shared_transformed_centre():
    transform = SkyModelParameterTransform(
        wcs=_wcs(),
        geometry=get_model_geometry(
            "sersic_point_source"
        ),
    )

    result = transform(
        {
            "ra": 150.0,
            "dec": 30.0,
            "re_arcsec": 1.0,
            "q": 0.7,
            "position_angle": 0.2,
            "flux_sersic": 700.0,
            "n": 2.0,
            "flux_point_source": 100.0,
            "background_level": 0.0,
        }
    )

    assert result["x0"] == pytest.approx(
        100.0,
        abs=1.0e-8,
    )
    assert result["y0"] == pytest.approx(
        100.0,
        abs=1.0e-8,
    )

    assert result["flux_point_source"] == pytest.approx(
        100.0
    )


def test_double_sersic_transform_handles_two_ellipses():
    transform = SkyModelParameterTransform(
        wcs=_wcs(),
        geometry=get_model_geometry(
            "double_sersic"
        ),
    )

    result = transform(
        {
            "ra": 150.0,
            "dec": 30.0,
            "re_arcsec_bulge": 0.8,
            "q_bulge": 0.75,
            "position_angle_bulge": 0.1,
            "re_arcsec_disk": 2.0,
            "q_disk": 0.55,
            "position_angle_disk": 0.35,
            "flux_bulge": 300.0,
            "n_bulge": 3.5,
            "flux_disk": 700.0,
            "n_disk": 1.2,
            "background_level": 0.5,
        }
    )

    assert result["re_bulge"] == pytest.approx(
        4.0,
        rel=1.0e-7,
    )
    assert result["re_disk"] == pytest.approx(
        10.0,
        rel=1.0e-7,
    )

    assert result["x0"] == pytest.approx(
        100.0,
        abs=1.0e-8,
    )
    assert result["y0"] == pytest.approx(
        100.0,
        abs=1.0e-8,
    )

    assert result["flux_bulge"] == pytest.approx(
        300.0
    )
    assert result["n_bulge"] == pytest.approx(
        3.5
    )
    assert result["flux_disk"] == pytest.approx(
        700.0
    )
    assert result["n_disk"] == pytest.approx(
        1.2
    )
