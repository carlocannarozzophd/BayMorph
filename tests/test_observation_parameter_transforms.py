"""Tests for observation-specific parameter transformations."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.wcs import WCS

from baymorph.observation.parameter_transforms import (
    SkyEllipseParameterTransform,
)


def _wcs(
    pixel_scale_arcsec: float = 0.2,
) -> WCS:
    wcs = WCS(
        naxis=2
    )
    wcs.wcs.crpix = [
        101.0,
        101.0,
    ]
    wcs.wcs.crval = [
        150.0,
        30.0,
    ]
    wcs.wcs.ctype = [
        "RA---TAN",
        "DEC--TAN",
    ]
    wcs.wcs.cdelt = np.array(
        [
            -pixel_scale_arcsec / 3600.0,
            pixel_scale_arcsec / 3600.0,
        ]
    )

    return wcs


def test_sky_ellipse_transform_produces_detector_parameters():
    transform = SkyEllipseParameterTransform(
        _wcs()
    )

    parameters = transform(
        {
            "ra": 150.0,
            "dec": 30.0,
            "re_arcsec": 1.2,
            "q": 0.6,
            "position_angle": 0.0,
            "flux": 800.0,
            "n": 2.5,
            "sky": 1.0,
        }
    )

    assert parameters["x0"] == pytest.approx(
        100.0,
        abs=1.0e-8,
    )
    assert parameters["y0"] == pytest.approx(
        100.0,
        abs=1.0e-8,
    )
    assert parameters["re"] == pytest.approx(
        6.0,
        rel=1.0e-7,
    )
    assert parameters["q"] == pytest.approx(
        0.6,
        rel=1.0e-7,
    )

    assert parameters["flux"] == pytest.approx(
        800.0
    )
    assert parameters["n"] == pytest.approx(
        2.5
    )
    assert parameters["sky"] == pytest.approx(
        1.0
    )

    assert "ra" not in parameters
    assert "dec" not in parameters
    assert "re_arcsec" not in parameters
    assert "position_angle" not in parameters


def test_sky_ellipse_transform_allows_observation_specific_size():
    g_transform = SkyEllipseParameterTransform(
        _wcs(
            pixel_scale_arcsec=0.2
        )
    )
    r_transform = SkyEllipseParameterTransform(
        _wcs(
            pixel_scale_arcsec=0.4
        )
    )

    common = {
        "ra": 150.0,
        "dec": 30.0,
        "q": 0.7,
        "position_angle": 0.2,
        "flux": 500.0,
        "n": 2.0,
        "sky": 0.0,
    }

    g_parameters = g_transform(
        {
            **common,
            "re_arcsec": 1.0,
        }
    )
    r_parameters = r_transform(
        {
            **common,
            "re_arcsec": 1.6,
        }
    )

    assert g_parameters["re"] == pytest.approx(
        5.0,
        rel=1.0e-7,
    )
    assert r_parameters["re"] == pytest.approx(
        4.0,
        rel=1.0e-7,
    )
