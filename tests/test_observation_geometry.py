"""Tests for local affine morphology transformations."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.observation.geometry import (
    ellipse_quadratic_form,
    quadratic_form_to_ellipse,
    transform_quadratic_form,
)


def _axis_angle_difference(
    first: float,
    second: float,
) -> float:
    """Return the absolute difference between unoriented axes."""
    return abs(
        (
            first
            - second
            + 0.5 * np.pi
        )
        % np.pi
        - 0.5 * np.pi
    )


def test_ellipse_quadratic_form_round_trip():
    expected_re = 5.0
    expected_q = 0.6
    expected_theta = 0.37

    matrix = ellipse_quadratic_form(
        re=expected_re,
        q=expected_q,
        theta=expected_theta,
    )

    re, q, theta = quadratic_form_to_ellipse(
        matrix
    )

    assert re == pytest.approx(
        expected_re
    )
    assert q == pytest.approx(
        expected_q
    )
    assert _axis_angle_difference(
        theta,
        expected_theta,
    ) == pytest.approx(
        0.0,
        abs=1.0e-12,
    )


def test_identity_transformation_preserves_geometry():
    matrix = ellipse_quadratic_form(
        re=4.0,
        q=0.7,
        theta=-0.4,
    )

    transformed = transform_quadratic_form(
        matrix,
        np.eye(2),
    )

    assert transformed == pytest.approx(
        matrix
    )


def test_isotropic_scale_changes_radius_only():
    matrix = ellipse_quadratic_form(
        re=3.0,
        q=0.5,
        theta=0.2,
    )

    transformed = transform_quadratic_form(
        matrix,
        2.0 * np.eye(2),
    )

    re, q, theta = quadratic_form_to_ellipse(
        transformed
    )

    assert re == pytest.approx(
        6.0
    )
    assert q == pytest.approx(
        0.5
    )
    assert _axis_angle_difference(
        theta,
        0.2,
    ) == pytest.approx(
        0.0,
        abs=1.0e-12,
    )


def test_rotation_rotates_major_axis():
    phi = 0.4

    rotation = np.array(
        [
            [
                np.cos(phi),
                -np.sin(phi),
            ],
            [
                np.sin(phi),
                np.cos(phi),
            ],
        ]
    )

    matrix = ellipse_quadratic_form(
        re=4.0,
        q=0.6,
        theta=0.2,
    )

    transformed = transform_quadratic_form(
        matrix,
        rotation,
    )

    re, q, theta = quadratic_form_to_ellipse(
        transformed
    )

    assert re == pytest.approx(
        4.0
    )
    assert q == pytest.approx(
        0.6
    )
    assert _axis_angle_difference(
        theta,
        0.6,
    ) == pytest.approx(
        0.0,
        abs=1.0e-12,
    )


def test_anisotropic_scale_changes_ellipse_geometry():
    circular = ellipse_quadratic_form(
        re=2.0,
        q=1.0,
        theta=0.0,
    )

    transformed = transform_quadratic_form(
        circular,
        np.array(
            [
                [2.0, 0.0],
                [0.0, 1.0],
            ]
        ),
    )

    re, q, theta = quadratic_form_to_ellipse(
        transformed
    )

    assert re == pytest.approx(
        4.0
    )
    assert q == pytest.approx(
        0.5
    )
    assert _axis_angle_difference(
        theta,
        0.0,
    ) == pytest.approx(
        0.0,
        abs=1.0e-12,
    )


def test_transform_rejects_singular_jacobian():
    matrix = ellipse_quadratic_form(
        re=2.0,
        q=0.8,
        theta=0.0,
    )

    with pytest.raises(
        ValueError,
        match="invertible",
    ):
        transform_quadratic_form(
            matrix,
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 0.0],
                ]
            ),
        )


def test_transformed_quadratic_form_preserves_ellipse_points():
    source = ellipse_quadratic_form(
        re=5.0,
        q=0.55,
        theta=0.31,
    )

    jacobian = np.array(
        [
            [1.8, 0.35],
            [-0.2, 0.9],
        ],
        dtype=float,
    )

    transformed = transform_quadratic_form(
        source,
        jacobian,
    )

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        64,
        endpoint=False,
    )

    rotation = np.array(
        [
            [
                np.cos(0.31),
                -np.sin(0.31),
            ],
            [
                np.sin(0.31),
                np.cos(0.31),
            ],
        ],
        dtype=float,
    )

    for angle in angles:
        principal_point = np.array(
            [
                5.0 * np.cos(angle),
                0.55 * 5.0 * np.sin(angle),
            ],
            dtype=float,
        )

        source_point = (
            rotation
            @ principal_point
        )

        detector_point = (
            jacobian
            @ source_point
        )

        source_radius_squared = float(
            source_point.T
            @ source
            @ source_point
        )

        detector_radius_squared = float(
            detector_point.T
            @ transformed
            @ detector_point
        )

        assert source_radius_squared == pytest.approx(
            1.0,
            abs=1.0e-12,
        )

        assert detector_radius_squared == pytest.approx(
            source_radius_squared,
            abs=1.0e-12,
        )


def test_general_affine_quadratic_form_round_trip():
    source = ellipse_quadratic_form(
        re=6.0,
        q=0.43,
        theta=-0.52,
    )

    jacobian = np.array(
        [
            [1.7, 0.28],
            [-0.31, 0.82],
        ],
        dtype=float,
    )

    detector = transform_quadratic_form(
        source,
        jacobian,
    )

    recovered = transform_quadratic_form(
        detector,
        np.linalg.inv(jacobian),
    )

    assert recovered == pytest.approx(
        source,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def _make_tan_wcs(
    *,
    pixel_scale_arcsec: float = 0.2,
):
    from astropy.wcs import WCS

    wcs = WCS(
        naxis=2
    )
    wcs.wcs.crpix = [
        101.0,
        101.0,
    ]
    wcs.wcs.cdelt = np.array(
        [
            -pixel_scale_arcsec / 3600.0,
            pixel_scale_arcsec / 3600.0,
        ]
    )
    wcs.wcs.crval = [
        150.0,
        30.0,
    ]
    wcs.wcs.ctype = [
        "RA---TAN",
        "DEC--TAN",
    ]

    return wcs


def test_local_wcs_jacobian_recovers_tangent_plane_scale():
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from baymorph.observation.geometry import (
        local_wcs_jacobian,
    )

    wcs = _make_tan_wcs(
        pixel_scale_arcsec=0.2
    )

    centre = SkyCoord(
        150.0 * u.deg,
        30.0 * u.deg,
        frame="icrs",
    )

    jacobian = local_wcs_jacobian(
        wcs,
        centre,
        step_arcsec=0.01,
    )

    assert jacobian == pytest.approx(
        np.array(
            [
                [-5.0, 0.0],
                [0.0, 5.0],
            ]
        ),
        rel=1.0e-7,
        abs=1.0e-7,
    )


def test_local_wcs_jacobian_preserves_expected_parity():
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from baymorph.observation.geometry import (
        local_wcs_jacobian,
    )

    wcs = _make_tan_wcs()

    centre = SkyCoord(
        150.0 * u.deg,
        30.0 * u.deg,
        frame="icrs",
    )

    jacobian = local_wcs_jacobian(
        wcs,
        centre,
    )

    assert np.linalg.det(
        jacobian
    ) < 0.0


def test_wcs_jacobian_transforms_elliptical_geometry_consistently():
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from baymorph.observation.geometry import (
        local_wcs_jacobian,
    )

    wcs = _make_tan_wcs(
        pixel_scale_arcsec=0.2
    )

    centre = SkyCoord(
        150.0 * u.deg,
        30.0 * u.deg,
        frame="icrs",
    )

    source = ellipse_quadratic_form(
        re=1.0,
        q=0.6,
        theta=0.0,
    )

    jacobian = local_wcs_jacobian(
        wcs,
        centre,
    )

    detector = transform_quadratic_form(
        source,
        jacobian,
    )

    re, q, _ = quadratic_form_to_ellipse(
        detector
    )

    assert re == pytest.approx(
        5.0,
        rel=1.0e-7,
    )
    assert q == pytest.approx(
        0.6,
        rel=1.0e-7,
    )


def test_local_wcs_jacobian_recovers_rotated_anisotropic_reflected_mapping():
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS

    from baymorph.observation.geometry import (
        local_wcs_jacobian,
    )

    pixel_scale_x = 0.20
    pixel_scale_y = 0.35
    angle = np.deg2rad(27.0)

    rotation = np.array(
        [
            [
                np.cos(angle),
                -np.sin(angle),
            ],
            [
                np.sin(angle),
                np.cos(angle),
            ],
        ],
        dtype=float,
    )

    reflection = np.array(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    expected = (
        rotation
        @ reflection
        @ np.diag(
            [
                1.0 / pixel_scale_x,
                1.0 / pixel_scale_y,
            ]
        )
    )

    pixel_to_sky_arcsec = np.linalg.inv(
        expected
    )

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
    wcs.wcs.cd = (
        pixel_to_sky_arcsec
        / 3600.0
    )

    centre = SkyCoord(
        150.0 * u.deg,
        30.0 * u.deg,
        frame="icrs",
    )

    jacobian = local_wcs_jacobian(
        wcs,
        centre,
        step_arcsec=0.01,
    )

    assert jacobian == pytest.approx(
        expected,
        rel=1.0e-7,
        abs=1.0e-7,
    )

    assert np.linalg.det(
        jacobian
    ) == pytest.approx(
        np.linalg.det(
            expected
        ),
        rel=1.0e-7,
    )

    assert np.linalg.det(
        jacobian
    ) < 0.0


def test_sky_ellipse_to_detector_recovers_centre_scale_and_shape():
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from baymorph.observation.geometry import (
        sky_ellipse_to_detector,
    )

    wcs = _make_tan_wcs(
        pixel_scale_arcsec=0.2
    )

    centre = SkyCoord(
        150.0 * u.deg,
        30.0 * u.deg,
        frame="icrs",
    )

    geometry = sky_ellipse_to_detector(
        wcs=wcs,
        centre=centre,
        re_arcsec=1.2,
        q=0.6,
        position_angle=0.0,
    )

    assert geometry.x0 == pytest.approx(
        100.0,
        abs=1.0e-8,
    )
    assert geometry.y0 == pytest.approx(
        100.0,
        abs=1.0e-8,
    )
    assert geometry.re == pytest.approx(
        6.0,
        rel=1.0e-7,
    )
    assert geometry.q == pytest.approx(
        0.6,
        rel=1.0e-7,
    )


def test_sky_ellipse_to_detector_matches_general_affine_geometry():
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS

    from baymorph.observation.geometry import (
        sky_ellipse_to_detector,
    )

    pixel_scale_x = 0.20
    pixel_scale_y = 0.35
    detector_rotation = np.deg2rad(
        27.0
    )

    rotation = np.array(
        [
            [
                np.cos(detector_rotation),
                -np.sin(detector_rotation),
            ],
            [
                np.sin(detector_rotation),
                np.cos(detector_rotation),
            ],
        ]
    )

    reflection = np.array(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    expected_jacobian = (
        rotation
        @ reflection
        @ np.diag(
            [
                1.0 / pixel_scale_x,
                1.0 / pixel_scale_y,
            ]
        )
    )

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
    wcs.wcs.cd = (
        np.linalg.inv(
            expected_jacobian
        )
        / 3600.0
    )

    centre = SkyCoord(
        150.0 * u.deg,
        30.0 * u.deg,
        frame="icrs",
    )

    re_arcsec = 1.4
    q = 0.53
    angle = -0.38

    geometry = sky_ellipse_to_detector(
        wcs=wcs,
        centre=centre,
        re_arcsec=re_arcsec,
        q=q,
        position_angle=angle,
    )

    source = ellipse_quadratic_form(
        re=re_arcsec,
        q=q,
        theta=angle,
    )

    expected_detector = (
        transform_quadratic_form(
            source,
            expected_jacobian,
        )
    )

    expected_re, expected_q, expected_theta = (
        quadratic_form_to_ellipse(
            expected_detector
        )
    )

    assert geometry.x0 == pytest.approx(
        100.0,
        abs=1.0e-8,
    )
    assert geometry.y0 == pytest.approx(
        100.0,
        abs=1.0e-8,
    )

    assert geometry.re == pytest.approx(
        expected_re,
        rel=1.0e-7,
    )
    assert geometry.q == pytest.approx(
        expected_q,
        rel=1.0e-7,
    )
    assert _axis_angle_difference(
        geometry.theta,
        expected_theta,
    ) == pytest.approx(
        0.0,
        abs=1.0e-7,
    )


def test_slice_wcs_to_cutout_translates_detector_centre():
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from baymorph.observation.geometry import (
        slice_wcs_to_cutout,
    )

    wcs = _make_tan_wcs(
        pixel_scale_arcsec=0.2
    )

    centre = SkyCoord(
        150.0 * u.deg,
        30.0 * u.deg,
        frame="icrs",
    )

    xmin = 40
    xmax = 160
    ymin = 55
    ymax = 175

    cutout_wcs = slice_wcs_to_cutout(
        wcs,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )

    x_parent, y_parent = wcs.world_to_pixel(
        centre
    )
    x_cutout, y_cutout = (
        cutout_wcs.world_to_pixel(
            centre
        )
    )

    assert x_cutout == pytest.approx(
        x_parent - xmin,
        abs=1.0e-10,
    )
    assert y_cutout == pytest.approx(
        y_parent - ymin,
        abs=1.0e-10,
    )


def test_slice_wcs_to_cutout_preserves_local_jacobian():
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from baymorph.observation.geometry import (
        local_wcs_jacobian,
        slice_wcs_to_cutout,
    )

    wcs = _make_tan_wcs(
        pixel_scale_arcsec=0.2
    )

    centre = SkyCoord(
        150.0 * u.deg,
        30.0 * u.deg,
        frame="icrs",
    )

    cutout_wcs = slice_wcs_to_cutout(
        wcs,
        xmin=40,
        xmax=160,
        ymin=55,
        ymax=175,
    )

    parent_jacobian = local_wcs_jacobian(
        wcs,
        centre,
    )
    cutout_jacobian = (
        local_wcs_jacobian(
            cutout_wcs,
            centre,
        )
    )

    assert cutout_jacobian == pytest.approx(
        parent_jacobian,
        rel=1.0e-10,
        abs=1.0e-10,
    )


def test_slice_wcs_to_cutout_rejects_invalid_bounds():
    from baymorph.observation.geometry import (
        slice_wcs_to_cutout,
    )

    wcs = _make_tan_wcs()

    with pytest.raises(
        ValueError,
        match="xmax",
    ):
        slice_wcs_to_cutout(
            wcs,
            xmin=10,
            xmax=10,
            ymin=20,
            ymax=30,
        )
