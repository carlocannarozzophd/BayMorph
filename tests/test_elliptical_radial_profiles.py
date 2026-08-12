"""Tests for elliptical radial-profile geometry."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.analysis.radial_profiles import (
    elliptical_radius_map,
    radial_bins,
    radial_profile_from_image,
)


def test_circular_radius_is_recovered_for_unit_axis_ratio():
    radius = elliptical_radius_map(
        (7, 7),
        center_x=3.0,
        center_y=3.0,
        axis_ratio=1.0,
        position_angle=0.73,
    )

    yy, xx = np.indices(
        (7, 7),
        dtype=float,
    )

    expected = np.sqrt(
        (xx - 3.0) ** 2
        + (yy - 3.0) ** 2
    )

    np.testing.assert_allclose(
        radius,
        expected,
    )


def test_elliptical_radius_uses_major_and_minor_axes():
    radius = elliptical_radius_map(
        (9, 9),
        center_x=4.0,
        center_y=4.0,
        axis_ratio=0.5,
        position_angle=0.0,
    )

    assert radius[
        4,
        6,
    ] == pytest.approx(
        2.0
    )

    assert radius[
        5,
        4,
    ] == pytest.approx(
        2.0
    )


def test_position_angle_rotates_elliptical_annuli():
    radius = elliptical_radius_map(
        (9, 9),
        center_x=4.0,
        center_y=4.0,
        axis_ratio=0.5,
        position_angle=0.5 * np.pi,
    )

    assert radius[
        6,
        4,
    ] == pytest.approx(
        2.0
    )

    assert radius[
        4,
        5,
    ] == pytest.approx(
        2.0
    )


def test_profile_follows_elliptical_annuli():
    shape = (
        21,
        21,
    )

    radius = elliptical_radius_map(
        shape,
        center_x=10.0,
        center_y=10.0,
        axis_ratio=0.5,
        position_angle=np.deg2rad(
            30.0
        ),
    )

    image = np.full(
        shape,
        np.nan,
        dtype=float,
    )

    image[
        radius < 1.0
    ] = 1.0

    image[
        (
            radius >= 1.0
        )
        & (
            radius < 2.0
        )
    ] = 2.0

    image[
        (
            radius >= 2.0
        )
        & (
            radius < 3.0
        )
    ] = 3.0

    profile = radial_profile_from_image(
        image,
        center_x=10.0,
        center_y=10.0,
        radii=np.asarray(
            [
                0.0,
                1.0,
                2.0,
                3.0,
            ]
        ),
        axis_ratio=0.5,
        position_angle=np.deg2rad(
            30.0
        ),
    )

    np.testing.assert_allclose(
        profile,
        [
            1.0,
            2.0,
            3.0,
        ],
    )


def test_profile_geometry_changes_when_axis_ratio_changes():
    shape = (
        21,
        21,
    )

    elliptical_radius = elliptical_radius_map(
        shape,
        center_x=10.0,
        center_y=10.0,
        axis_ratio=0.4,
        position_angle=0.0,
    )

    image = elliptical_radius.copy()

    radii = np.asarray(
        [
            0.0,
            2.0,
            4.0,
            6.0,
        ]
    )

    elliptical_profile = radial_profile_from_image(
        image,
        center_x=10.0,
        center_y=10.0,
        radii=radii,
        axis_ratio=0.4,
        position_angle=0.0,
    )

    circular_profile = radial_profile_from_image(
        image,
        center_x=10.0,
        center_y=10.0,
        radii=radii,
        axis_ratio=1.0,
        position_angle=0.0,
    )

    assert not np.allclose(
        elliptical_profile,
        circular_profile,
        equal_nan=True,
    )


def test_adaptive_bins_account_for_axis_ratio():
    circular = radial_bins(
        20.0,
        mode="adaptive",
        dr=0.5,
        min_pixels=50,
        axis_ratio=1.0,
    )

    flattened = radial_bins(
        20.0,
        mode="adaptive",
        dr=0.5,
        min_pixels=50,
        axis_ratio=0.5,
    )

    assert flattened[
        1
    ] > circular[
        1
    ]


@pytest.mark.parametrize(
    "axis_ratio",
    [
        0.0,
        -0.1,
        1.1,
        np.nan,
    ],
)
def test_invalid_axis_ratio_is_rejected(
    axis_ratio,
):
    with pytest.raises(
        ValueError,
        match="axis_ratio",
    ):
        elliptical_radius_map(
            (5, 5),
            center_x=2.0,
            center_y=2.0,
            axis_ratio=axis_ratio,
        )


def test_radial_profile_statistics_report_median_mean_and_percentiles():
    from baymorph.analysis.radial_profiles import (
        radial_profile_statistics_from_image,
    )

    shape = (
        31,
        31,
    )

    radius = elliptical_radius_map(
        shape,
        center_x=15.0,
        center_y=15.0,
        axis_ratio=0.5,
        position_angle=np.deg2rad(
            25.0
        ),
    )

    image = (
        2.0
        + radius
        + 0.05
        * np.arange(
            shape[0]
        )[
            :,
            None,
        ]
    )

    radii = np.asarray(
        [
            0.0,
            3.0,
            6.0,
            9.0,
        ]
    )

    statistics = radial_profile_statistics_from_image(
        image,
        center_x=15.0,
        center_y=15.0,
        radii=radii,
        axis_ratio=0.5,
        position_angle=np.deg2rad(
            25.0
        ),
    )

    assert statistics.radius.shape == (
        3,
    )

    assert statistics.median.shape == (
        3,
    )

    assert statistics.mean.shape == (
        3,
    )

    assert statistics.p16.shape == (
        3,
    )

    assert statistics.p84.shape == (
        3,
    )

    assert np.all(
        statistics.n_pixels
        > 0
    )

    assert np.all(
        statistics.p16
        <= statistics.median
    )

    assert np.all(
        statistics.median
        <= statistics.p84
    )


def test_radial_profile_compatibility_interface_returns_statistics_median():
    from baymorph.analysis.radial_profiles import (
        radial_profile_statistics_from_image,
    )

    image = np.arange(
        21 * 21,
        dtype=float,
    ).reshape(
        21,
        21,
    )

    radii = np.asarray(
        [
            0.0,
            2.0,
            4.0,
            6.0,
        ]
    )

    kwargs = {
        "center_x": 10.0,
        "center_y": 10.0,
        "radii": radii,
        "axis_ratio": 0.6,
        "position_angle": 0.4,
    }

    profile = radial_profile_from_image(
        image,
        **kwargs,
    )

    statistics = radial_profile_statistics_from_image(
        image,
        **kwargs,
    )

    np.testing.assert_allclose(
        profile,
        statistics.median,
    )
