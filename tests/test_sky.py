import numpy as np
import pytest

from baymorph.core.sky import build_elliptical_radius_map


def test_radius_map_shape():
    r = build_elliptical_radius_map(
        shape=(50, 40),
        x0=20.0,
        y0=25.0,
        q=1.0,
        theta=0.0,
    )

    assert r.shape == (50, 40)


def test_radius_map_centre():
    r = build_elliptical_radius_map(
        shape=(21, 21),
        x0=10.0,
        y0=10.0,
        q=1.0,
        theta=0.0,
    )

    assert r[10, 10] == pytest.approx(0.0)


def test_radius_map_is_circular_for_q1():
    r = build_elliptical_radius_map(
        shape=(21, 21),
        x0=10.0,
        y0=10.0,
        q=1.0,
        theta=0.0,
    )

    assert r[10, 15] == pytest.approx(5.0)
    assert r[15, 10] == pytest.approx(5.0)
    assert r[5, 10] == pytest.approx(5.0)
    assert r[10, 5] == pytest.approx(5.0)


def test_radius_map_axis_ratio():
    r = build_elliptical_radius_map(
        shape=(41, 41),
        x0=20.0,
        y0=20.0,
        q=0.5,
        theta=0.0,
    )

    assert r[20, 25] == pytest.approx(5.0)
    assert r[25, 20] == pytest.approx(10.0)


def test_radius_map_non_negative():
    r = build_elliptical_radius_map(
        shape=(50, 50),
        x0=25.0,
        y0=25.0,
        q=0.7,
        theta=np.pi / 6,
    )

    assert np.all(r >= 0.0)


def test_invalid_axis_ratio():
    with pytest.raises(ValueError):
        build_elliptical_radius_map(
            shape=(20, 20),
            x0=10.0,
            y0=10.0,
            q=0.0,
            theta=0.0,
        )

    with pytest.raises(ValueError):
        build_elliptical_radius_map(
            shape=(20, 20),
            x0=10.0,
            y0=10.0,
            q=1.2,
            theta=0.0,
        )