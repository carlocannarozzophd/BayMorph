import numpy as np
import pytest

from baymorph.pipeline.preparation_steps import (
    estimate_target_geometry_from_segment,
)


def test_geometry_of_circular_segment():
    yy, xx = np.indices((41, 41))

    segment = (
        (xx - 20.0) ** 2
        + (yy - 20.0) ** 2
        <= 8.0**2
    )

    geometry = estimate_target_geometry_from_segment(segment)

    assert geometry.x0 == pytest.approx(20.0)
    assert geometry.y0 == pytest.approx(20.0)
    assert geometry.q == pytest.approx(1.0, abs=0.02)


def test_geometry_of_horizontal_ellipse():
    yy, xx = np.indices((61, 61))

    segment = (
        ((xx - 30.0) / 12.0) ** 2
        + ((yy - 30.0) / 6.0) ** 2
        <= 1.0
    )

    geometry = estimate_target_geometry_from_segment(segment)

    assert geometry.x0 == pytest.approx(30.0)
    assert geometry.y0 == pytest.approx(30.0)
    assert geometry.q == pytest.approx(0.5, abs=0.05)

    assert abs(np.sin(geometry.theta)) == pytest.approx(
        0.0,
        abs=0.05,
    )


def test_geometry_rejects_non_2d_segment():
    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        estimate_target_geometry_from_segment(
            np.ones(10, dtype=bool)
        )


def test_geometry_rejects_too_few_pixels():
    segment = np.zeros((10, 10), dtype=bool)
    segment[4, 4] = True
    segment[4, 5] = True

    with pytest.raises(
        ValueError,
        match="At least three",
    ):
        estimate_target_geometry_from_segment(segment)