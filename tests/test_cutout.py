"""Tests for cutout-definition utilities."""

import numpy as np
import pytest

from baymorph.pipeline.cutout import (
    resolve_plateau_cutout,
    resolve_preliminary_auto_cutout,
    validate_plateau_cutout_containment,
)


def test_preliminary_auto_cutout_is_centered_on_target_position():
    """The preliminary cutout must remain anchored to the target seed."""
    segment = np.zeros((200, 200), dtype=bool)

    # Simulate a blended footprint extending strongly toward a contaminant.
    segment[90:111, 40:61] = True
    segment[95:106, 61:151] = True

    cutout = resolve_preliminary_auto_cutout(
        image_shape=segment.shape,
        target_segment=segment,
        center_x=50.0,
        center_y=100.0,
        initial_factor=2.0,
        min_size=20,
        max_size=100,
    )

    resolved_center_x = 0.5 * (cutout.xmin + cutout.xmax)
    resolved_center_y = 0.5 * (cutout.ymin + cutout.ymax)

    assert resolved_center_x == pytest.approx(50.0)
    assert resolved_center_y == pytest.approx(100.0)

def test_resolved_plateau_cutout_contains_requested_footprint():
    cutout = resolve_plateau_cutout(
        image_shape=(300, 300),
        center_x=150.0,
        center_y=150.0,
        plateau_radius=40.0,
        sky_margin=10.0,
    )

    validate_plateau_cutout_containment(
        image_shape=(cutout.height, cutout.width),
        center_x=150.0 - cutout.xmin,
        center_y=150.0 - cutout.ymin,
        plateau_radius=40.0,
        sky_margin=10.0,
    )

    assert cutout.width == 101
    assert cutout.height == 101


def test_capped_pgc_scale_cutout_fails_containment():
    cutout = resolve_plateau_cutout(
        image_shape=(1451, 1451),
        center_x=725.0,
        center_y=725.0,
        plateau_radius=371.0,
        sky_margin=20.0,
        max_size=694,
    )

    with pytest.raises(
        RuntimeError,
        match="required=391.000",
    ):
        validate_plateau_cutout_containment(
            image_shape=(cutout.height, cutout.width),
            center_x=725.0 - cutout.xmin,
            center_y=725.0 - cutout.ymin,
            plateau_radius=371.0,
            sky_margin=20.0,
        )


def test_plateau_cutout_containment_rejects_truncated_footprint():
    with pytest.raises(
        RuntimeError,
        match="does not contain the final plateau radius",
    ):
        validate_plateau_cutout_containment(
            image_shape=(100, 100),
            center_x=49.5,
            center_y=49.5,
            plateau_radius=40.0,
            sky_margin=10.1,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("center_x", np.nan),
        ("center_y", np.inf),
        ("plateau_radius", np.nan),
        ("sky_margin", np.inf),
    ],
)
def test_plateau_cutout_containment_rejects_non_finite_values(
    name,
    value,
):
    arguments = {
        "image_shape": (100, 100),
        "center_x": 49.5,
        "center_y": 49.5,
        "plateau_radius": 40.0,
        "sky_margin": 10.0,
    }
    arguments[name] = value

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        validate_plateau_cutout_containment(**arguments)
