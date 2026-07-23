"""Tests for cutout-definition utilities."""

import numpy as np
import pytest

from baymorph.pipeline.cutout import (
    resolve_preliminary_auto_cutout,
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