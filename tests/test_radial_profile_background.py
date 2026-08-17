"""Tests for fitted-background handling in radial diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.pipeline.radial_profile import (
    _residual_background_image,
)


class PlaneBackgroundModel:
    """Minimal observed model exposing a spatial background map."""

    def render_background(
        self,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Render a deterministic plane."""
        y, x = np.indices(
            shape,
            dtype=float,
        )

        x_ref = 0.5 * (shape[1] - 1)
        y_ref = 0.5 * (shape[0] - 1)

        return (
            2.0
            + 0.1 * (x - x_ref)
            - 0.2 * (y - y_ref)
        )


class LegacyModel:
    """Minimal model without explicit background rendering."""


class InvalidBackgroundModel:
    """Observed model returning an invalid background shape."""

    def render_background(
        self,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Return an intentionally invalid background."""
        return np.zeros(
            (
                shape[0] - 1,
                shape[1],
            ),
            dtype=float,
        )


def test_radial_profile_uses_full_rendered_background():
    """Spatial backgrounds must be removed pixel by pixel."""
    shape = (
        5,
        7,
    )

    background = _residual_background_image(
        PlaneBackgroundModel(),
        {
            "background_level": 999.0,
            "sky": 999.0,
        },
        shape,
    )

    assert background.shape == shape

    assert background[2, 3] == pytest.approx(
        2.0
    )

    assert background[2, 4] == pytest.approx(
        2.1
    )

    assert background[3, 3] == pytest.approx(
        1.8
    )


def test_radial_profile_preserves_legacy_scalar_background():
    """Legacy models must retain scalar-background diagnostics."""
    background = _residual_background_image(
        LegacyModel(),
        {
            "sky": 3.5,
        },
        (
            4,
            6,
        ),
    )

    np.testing.assert_allclose(
        background,
        3.5,
    )


def test_background_level_precedes_legacy_sky_fallback():
    """The new background name must take precedence when both are present."""
    background = _residual_background_image(
        LegacyModel(),
        {
            "background_level": 1.25,
            "sky": 99.0,
        },
        (
            3,
            3,
        ),
    )

    np.testing.assert_allclose(
        background,
        1.25,
    )


def test_radial_profile_rejects_wrong_background_shape():
    """Background and fitted-image grids must agree exactly."""
    with pytest.raises(
        ValueError,
        match="background shape",
    ):
        _residual_background_image(
            InvalidBackgroundModel(),
            {},
            (
                5,
                5,
            ),
        )
