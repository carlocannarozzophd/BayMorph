"""Tests for observation-level additive background models."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.observation.background import (
    BackgroundAppliedModel,
    ConstantBackground,
    TiltedPlaneBackground,
    make_background_effect,
)


class ZeroSourceModel:
    """Minimal source model used to test background composition."""

    def render(
        self,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Render a zero-valued expected source image."""
        return np.zeros(
            shape,
            dtype=float,
        )

    def render_components(
        self,
        shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        """Render one zero-valued named source component."""
        return {
            "source": np.zeros(
                shape,
                dtype=float,
            )
        }


def test_constant_background_renders_uniform_level():
    """Constant background must add the same level to every pixel."""
    background = ConstantBackground(
        level=2.5
    )

    image = background.render(
        (3, 5)
    )

    np.testing.assert_allclose(
        image,
        2.5,
    )


def test_tilted_plane_uses_geometric_image_centre():
    """The plane intercept must be defined at the geometric image centre."""
    background = TiltedPlaneBackground(
        level=10.0,
        gradient_x=2.0,
        gradient_y=-3.0,
    )

    image = background.render(
        (3, 5)
    )

    assert image[1, 2] == pytest.approx(
        10.0
    )

    assert image[1, 3] == pytest.approx(
        12.0
    )

    assert image[2, 2] == pytest.approx(
        7.0
    )


def test_plane_with_zero_gradients_matches_constant_background():
    """The zero-gradient plane must reduce exactly to a constant."""
    shape = (
        7,
        9,
    )

    constant = ConstantBackground(
        level=-1.25
    ).render(
        shape
    )

    plane = TiltedPlaneBackground(
        level=-1.25,
        gradient_x=0.0,
        gradient_y=0.0,
    ).render(
        shape
    )

    np.testing.assert_allclose(
        plane,
        constant,
        rtol=0.0,
        atol=0.0,
    )


def test_background_applied_model_keeps_background_out_of_components():
    """Background must affect total rendering but not source components."""
    model = BackgroundAppliedModel(
        model=ZeroSourceModel(),
        background=TiltedPlaneBackground(
            level=2.0,
            gradient_x=0.1,
            gradient_y=-0.2,
        ),
    )

    image = model.render(
        (5, 5)
    )

    components = model.render_components(
        (5, 5)
    )

    np.testing.assert_allclose(
        image,
        model.render_background(
            (5, 5)
        ),
    )

    np.testing.assert_allclose(
        components["source"],
        0.0,
    )


def test_background_factory_constructs_supported_models():
    """Background factory must preserve the explicit model family."""
    constant = make_background_effect(
        "constant",
        level=1.0,
    )

    plane = make_background_effect(
        "plane",
        level=1.0,
        gradient_x=0.1,
        gradient_y=-0.1,
    )

    assert isinstance(
        constant,
        ConstantBackground,
    )

    assert isinstance(
        plane,
        TiltedPlaneBackground,
    )


def test_constant_background_rejects_nonzero_gradients():
    """Constant background must not silently discard supplied gradients."""
    with pytest.raises(
        ValueError,
        match="does not accept non-zero gradients",
    ):
        make_background_effect(
            "constant",
            level=0.0,
            gradient_x=0.1,
        )


@pytest.mark.parametrize(
    "model",
    [
        "",
        "quadratic",
        "spline",
    ],
)
def test_background_factory_rejects_unknown_models(
    model: str,
):
    """Only explicitly implemented background families are accepted."""
    with pytest.raises(
        ValueError,
        match="Unsupported background model",
    ):
        make_background_effect(
            model,
            level=0.0,
        )
