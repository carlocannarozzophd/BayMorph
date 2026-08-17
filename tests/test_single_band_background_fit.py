"""Tests for single-observation residual-background inference."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph.observation.background import (
    BackgroundAppliedModel,
    ConstantBackground,
    TiltedPlaneBackground,
)
from baymorph.pipeline.config import (
    DEFAULT_CONFIG,
    resolve_config,
)
from baymorph.pipeline.real_fit import (
    _build_parameter_set,
    _make_model_factory,
)


class DeltaPSF:
    """Identity PSF used for background-integration tests."""

    def kernel(self):
        """Return a normalized delta-function kernel."""
        kernel = np.zeros(
            (9, 9),
            dtype=float,
        )
        kernel[4, 4] = 1.0
        return kernel


def _minimal_config() -> dict:
    """Return a minimal valid single-observation configuration."""
    return {
        "object": {
            "name": "test",
        },
        "data": {
            "image_path": "test.fits",
        },
        "outputs": {
            "root": "outputs",
        },
    }


def test_default_single_band_background_is_constant():
    """Single-band production inference must default to constant residual sky."""
    resolved = resolve_config(
        _minimal_config()
    )

    assert resolved["fit"]["background"]["model"] == "constant"


def test_plane_background_config_is_resolved():
    """Plane background must be selectable without redefining default priors."""
    config = _minimal_config()

    config["fit"] = {
        "background": {
            "model": "plane",
        }
    }

    resolved = resolve_config(
        config
    )

    background = resolved["fit"]["background"]

    assert background["model"] == "plane"

    assert set(
        background["priors"]
    ) == {
        "background_level",
        "background_gradient_x",
        "background_gradient_y",
    }


def test_unknown_background_model_is_rejected():
    """Unsupported background families must fail during configuration."""
    config = _minimal_config()

    config["fit"] = {
        "background": {
            "model": "quadratic",
        }
    }

    with pytest.raises(
        ValueError,
        match="Unsupported fit.background.model",
    ):
        resolve_config(
            config
        )


def test_constant_background_parameter_contract_replaces_legacy_sky():
    """Single-band inference must separate background from morphology."""
    shape = (
        21,
        21,
    )

    parameter_names, parameters, _, _ = _build_parameter_set(
        "single_sersic",
        data_fit=np.ones(
            shape,
            dtype=float,
        ),
        noise_fit=np.ones(
            shape,
            dtype=float,
        ),
        mask_fit=np.zeros(
            shape,
            dtype=bool,
        ),
        target_x=10.0,
        target_y=10.0,
    )

    assert "sky" not in parameter_names
    assert "background_level" in parameter_names
    assert "background_gradient_x" not in parameter_names
    assert "background_gradient_y" not in parameter_names

    assert parameters.all_names == parameter_names


def test_plane_background_parameter_contract_adds_two_gradients():
    """Plane inference must sample level and detector-frame gradients."""
    shape = (
        21,
        21,
    )

    background_cfg = {
        "model": "plane",
        "priors": DEFAULT_CONFIG[
            "fit"
        ][
            "background"
        ][
            "priors"
        ],
    }

    parameter_names, parameters, _, _ = _build_parameter_set(
        "single_sersic",
        data_fit=np.ones(
            shape,
            dtype=float,
        ),
        noise_fit=np.ones(
            shape,
            dtype=float,
        ),
        mask_fit=np.zeros(
            shape,
            dtype=bool,
        ),
        target_x=10.0,
        target_y=10.0,
        background_cfg=background_cfg,
    )

    assert "sky" not in parameter_names

    assert parameter_names[-3:] == [
        "background_level",
        "background_gradient_x",
        "background_gradient_y",
    ]

    assert parameters.all_names == parameter_names


def test_single_sersic_factory_applies_constant_background():
    """Production factory must apply constant background outside morphology."""
    factory = _make_model_factory(
        "single_sersic",
        DeltaPSF(),
        background_cfg={
            "model": "constant",
        },
    )

    model = factory(
        flux=100.0,
        re=3.0,
        n=2.0,
        q=0.8,
        theta=0.0,
        x0=10.0,
        y0=10.0,
        background_level=2.0,
    )

    assert isinstance(
        model,
        BackgroundAppliedModel,
    )

    assert isinstance(
        model.background,
        ConstantBackground,
    )

    np.testing.assert_allclose(
        model.render_background(
            (21, 21)
        ),
        2.0,
    )


def test_single_sersic_factory_applies_tilted_plane():
    """Production factory must add the full detector-frame background plane."""
    factory = _make_model_factory(
        "single_sersic",
        DeltaPSF(),
        background_cfg={
            "model": "plane",
        },
    )

    model = factory(
        flux=100.0,
        re=3.0,
        n=2.0,
        q=0.8,
        theta=0.0,
        x0=10.0,
        y0=10.0,
        background_level=2.0,
        background_gradient_x=0.1,
        background_gradient_y=-0.2,
    )

    assert isinstance(
        model.background,
        TiltedPlaneBackground,
    )

    background = model.render_background(
        (21, 21)
    )

    assert background[10, 10] == pytest.approx(
        2.0
    )

    assert background[10, 11] == pytest.approx(
        2.1
    )

    assert background[11, 10] == pytest.approx(
        1.8
    )

    source_components = model.render_components(
        (21, 21)
    )

    source_image = sum(
        source_components.values()
    )

    np.testing.assert_allclose(
        model.render(
            (21, 21)
        ),
        source_image
        + background,
        rtol=0.0,
        atol=1.0e-12,
    )
