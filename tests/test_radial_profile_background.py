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


class _StubPosterior:
    """Minimal posterior exposing equal-weight samples and parameter names."""

    def __init__(
        self,
        parameter_names: list[str],
        samples: np.ndarray,
    ) -> None:
        self.parameter_names = parameter_names
        self.equal_weight_samples = samples


class _StubFit:
    """Minimal fit product exposing only the posterior."""

    def __init__(
        self,
        posterior,
    ) -> None:
        self.posterior = posterior


PLANE_PARAMETER_NAMES = [
    "flux",
    "re",
    "n",
    "q",
    "theta",
    "x0",
    "y0",
    "background_level",
    "background_gradient_x",
    "background_gradient_y",
]


def _plane_posterior() -> _StubPosterior:
    """Return samples carrying non-zero residual-background gradients."""
    rng = np.random.default_rng(20260921)
    samples = np.empty((8, len(PLANE_PARAMETER_NAMES)), dtype=float)

    samples[:, 0] = 1.0e4 + rng.normal(0.0, 10.0, size=8)
    samples[:, 1] = 4.0
    samples[:, 2] = 2.0
    samples[:, 3] = 0.8
    samples[:, 4] = 0.2
    samples[:, 5] = 20.0
    samples[:, 6] = 20.0
    samples[:, 7] = 0.5
    samples[:, 8] = rng.normal(0.01, 1e-3, size=8)
    samples[:, 9] = rng.normal(-0.01, 1e-3, size=8)

    return _StubPosterior(PLANE_PARAMETER_NAMES, samples)


def _profile_arguments() -> dict:
    """Return the fixed geometry shared by the posterior-band tests."""
    return {
        "image_shape": (41, 41),
        "radii": np.linspace(0.0, 18.0, 5),
        "center_x": 20.0,
        "center_y": 20.0,
        "axis_ratio": 1.0,
        "position_angle": 0.0,
        "mask": None,
    }


def test_posterior_bands_use_the_supplied_model_factory() -> None:
    """A plane-background posterior must be rendered with a plane background.

    Rebuilding a factory inside the posterior-band computation silently fell
    back to the default constant background, which rejects the non-zero
    gradients every plane-background posterior carries.
    """
    from baymorph.pipeline.radial_profile import _compute_posterior_profiles
    from baymorph.pipeline.real_fit import _make_model_factory
    from baymorph.simulation.psf import GaussianPSF

    factory = _make_model_factory(
        "single_sersic",
        GaussianPSF(fwhm=2.5),
        background_cfg={"model": "plane"},
    )

    result = _compute_posterior_profiles(
        fit=_StubFit(_plane_posterior()),
        model_factory=factory,
        **_profile_arguments(),
    )

    assert result is not None
    assert np.all(np.isfinite(result.total.p50))
    assert np.all(result.total.p16 <= result.total.p50)
    assert np.all(result.total.p50 <= result.total.p84)


def test_posterior_bands_reject_a_background_the_samples_do_not_match() -> None:
    """The failure the supplied factory avoids must still be a hard error."""
    from baymorph.pipeline.radial_profile import _compute_posterior_profiles
    from baymorph.pipeline.real_fit import _make_model_factory
    from baymorph.simulation.psf import GaussianPSF

    factory = _make_model_factory(
        "single_sersic",
        GaussianPSF(fwhm=2.5),
        background_cfg={"model": "constant"},
    )

    with pytest.raises(ValueError, match="background_gradient"):
        _compute_posterior_profiles(
            fit=_StubFit(_plane_posterior()),
            model_factory=factory,
            **_profile_arguments(),
        )


def test_posterior_bands_return_none_without_a_posterior() -> None:
    """A fit without a posterior yields no bands rather than failing."""
    from baymorph.pipeline.radial_profile import _compute_posterior_profiles
    from baymorph.pipeline.real_fit import _make_model_factory
    from baymorph.simulation.psf import GaussianPSF

    factory = _make_model_factory(
        "single_sersic",
        GaussianPSF(fwhm=2.5),
        background_cfg={"model": "plane"},
    )

    assert (
        _compute_posterior_profiles(
            fit=_StubFit(None),
            model_factory=factory,
            **_profile_arguments(),
        )
        is None
    )
