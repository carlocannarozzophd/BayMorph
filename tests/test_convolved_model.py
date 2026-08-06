"""Tests for PSF-convolved morphological models."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph import (
    ConvolvedModel,
    GaussianPSF,
    SersicModel,
)
from baymorph.observation import PSFEffect
from baymorph.simulation import convolve_with_psf


def make_model(
    *,
    flux: float = 1000.0,
    re: float = 5.0,
    n: float = 1.5,
    q: float = 0.7,
    theta: float = 0.4,
    x0: float = 15.0,
    y0: float = 15.0,
) -> SersicModel:
    """Return a valid Sérsic model."""
    return SersicModel(
        flux=flux,
        re=re,
        n=n,
        q=q,
        theta=theta,
        x0=x0,
        y0=y0,
    )


def test_convolved_model_preserves_shape() -> None:
    """Convolution must preserve the requested output shape."""
    convolved = ConvolvedModel(
        make_model(
            re=8.0,
            n=2.0,
            q=0.8,
            theta=0.0,
            x0=30.0,
            y0=30.0,
        ),
        GaussianPSF(fwhm=3.0),
    )

    image = convolved.render(
        shape=(60, 60),
    )

    assert image.shape == (60, 60)


def test_convolved_model_uses_psf_effect() -> None:
    """The wrapper must delegate finite convolution to PSFEffect."""
    convolved = ConvolvedModel(
        make_model(),
        GaussianPSF(fwhm=2.5),
        kernel_size=15,
    )

    assert isinstance(
        convolved.psf_effect,
        PSFEffect,
    )
    assert convolved.kernel_size == 15


def test_convolved_model_includes_flux_from_outside_cutout() -> None:
    """Extended rendering must retain light scattered into the cutout."""
    model = make_model(
        re=3.0,
        n=1.0,
        q=1.0,
        theta=0.0,
        x0=-0.75,
        y0=15.0,
    )

    psf = GaussianPSF(fwhm=3.0)
    convolved = ConvolvedModel(
        model,
        psf,
        kernel_size=15,
    )

    physical = convolved.render(
        shape=(31, 31),
    )

    truncated_intrinsic = model.render(
        shape=(31, 31),
    )
    truncated_convolution = convolve_with_psf(
        truncated_intrinsic,
        psf,
        kernel_size=15,
    )

    assert physical.sum() > truncated_convolution.sum()


def test_convolved_model_matches_large_domain_reference() -> None:
    """The wrapper must match explicit extended-domain convolution."""
    model = make_model(
        x0=12.3,
        y0=14.7,
    )

    psf = GaussianPSF(fwhm=2.5)
    convolved = ConvolvedModel(
        model,
        psf,
        kernel_size=15,
    )

    actual = convolved.render(
        shape=(31, 29),
    )

    pad = 7
    extended = model.render(
        shape=(31 + 2 * pad, 29 + 2 * pad),
        origin=(-pad, -pad),
    )
    reference_extended = convolve_with_psf(
        extended,
        psf,
        kernel_size=15,
    )
    expected = reference_extended[
        pad : pad + 31,
        pad : pad + 29,
    ]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_convolved_components_sum_to_total() -> None:
    """Named component images must reproduce the convolved total."""
    convolved = ConvolvedModel(
        make_model(),
        GaussianPSF(fwhm=2.5),
        kernel_size=15,
    )

    total = convolved.render(
        shape=(31, 31),
    )
    components = convolved.render_components(
        shape=(31, 31),
    )

    np.testing.assert_allclose(
        components["sersic"],
        total,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_component_rendering_requires_decomposition() -> None:
    """Models without render_components() must fail explicitly."""

    class RenderOnlyModel:
        def render(
            self,
            shape,
            *,
            origin=(0.0, 0.0),
        ):
            del origin
            return np.ones(
                shape,
                dtype=float,
            )

    convolved = ConvolvedModel(
        RenderOnlyModel(),
        GaussianPSF(fwhm=2.0),
    )

    with pytest.raises(
        TypeError,
        match=r"must implement render_components",
    ):
        convolved.render_components(
            shape=(11, 11),
        )
