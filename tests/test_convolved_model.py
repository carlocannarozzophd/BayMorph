"""Tests for PSF-convolved morphological models."""

from __future__ import annotations

import numpy as np
import pytest

from baymorph import (
    ConvolvedModel,
    EmpiricalPSF,
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


def test_convolved_model_point_samples_for_empirical_psf() -> None:
    """An empirical PSF must trigger point sampling, not pixel integration."""
    model = make_model(
        re=0.6,
        n=6.0,
        q=0.6,
        theta=0.2,
        x0=15.3,
        y0=14.7,
    )

    psf = EmpiricalPSF(
        kernel_array=np.ones((7, 7)),
    )

    convolved = ConvolvedModel(
        model,
        psf,
    )

    actual = convolved.render(
        shape=(31, 31),
    )

    pad = 3
    extended_shape = (31 + 2 * pad, 31 + 2 * pad)
    extended_origin = (-float(pad), -float(pad))

    point_sampled = model.render(
        shape=extended_shape,
        origin=extended_origin,
        renderer="point",
    )
    expected = convolve_with_psf(
        point_sampled,
        psf,
    )[
        pad : pad + 31,
        pad : pad + 31,
    ]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )

    pixel_integrated = model.render(
        shape=extended_shape,
        origin=extended_origin,
    )
    double_counted_reference = convolve_with_psf(
        pixel_integrated,
        psf,
    )[
        pad : pad + 31,
        pad : pad + 31,
    ]

    assert (
        np.linalg.norm(
            actual - double_counted_reference
        )
        > 1.0e-6
    )


def test_convolved_model_gaussian_psf_still_pixel_integrates() -> None:
    """A Gaussian PSF must keep the historical pixel-integration behavior."""
    model = make_model()
    psf = GaussianPSF(fwhm=2.5)

    convolved = ConvolvedModel(
        model,
        psf,
        kernel_size=15,
    )

    actual = convolved.render(
        shape=(31, 31),
    )

    pad = 7
    extended = model.render(
        shape=(31 + 2 * pad, 31 + 2 * pad),
        origin=(-pad, -pad),
    )
    expected = convolve_with_psf(
        extended,
        psf,
        kernel_size=15,
    )[
        pad : pad + 31,
        pad : pad + 31,
    ]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_convolved_model_point_samples_double_sersic_for_empirical_psf() -> None:
    """The empirical-PSF contract must reach composite morphology models too."""
    from baymorph import DoubleSersicModel

    bulge = make_model(
        flux=300.0,
        re=0.8,
        n=5.0,
        q=0.8,
        theta=0.1,
        x0=15.0,
        y0=15.0,
    )
    disk = make_model(
        flux=700.0,
        re=10.0,
        n=1.0,
        q=0.6,
        theta=0.2,
        x0=15.0,
        y0=15.0,
    )
    galaxy = DoubleSersicModel(
        bulge=bulge,
        disk=disk,
    )

    psf = EmpiricalPSF(
        kernel_array=np.ones((7, 7)),
    )

    convolved = ConvolvedModel(
        galaxy,
        psf,
    )

    actual = convolved.render(
        shape=(31, 31),
    )

    pad = 3
    extended_shape = (31 + 2 * pad, 31 + 2 * pad)
    extended_origin = (-float(pad), -float(pad))

    point_sampled = galaxy.render(
        shape=extended_shape,
        origin=extended_origin,
        renderer="point",
    )
    expected = convolve_with_psf(
        point_sampled,
        psf,
    )[
        pad : pad + 31,
        pad : pad + 31,
    ]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_convolved_model_requires_pixel_response_contract() -> None:
    """PSF objects that omit the pixel-response contract must fail closed."""

    class BarePSF:
        def kernel(self):
            return np.ones((5, 5)) / 25.0

    convolved = ConvolvedModel(
        make_model(),
        BarePSF(),
    )

    with pytest.raises(
        TypeError,
        match="includes_pixel_response",
    ):
        convolved.render(
            shape=(21, 21),
        )
