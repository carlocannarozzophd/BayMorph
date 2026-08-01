import numpy as np

from baymorph import (
    ConvolvedModel,
    GaussianPSF,
    SersicModel,
)


def test_convolved_model_preserves_shape():
    model = SersicModel(
        flux=1000.0,
        re=8.0,
        n=2.0,
        q=0.8,
        theta=0.0,
        x0=30.0,
        y0=30.0,
    )

    convolved = ConvolvedModel(
        model,
        GaussianPSF(fwhm=3.0),
    )

    image = convolved.render(shape=(60, 60))

    assert image.shape == (60, 60)


def test_convolved_model_includes_flux_from_outside_cutout():
    model = SersicModel(
        flux=1000.0,
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

    physical = convolved.render(shape=(31, 31))

    truncated_intrinsic = model.render(shape=(31, 31))
    truncated_convolution = convolve_reference(
        truncated_intrinsic,
        psf,
        kernel_size=15,
    )

    assert physical.sum() > truncated_convolution.sum()


def test_convolved_model_matches_large_domain_reference():
    model = SersicModel(
        flux=1000.0,
        re=5.0,
        n=1.5,
        q=0.7,
        theta=0.4,
        x0=12.3,
        y0=14.7,
    )

    psf = GaussianPSF(fwhm=2.5)
    convolved = ConvolvedModel(
        model,
        psf,
        kernel_size=15,
    )

    actual = convolved.render(shape=(31, 29))

    pad = 7
    extended = model.render(
        shape=(31 + 2 * pad, 29 + 2 * pad),
        origin=(-pad, -pad),
    )
    reference_extended = convolve_reference(
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


def test_convolved_components_sum_to_total():
    model = SersicModel(
        flux=1000.0,
        re=5.0,
        n=1.5,
        q=0.7,
        theta=0.4,
        x0=15.0,
        y0=15.0,
    )

    convolved = ConvolvedModel(
        model,
        GaussianPSF(fwhm=2.5),
        kernel_size=15,
    )

    total = convolved.render(shape=(31, 31))
    components = convolved.render_components(shape=(31, 31))

    np.testing.assert_allclose(
        components["sersic"],
        total,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def convolve_reference(
    image: np.ndarray,
    psf: GaussianPSF,
    *,
    kernel_size: int,
) -> np.ndarray:
    """Return the finite-domain reference convolution."""
    from baymorph.simulation import convolve_with_psf

    return convolve_with_psf(
        image,
        psf,
        kernel_size=kernel_size,
    )
