import numpy as np
import pytest

from baymorph import (
    GaussianPSF,
    SyntheticSersic,
)
from baymorph.simulation import convolve_with_psf


def test_psf_kernel_normalization():
    psf = GaussianPSF(fwhm=2.5)

    kernel = psf.kernel()

    assert kernel.sum() == pytest.approx(1.0)


def test_gaussian_psf_does_not_include_pixel_response():
    psf = GaussianPSF(fwhm=2.5)

    assert psf.includes_pixel_response is False


def test_convolution_preserves_flux_away_from_boundaries():
    galaxy = SyntheticSersic(
        flux=1000.0,
        re=5.0,
        n=1.0,
        q=0.8,
        theta=0.0,
        x0=100.0,
        y0=100.0,
    )

    image = galaxy.render(
        shape=(201, 201),
        integration_order=7,
    )

    convolved = convolve_with_psf(
        image,
        GaussianPSF(fwhm=2.5),
    )

    assert convolved.sum() == pytest.approx(
        image.sum(),
        rel=1.0e-6,
    )


def test_convolution_loses_flux_scattered_outside_footprint():
    image = np.zeros((9, 9), dtype=float)
    image[0, 0] = 1.0

    convolved = convolve_with_psf(
        image,
        GaussianPSF(fwhm=2.0),
        kernel_size=7,
    )

    assert convolved.sum() < image.sum()
    assert convolved[-1, -1] == pytest.approx(
        0.0,
        abs=1.0e-12,
    )


def test_convolution_preserves_image_shape():
    image = np.zeros((13, 17), dtype=float)

    convolved = convolve_with_psf(
        image,
        GaussianPSF(fwhm=2.0),
    )

    assert convolved.shape == image.shape


@pytest.mark.parametrize(
    "image",
    [
        np.zeros(5),
        np.zeros((2, 3, 4)),
        np.full((5, 5), np.nan),
        np.full((5, 5), np.inf),
    ],
)
def test_convolution_rejects_invalid_images(image):
    with pytest.raises(ValueError):
        convolve_with_psf(
            image,
            GaussianPSF(fwhm=2.0),
        )


@pytest.mark.parametrize(
    "fwhm",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
    ],
)
def test_gaussian_psf_rejects_invalid_fwhm(fwhm):
    with pytest.raises(ValueError):
        GaussianPSF(fwhm=fwhm)


@pytest.mark.parametrize(
    "size",
    [
        0,
        -1,
        4,
        2.5,
    ],
)
def test_gaussian_psf_rejects_invalid_kernel_size(size):
    with pytest.raises(ValueError):
        GaussianPSF(fwhm=2.0).kernel(size=size)
