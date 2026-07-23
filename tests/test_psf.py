import numpy as np

from baymorph import (
    GaussianPSF,
    SyntheticSersic,
)
from baymorph.simulation import convolve_with_psf



def test_psf_kernel_normalization():

    psf = GaussianPSF(
        fwhm=2.5
    )

    kernel = psf.kernel()

    assert np.isclose(
        kernel.sum(),
        1.0,
    )


def test_convolution_flux_conservation():

    galaxy = SyntheticSersic(
        flux=1000.0,
        re=10.0,
        n=4.0,
        q=0.8,
        theta=0.0,
        x0=50.0,
        y0=50.0,
    )

    image = galaxy.render(
        shape=(100, 100)
    )

    image_conv = convolve_with_psf(
        image,
        GaussianPSF(fwhm=2.5),
    )

    assert np.isclose(
        image.sum(),
        image_conv.sum(),
        rtol=1e-3,
    )