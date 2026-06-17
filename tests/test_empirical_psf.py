import numpy as np

from baymorph import (
    EmpiricalPSF,
    convolve_with_psf,
)


def test_empirical_psf_normalization():

    kernel = np.ones(
        (11, 11),
    )

    psf = EmpiricalPSF(
        kernel_array=kernel,
    )

    normalized = psf.kernel()

    assert np.isclose(
        normalized.sum(),
        1.0,
    )


def test_empirical_psf_convolution_preserves_flux():

    image = np.zeros(
        (101, 101),
    )

    image[
        50,
        50,
    ] = 100.0

    kernel = np.ones(
        (11, 11),
    )

    psf = EmpiricalPSF(
        kernel_array=kernel,
    )

    convolved = convolve_with_psf(
        image,
        psf,
    )

    assert np.isclose(
        convolved.sum(),
        image.sum(),
        rtol=1e-6,
    )


def test_empirical_psf_requires_positive_flux():

    import pytest

    psf = EmpiricalPSF(
        kernel_array=np.zeros(
            (11, 11),
        ),
    )

    with pytest.raises(
        ValueError,
    ):
        psf.kernel()