import numpy as np

from baymorph import (
    make_double_sersic_psf_sky_model,
)


class DeltaPSF:
    def kernel(self):
        psf = np.zeros((9, 9))
        psf[4, 4] = 1.0
        return psf


def test_make_double_sersic_psf_sky_model_renders_expected_shape():
    psf = DeltaPSF()

    model = make_double_sersic_psf_sky_model(
        100.0,
        3.0,
        2.0,
        50.0,
        8.0,
        1.0,
        0.8,
        0.0,
        15.0,
        15.0,
        2.0,
        psf=psf,
    )

    image = model.render(
        (31, 31),
    )

    assert image.shape == (31, 31)
    assert np.all(
        np.isfinite(image)
    )


def test_make_double_sersic_psf_sky_model_adds_sky_level():
    psf = DeltaPSF()

    model_no_sky = make_double_sersic_psf_sky_model(
        100.0,
        3.0,
        2.0,
        50.0,
        8.0,
        1.0,
        0.8,
        0.0,
        15.0,
        15.0,
        0.0,
        psf=psf,
    )

    model_with_sky = make_double_sersic_psf_sky_model(
        100.0,
        3.0,
        2.0,
        50.0,
        8.0,
        1.0,
        0.8,
        0.0,
        15.0,
        15.0,
        5.0,
        psf=psf,
    )

    np.testing.assert_allclose(
        model_with_sky.render((31, 31)),
        model_no_sky.render((31, 31)) + 5.0,
    )