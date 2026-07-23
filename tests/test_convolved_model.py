import numpy as np

from baymorph import (
    ConvolvedModel,
    GaussianPSF,
    SersicModel,
)



def test_convolved_model_preserves_flux():

    model = SersicModel(
        flux=1000,
        re=8,
        n=2,
        q=0.8,
        theta=0.0,
        x0=30,
        y0=30,
    )

    psf = GaussianPSF(
        fwhm=3.0,
    )

    convolved = ConvolvedModel(
        model,
        psf,
    )

    base_image = model.render(
        shape=(60, 60),
    )

    convolved_image = convolved.render(
        shape=(60, 60),
    )

    assert convolved_image.shape == base_image.shape

    assert np.isclose(
        convolved_image.sum(),
        base_image.sum(),
        rtol=5e-3,
    )