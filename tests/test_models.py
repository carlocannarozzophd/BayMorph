import numpy as np

from baymorph import SersicModel


def test_sersic_model_shape():
    model = SersicModel(
        flux=1000.0,
        re=10.0,
        n=4.0,
        q=0.7,
        theta=0.5,
        x0=50.0,
        y0=50.0,
    )

    image = model.render(shape=(100, 100))

    assert image.shape == (100, 100)


def test_sersic_model_flux_conservation():
    model = SersicModel(
        flux=1000.0,
        re=10.0,
        n=4.0,
        q=0.7,
        theta=0.5,
        x0=50.0,
        y0=50.0,
    )

    image = model.render(shape=(100, 100))

    assert np.isclose(image.sum(), 1000.0)


def test_sersic_model_peak_positive():
    model = SersicModel(
        flux=1000.0,
        re=10.0,
        n=4.0,
        q=0.7,
        theta=0.5,
        x0=50.0,
        y0=50.0,
    )

    image = model.render(shape=(100, 100))

    assert image.max() > 0