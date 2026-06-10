import numpy as np

from baymorph import SyntheticSersic


def test_synthetic_sersic_shape():
    model = SyntheticSersic(
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


def test_synthetic_sersic_flux_conservation():
    model = SyntheticSersic(
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