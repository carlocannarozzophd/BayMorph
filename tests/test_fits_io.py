import numpy as np

from baymorph import (
    ImageData,
    SyntheticSersic,
)
from baymorph.simulation import make_observation



def test_image_data_fits_roundtrip(tmp_path):
    data = np.ones((20, 20))

    image = ImageData(data=data)

    filename = tmp_path / "test.fits"

    image.to_fits(filename)

    loaded = ImageData.from_fits(filename)

    assert loaded.shape == (20, 20)
    assert np.allclose(loaded.data, data)


def test_synthetic_observation_to_fits(tmp_path):
    galaxy = SyntheticSersic(
        flux=1000,
        re=10,
        n=4,
        q=0.8,
        theta=0.0,
        x0=50,
        y0=50,
    )

    obs = make_observation(
        galaxy,
        shape=(100, 100),
        psf_fwhm=2.5,
        poisson_noise=True,
        read_noise_sigma=1.0,
        seed=42,
    )

    filename = tmp_path / "mock.fits"

    obs.to_fits(filename)

    loaded = ImageData.from_fits(filename)

    assert loaded.shape == obs.shape
    assert np.allclose(loaded.data, obs.image)