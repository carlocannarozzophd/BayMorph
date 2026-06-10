from baymorph import (
    SyntheticSersic,
    make_observation,
)


def test_make_observation():

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

    assert obs.shape == (100, 100)