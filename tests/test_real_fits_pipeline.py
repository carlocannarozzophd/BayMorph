import numpy as np

from baymorph import (
    SyntheticSersic,
    prepare_real_fits_image,
)


def test_prepare_real_fits_image():

    image = SyntheticSersic(
        flux=1000,
        re=8,
        n=2,
        q=0.8,
        theta=0.0,
        x0=30,
        y0=30,
    ).render(
        shape=(60, 60),
    )

    result = prepare_real_fits_image(
        image,
    )

    assert result.data.shape == image.shape

    assert result.mask.shape == image.shape

    assert result.noise.shape == image.shape

    assert np.isfinite(
        result.background
    )