import numpy as np
import pytest

from baymorph import SyntheticSersic
from baymorph.pipeline import prepare_real_fits_image


def _sersic_image(size):
    """Render one centred Sérsic source on a square frame."""
    return SyntheticSersic(
        flux=1000,
        re=8,
        n=2,
        q=0.8,
        theta=0.0,
        x0=size / 2,
        y0=size / 2,
    ).render(
        shape=(size, size),
    )


def test_prepare_real_fits_image():

    image = _sersic_image(120)

    result = prepare_real_fits_image(
        image,
    )

    assert result.data.shape == image.shape

    assert result.mask.shape == image.shape

    assert result.noise.shape == image.shape

    assert np.isfinite(
        result.background
    )

    assert result.detection is not None

    assert result.detection.mask.shape == image.shape

    assert result.mask.shape == image.shape


def test_prepare_real_fits_image_rejects_unestimable_background():
    """No background may be estimated without the detection mask."""
    with pytest.raises(
        ValueError,
        match="mesh background could not be estimated",
    ):
        prepare_real_fits_image(
            _sersic_image(60),
        )
