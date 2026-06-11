import numpy as np

from baymorph import (
    SyntheticSersic,
    detect_sources_mask,
)


def test_detect_sources_mask():

    galaxy = SyntheticSersic(
        flux=1000,
        re=8,
        n=2,
        q=0.8,
        theta=0.0,
        x0=50,
        y0=50,
    )

    image = galaxy.render(
        shape=(100, 100)
    )

    mask = detect_sources_mask(
        image,
    )

    assert mask.shape == image.shape

    assert mask.sum() > 0

    assert mask[
        50,
        50,
    ]