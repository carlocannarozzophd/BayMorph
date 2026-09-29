"""Tests for the source mask produced by segmentation detection."""

from baymorph import SyntheticSersic
from baymorph.core import detect_sources_segmentation


def test_detection_mask_covers_the_source():
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

    mask = detect_sources_segmentation(
        image,
    ).mask

    assert mask.shape == image.shape

    assert mask.sum() > 0

    assert mask[
        50,
        50,
    ]