import numpy as np
import pytest

from baymorph import (
    connected_source_mask,
    star_halo_mask,
)


def test_connected_source_mask_detects_source():

    yy, xx = np.indices((101, 101))

    image = np.exp(
        -(
            (xx - 50) ** 2
            + (yy - 50) ** 2
        ) / (2 * 5**2)
    )

    mask = connected_source_mask(
        image,
        x=50,
        y=50,
    )

    assert mask.sum() > 0
    assert mask[50, 50]


@pytest.mark.filterwarnings(
    "ignore::photutils.segmentation.detect.NoDetectionsWarning"
)
def test_connected_source_mask_background_click():

    image = np.zeros((100, 100))

    mask = connected_source_mask(
        image,
        x=50,
        y=50,
    )

    assert mask.sum() == 0


@pytest.mark.filterwarnings(
    "ignore::photutils.segmentation.detect.NoDetectionsWarning"
)
def test_connected_source_mask_preserves_shape():

    image = np.random.random((87, 113))

    mask = connected_source_mask(
        image,
        x=40,
        y=40,
    )

    assert mask.shape == image.shape


def test_connected_source_mask_dilation():

    yy, xx = np.indices((101, 101))

    image = np.exp(
        -(
            (xx - 50) ** 2
            + (yy - 50) ** 2
        ) / (2 * 5**2)
    )

    mask1 = connected_source_mask(
        image,
        x=50,
        y=50,
        dilation_iterations=0,
    )

    mask2 = connected_source_mask(
        image,
        x=50,
        y=50,
        dilation_iterations=5,
    )

    assert mask2.sum() > mask1.sum()


def test_connected_source_mask_near_border():

    yy, xx = np.indices((101, 101))

    image = np.exp(
        -(
            (xx - 5) ** 2
            + (yy - 5) ** 2
        ) / (2 * 3**2)
    )

    mask = connected_source_mask(
        image,
        x=5,
        y=5,
    )

    assert mask.sum() > 0


def test_star_halo_mask_contains_click():

    yy, xx = np.indices((101, 101))

    image = np.exp(
        -(
            (xx - 50) ** 2
            + (yy - 50) ** 2
        ) / (2 * 5**2)
    )

    mask = star_halo_mask(
        image,
        x=50,
        y=50,
        radius=20,
    )

    assert mask[50, 50]


def test_star_halo_mask_larger_than_source_mask():

    yy, xx = np.indices((101, 101))

    image = np.exp(
        -(
            (xx - 50) ** 2
            + (yy - 50) ** 2
        ) / (2 * 5**2)
    )

    source_mask = connected_source_mask(
        image,
        x=50,
        y=50,
    )

    star_mask = star_halo_mask(
        image,
        x=50,
        y=50,
        radius=20,
    )

    assert star_mask.sum() > source_mask.sum()


@pytest.mark.filterwarnings(
    "ignore::photutils.segmentation.detect.NoDetectionsWarning"
)
def test_star_halo_mask_preserves_shape():

    image = np.random.random(
        (87, 113),
    )

    mask = star_halo_mask(
        image,
        x=40,
        y=40,
    )

    assert mask.shape == image.shape