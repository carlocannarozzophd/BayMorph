import numpy as np
import pytest

from baymorph.core import (
    connected_source_mask,
    star_halo_mask,
    intersection,
    union,
)


def test_connected_source_mask_detects_source():

    yy, xx = np.indices((101, 101))

    image = np.exp(-((xx - 50) ** 2 + (yy - 50) ** 2) / (2 * 5**2))

    mask = connected_source_mask(
        image,
        x=50,
        y=50,
    )

    assert mask.sum() > 0
    assert mask[50, 50]


@pytest.mark.filterwarnings("ignore::photutils.segmentation.detect.NoDetectionsWarning")
def test_connected_source_mask_background_click():

    image = np.zeros((100, 100))

    mask = connected_source_mask(
        image,
        x=50,
        y=50,
    )

    assert mask.sum() == 0


@pytest.mark.filterwarnings("ignore::photutils.segmentation.detect.NoDetectionsWarning")
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

    image = np.exp(-((xx - 50) ** 2 + (yy - 50) ** 2) / (2 * 5**2))

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

    image = np.exp(-((xx - 5) ** 2 + (yy - 5) ** 2) / (2 * 3**2))

    mask = connected_source_mask(
        image,
        x=5,
        y=5,
    )

    assert mask.sum() > 0


def test_star_halo_mask_contains_click():

    yy, xx = np.indices((101, 101))

    image = np.exp(-((xx - 50) ** 2 + (yy - 50) ** 2) / (2 * 5**2))

    mask = star_halo_mask(
        image,
        x=50,
        y=50,
        radius=20,
    )

    assert mask[50, 50]


def test_star_halo_mask_larger_than_source_mask():

    yy, xx = np.indices((101, 101))

    image = np.exp(-((xx - 50) ** 2 + (yy - 50) ** 2) / (2 * 5**2))

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


@pytest.mark.filterwarnings("ignore::photutils.segmentation.detect.NoDetectionsWarning")
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


def test_union_combines_masks_without_mutating_inputs():
    first = np.array(
        [
            [True, False],
            [False, False],
        ],
        dtype=bool,
    )
    second = np.array(
        [
            [False, True],
            [False, False],
        ],
        dtype=bool,
    )

    first_before = first.copy()
    second_before = second.copy()

    result = union(first, second)

    expected = np.array(
        [
            [True, True],
            [False, False],
        ],
        dtype=bool,
    )

    assert result.dtype == bool
    assert np.array_equal(result, expected)
    assert np.array_equal(first, first_before)
    assert np.array_equal(second, second_before)
    assert not np.shares_memory(result, first)
    assert not np.shares_memory(result, second)


def test_union_accepts_multiple_masks():
    first = np.eye(3, dtype=bool)
    second = np.fliplr(first)
    third = np.zeros((3, 3), dtype=bool)

    result = union(
        first,
        second,
        third,
    )

    assert result.sum() == 5


def test_intersection_combines_masks_without_mutating_inputs():
    first = np.array(
        [
            [True, True],
            [False, True],
        ],
        dtype=bool,
    )
    second = np.array(
        [
            [True, False],
            [False, True],
        ],
        dtype=bool,
    )

    first_before = first.copy()
    second_before = second.copy()

    result = intersection(first, second)

    expected = np.array(
        [
            [True, False],
            [False, True],
        ],
        dtype=bool,
    )

    assert result.dtype == bool
    assert np.array_equal(result, expected)
    assert np.array_equal(first, first_before)
    assert np.array_equal(second, second_before)
    assert not np.shares_memory(result, first)
    assert not np.shares_memory(result, second)


@pytest.mark.parametrize(
    "operation",
    [union, intersection],
)
def test_mask_collection_operations_require_two_masks(
    operation,
):
    with pytest.raises(
        ValueError,
        match="At least two masks",
    ):
        operation(
            np.zeros((2, 2), dtype=bool),
        )


@pytest.mark.parametrize(
    "operation",
    [union, intersection],
)
def test_mask_collection_operations_reject_shape_mismatch(
    operation,
):
    with pytest.raises(
        ValueError,
        match="same shape",
    ):
        operation(
            np.zeros((2, 2), dtype=bool),
            np.zeros((3, 3), dtype=bool),
        )


@pytest.mark.parametrize(
    "operation",
    [union, intersection],
)
def test_mask_collection_operations_reject_non_2d_input(
    operation,
):
    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        operation(
            np.zeros((2, 2), dtype=bool),
            np.zeros((2, 2, 2), dtype=bool),
        )
