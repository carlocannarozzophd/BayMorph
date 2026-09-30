import numpy as np
import pytest
from astropy.io import fits


from baymorph.core import (
    build_fitting_mask,
    connected_source_mask,
    star_halo_mask,
    intersection,
    union,
    load_external_mask,
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


def test_load_external_mask_converts_nonzero_pixels_to_bool(
    tmp_path,
):
    path = tmp_path / "mask.fits"

    data = np.array(
        [
            [0, 1, 0],
            [2, 0, -1],
        ],
        dtype=np.int16,
    )

    fits.writeto(
        path,
        data,
    )

    mask = load_external_mask(
        path,
        expected_shape=(2, 3),
    )

    expected = np.array(
        [
            [False, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )

    assert mask.dtype == bool
    assert np.array_equal(
        mask,
        expected,
    )


def test_load_external_mask_rejects_missing_file(
    tmp_path,
):
    path = tmp_path / "missing.fits"

    with pytest.raises(
        FileNotFoundError,
        match="External mask file not found",
    ):
        load_external_mask(path)


def test_load_external_mask_rejects_non_2d_data(
    tmp_path,
):
    path = tmp_path / "mask.fits"

    fits.writeto(
        path,
        np.zeros((2, 3, 4)),
    )

    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        load_external_mask(path)


def test_load_external_mask_rejects_non_finite_values(
    tmp_path,
):
    path = tmp_path / "mask.fits"

    data = np.zeros((4, 5))
    data[1, 2] = np.nan

    fits.writeto(
        path,
        data,
    )

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        load_external_mask(path)


def test_load_external_mask_rejects_shape_mismatch(
    tmp_path,
):
    path = tmp_path / "mask.fits"

    fits.writeto(
        path,
        np.zeros((4, 5)),
    )

    with pytest.raises(
        ValueError,
        match="does not match the science image",
    ):
        load_external_mask(
            path,
            expected_shape=(5, 4),
        )


def test_build_fitting_mask_exact_replaces_automatic_mask():
    automatic = np.array(
        [
            [True, True, False],
            [False, False, True],
        ],
        dtype=bool,
    )
    curated = np.array(
        [
            [False, True, False],
            [True, False, False],
        ],
        dtype=bool,
    )

    result = build_fitting_mask(
        automatic,
        external_mask=curated,
        mode="exact",
    )

    assert np.array_equal(result, curated)
    assert not result[0, 0]
    assert not np.shares_memory(result, curated)


def test_build_fitting_mask_replace_alias_matches_exact():
    automatic = np.ones((3, 3), dtype=bool)
    curated = np.eye(3, dtype=bool)

    result = build_fitting_mask(
        automatic,
        external_mask=curated,
        mode="replace",
    )

    assert np.array_equal(result, curated)


def test_build_fitting_mask_union_preserves_legacy_behavior():
    automatic = np.array([[True, False]], dtype=bool)
    curated = np.array([[False, True]], dtype=bool)

    result = build_fitting_mask(
        automatic,
        external_mask=curated,
    )

    assert result.all()


@pytest.mark.parametrize("mode", ["exact", "replace"])
def test_build_fitting_mask_replacement_requires_curated_mask(mode):
    with pytest.raises(
        ValueError,
        match="external mask is required",
    ):
        build_fitting_mask(
            np.zeros((2, 2), dtype=bool),
            mode=mode,
        )


def test_build_fitting_mask_rejects_non_finite_input():
    automatic = np.zeros((2, 2))
    automatic[0, 1] = np.nan

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        build_fitting_mask(automatic)


def test_build_fitting_mask_rejects_unknown_mode():
    with pytest.raises(
        ValueError,
        match="must be one of",
    ):
        build_fitting_mask(
            np.zeros((2, 2), dtype=bool),
            mode="subtract",
        )
