"""Tests for likelihood inference-region construction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from baymorph.pipeline.inference_region import (
    build_inference_region,
)


def _products(
    shape: tuple[int, int] = (9, 9),
):
    data = np.arange(
        np.prod(shape),
        dtype=float,
    ).reshape(shape)

    return SimpleNamespace(
        prepared_data=data,
        noise=np.full(
            shape,
            2.0,
            dtype=float,
        ),
        fitting_mask=np.zeros(
            shape,
            dtype=bool,
        ),
        target_x=4,
        target_y=4,
    )


def test_full_region_preserves_prepared_footprint() -> None:
    products = _products()

    region = build_inference_region(
        products,
        mode="full",
        reference_x=3.5,
        reference_y=4.25,
    )

    assert region.shape == (9, 9)
    assert region.bounds == {
        "xmin": 0,
        "xmax": 9,
        "ymin": 0,
        "ymax": 9,
    }
    assert region.target_x == pytest.approx(
        3.5
    )
    assert region.target_y == pytest.approx(
        4.25
    )
    assert region.radius is None
    assert region.n_rendered_pixels == 81
    assert region.n_usable_pixels == 81

    np.testing.assert_array_equal(
        region.data,
        products.prepared_data,
    )
    np.testing.assert_array_equal(
        region.noise,
        products.noise,
    )
    np.testing.assert_array_equal(
        region.mask,
        products.fitting_mask,
    )


def test_circular_region_builds_minimal_odd_crop() -> None:
    products = _products()

    region = build_inference_region(
        products,
        mode="circular",
        radius=2.0,
        reference_x=4.0,
        reference_y=4.0,
    )

    assert region.shape == (5, 5)
    assert region.bounds == {
        "xmin": 2,
        "xmax": 7,
        "ymin": 2,
        "ymax": 7,
    }
    assert region.target_x == pytest.approx(
        2.0
    )
    assert region.target_y == pytest.approx(
        2.0
    )

    assert not region.mask[2, 2]
    assert region.mask[0, 0]
    assert region.mask[0, 4]
    assert region.mask[4, 0]
    assert region.mask[4, 4]

    assert region.n_rendered_pixels == 25
    assert region.n_usable_pixels == 13


def test_circular_region_uses_exact_fractional_reference() -> None:
    products = _products()

    region = build_inference_region(
        products,
        mode="circular",
        radius=2.0,
        reference_x=4.25,
        reference_y=3.75,
    )

    assert region.bounds == {
        "xmin": 2,
        "xmax": 7,
        "ymin": 2,
        "ymax": 7,
    }
    assert region.target_x == pytest.approx(
        2.25
    )
    assert region.target_y == pytest.approx(
        1.75
    )

    preparation_x, preparation_y = (
        region.to_preparation(
            x=region.target_x,
            y=region.target_y,
        )
    )

    assert preparation_x == pytest.approx(
        4.25
    )
    assert preparation_y == pytest.approx(
        3.75
    )


def test_circular_region_preserves_existing_mask() -> None:
    products = _products()
    products.fitting_mask[4, 4] = True

    region = build_inference_region(
        products,
        mode="circular",
        radius=2.0,
        reference_x=4.0,
        reference_y=4.0,
    )

    assert region.mask[2, 2]

    assert products.fitting_mask[4, 4]
    assert np.count_nonzero(
        products.fitting_mask
    ) == 1


def test_circular_region_does_not_modify_preparation_mask() -> None:
    products = _products()
    original = products.fitting_mask.copy()

    build_inference_region(
        products,
        mode="circular",
        radius=2.0,
        reference_x=4.0,
        reference_y=4.0,
    )

    np.testing.assert_array_equal(
        products.fitting_mask,
        original,
    )


def test_circular_region_rejects_out_of_bounds_crop() -> None:
    products = _products()

    with pytest.raises(
        ValueError,
        match="does not fit",
    ):
        build_inference_region(
            products,
            mode="circular",
            radius=4.0,
            reference_x=1.0,
            reference_y=4.0,
        )


@pytest.mark.parametrize(
    ("mode", "radius", "message"),
    [
        (
            "unknown",
            None,
            "mode",
        ),
        (
            "full",
            2.0,
            "radius",
        ),
        (
            "circular",
            None,
            "radius",
        ),
        (
            "circular",
            0.0,
            "positive",
        ),
        (
            "circular",
            -1.0,
            "positive",
        ),
    ],
)
def test_invalid_region_configuration(
    mode,
    radius,
    message,
) -> None:
    products = _products()

    with pytest.raises(
        ValueError,
        match=message,
    ):
        build_inference_region(
            products,
            mode=mode,
            radius=radius,
        )


def test_region_rejects_mismatched_preparation_arrays() -> None:
    products = _products()
    products.noise = np.ones(
        (8, 9),
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="noise map",
    ):
        build_inference_region(
            products
        )


def test_region_rejects_reference_outside_prepared_image() -> None:
    products = _products()

    with pytest.raises(
        ValueError,
        match="reference coordinates",
    ):
        build_inference_region(
            products,
            reference_x=9.0,
            reference_y=4.0,
        )
