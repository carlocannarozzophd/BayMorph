"""Tests for real-fit inference-region routing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from baymorph.pipeline.real_fit import (
    _build_fit_inference_region,
)


def _products():
    shape = (21, 21)

    return SimpleNamespace(
        prepared_data=np.arange(
            np.prod(shape),
            dtype=float,
        ).reshape(shape),
        noise=np.full(
            shape,
            2.0,
            dtype=float,
        ),
        fitting_mask=np.zeros(
            shape,
            dtype=bool,
        ),
        target_x=10,
        target_y=10,
    )


def test_full_fit_region_preserves_preparation_coordinates() -> None:
    products = _products()

    region = _build_fit_inference_region(
        products,
        references={
            "target_x": 9.5,
            "target_y": 10.25,
        },
        region_cfg={
            "mode": "full",
            "radius": None,
        },
    )

    assert region.mode == "full"
    assert region.shape == (21, 21)
    assert region.bounds == {
        "xmin": 0,
        "xmax": 21,
        "ymin": 0,
        "ymax": 21,
    }
    assert region.target_x == pytest.approx(
        9.5
    )
    assert region.target_y == pytest.approx(
        10.25
    )

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


def test_circular_fit_region_translates_prior_reference() -> None:
    products = _products()

    region = _build_fit_inference_region(
        products,
        references={
            "target_x": 10.25,
            "target_y": 9.75,
        },
        region_cfg={
            "mode": "circular",
            "radius": 4.0,
        },
    )

    assert region.mode == "circular"
    assert region.shape == (9, 9)
    assert region.bounds == {
        "xmin": 6,
        "xmax": 15,
        "ymin": 6,
        "ymax": 15,
    }

    assert region.target_x == pytest.approx(
        4.25
    )
    assert region.target_y == pytest.approx(
        3.75
    )

    x, y = region.to_preparation(
        x=region.target_x,
        y=region.target_y,
    )

    assert x == pytest.approx(
        10.25
    )
    assert y == pytest.approx(
        9.75
    )


def test_fit_region_rejects_non_mapping_configuration() -> None:
    products = _products()

    with pytest.raises(
        TypeError,
        match="fit.region must be a mapping",
    ):
        _build_fit_inference_region(
            products,
            references={
                "target_x": 10.0,
                "target_y": 10.0,
            },
            region_cfg="circular",
        )


def test_fit_region_rejects_unknown_configuration_entries() -> None:
    products = _products()

    with pytest.raises(
        ValueError,
        match="Unsupported fit.region entries",
    ):
        _build_fit_inference_region(
            products,
            references={
                "target_x": 10.0,
                "target_y": 10.0,
            },
            region_cfg={
                "mode": "circular",
                "radius": 4.0,
                "geometry": "ellipse",
            },
        )
