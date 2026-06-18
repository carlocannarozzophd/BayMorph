from astropy.io import fits
import numpy as np
import pytest

from baymorph import (
    ImagePreparationProducts,
    load_image_preparation_products,
    make_target_cutout,
    pipeline_paths,
    resolve_config,
    save_image_preparation_products,
)


def test_make_target_cutout_returns_expected_shape():
    data = np.zeros((100, 100))

    cutout, bounds = make_target_cutout(
        data,
        target_x=50,
        target_y=50,
        cutout_size=40,
    )

    assert cutout.shape == (40, 40)
    assert bounds == {
        "x_min": 30,
        "x_max": 70,
        "y_min": 30,
        "y_max": 70,
    }


def test_make_target_cutout_raises_if_outside_image():
    data = np.zeros((100, 100))

    with pytest.raises(ValueError, match="outside the image boundaries"):
        make_target_cutout(
            data,
            target_x=10,
            target_y=10,
            cutout_size=40,
        )


def test_save_and_load_image_preparation_products_roundtrip(tmp_path):
    config = {
        "object": {"name": "NGC1234"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path / "outputs"), "overwrite": True},
    }

    resolved = resolve_config(config)

    paths = pipeline_paths(resolved)["image_preparation"]
    paths["dir"].mkdir(parents=True, exist_ok=True)

    products = ImagePreparationProducts(
        cutout=np.ones((10, 10)),
        prepared_data=np.ones((10, 10)) * 2,
        noise=np.ones((10, 10)) * 3,
        automatic_mask=np.zeros((10, 10), dtype=bool),
        target_segment=np.zeros((10, 10), dtype=bool),
        contaminant_mask=np.ones((10, 10), dtype=bool),
        segmentation=np.ones((10, 10), dtype=int),
        background=1.5,
        target_x=5,
        target_y=6,
        target_label=2,
    )

    save_image_preparation_products(resolved, products)

    loaded = load_image_preparation_products(resolved)

    assert np.allclose(loaded.cutout, products.cutout)
    assert np.allclose(loaded.prepared_data, products.prepared_data)
    assert np.allclose(loaded.noise, products.noise)
    assert np.array_equal(loaded.contaminant_mask, products.contaminant_mask)
    assert np.array_equal(loaded.target_segment, products.target_segment)
    assert np.array_equal(loaded.segmentation, products.segmentation)
    assert loaded.background == products.background
    assert loaded.target_x == products.target_x
    assert loaded.target_y == products.target_y
    assert loaded.target_label == products.target_label