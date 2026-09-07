import numpy as np

from baymorph.pipeline import (
    PreparedObservation,
    load_image_preparation_products,
    make_target_cutout,
    pipeline_paths,
    resolve_config,
    resolve_fixed_cutout,
    save_image_preparation_products,
)


def test_make_target_cutout_returns_expected_shape():
    data = np.zeros((100, 100))

    cutout_definition = resolve_fixed_cutout(
        image_shape=data.shape,
        center_x=50,
        center_y=50,
        size=40,
    )

    cutout = make_target_cutout(
        data,
        cutout_definition,
    )

    assert cutout.shape == (40, 40)


def test_make_target_cutout_shifts_cutout_inside_image():
    data = np.zeros((100, 100))

    cutout_definition = resolve_fixed_cutout(
        image_shape=data.shape,
        center_x=5,
        center_y=5,
        size=40,
    )

    cutout = make_target_cutout(
        data,
        cutout_definition,
    )

    assert cutout.shape == (40, 40)
    assert cutout_definition.xmin == 0
    assert cutout_definition.xmax == 40
    assert cutout_definition.ymin == 0
    assert cutout_definition.ymax == 40


def test_save_and_load_image_preparation_products_roundtrip(tmp_path):
    config = {
        "object": {"name": "example_galaxy"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path / "outputs"), "overwrite": True},
    }

    resolved = resolve_config(config)

    paths = pipeline_paths(resolved)["image_preparation"]
    paths["dir"].mkdir(parents=True, exist_ok=True)

    products = PreparedObservation(
        cutout=np.ones((10, 10)),
        prepared_data=np.ones((10, 10)) * 2,
        noise=np.ones((10, 10)) * 3,
        automatic_mask=np.zeros((10, 10), dtype=bool),
        fitting_mask=np.eye(10, dtype=bool),
        external_mask=np.eye(10, dtype=bool),
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
    assert np.array_equal(loaded.fitting_mask, products.fitting_mask)
    assert np.array_equal(loaded.external_mask, products.external_mask)
    assert np.array_equal(loaded.target_segment, products.target_segment)
    assert np.array_equal(loaded.segmentation, products.segmentation)
    assert loaded.background == products.background
    assert loaded.target_x == products.target_x
    assert loaded.target_y == products.target_y
    assert loaded.target_label == products.target_label


def test_prepared_observation_alias():
    """The legacy public name must remain available during v0.4."""



def test_prepare_real_image_from_config_combines_external_mask(
    tmp_path,
):
    from astropy.io import fits

    from baymorph.pipeline import prepare_real_image_from_config

    yy, xx = np.indices((80, 80))

    radius_squared = (xx - 40.0) ** 2 + (yy - 40.0) ** 2

    image = 100.0 * np.exp(-0.5 * radius_squared / 4.0**2)

    external_mask = np.zeros_like(image, dtype=np.uint8)
    external_mask[35:38, 30:33] = 1

    image_path = tmp_path / "image.fits"
    external_mask_path = tmp_path / "external_mask.fits"

    fits.writeto(
        image_path,
        image,
    )
    fits.writeto(
        external_mask_path,
        external_mask,
    )

    config = {
        "object": {
            "name": "synthetic",
        },
        "data": {
            "image_path": str(image_path),
            "mask_path": str(external_mask_path),
            "target_position": {
                "x": 40,
                "y": 40,
            },
            "cutout": {
                "mode": "fixed",
                "size": 70,
            },
        },
        "preparation": {
            "background": {
                "box_size": 10,
                "filter_size": 3,
            },
            "detection": {
                "n_sigma": 2.0,
                "n_pixels": 5,
                "n_levels": 32,
            },
        },
        "outputs": {
            "root": str(tmp_path / "outputs"),
            "overwrite": True,
        },
    }

    products = prepare_real_image_from_config(
        config,
        save=False,
    )

    expected_external_mask = external_mask[
        products.cutout_bounds["ymin"] : products.cutout_bounds["ymax"],
        products.cutout_bounds["xmin"] : products.cutout_bounds["xmax"],
    ].astype(bool)

    assert products.external_mask is not None
    assert np.array_equal(
        products.external_mask,
        expected_external_mask,
    )
    assert np.all(products.fitting_mask[products.external_mask])
