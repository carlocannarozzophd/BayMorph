from types import SimpleNamespace

import numpy as np

from baymorph.pipeline import (
    build_image_preparation_runtime,
)


def test_build_image_preparation_runtime():
    products = SimpleNamespace(
        prepared_data=np.zeros((4, 5)),
        noise=np.full((4, 5), 2.0),
        background=1.5,
        target_x=2,
        target_y=1,
        target_label=3,
        cutout_mode="fixed",
        cutout_bounds={
            "xmin": 10,
            "xmax": 15,
            "ymin": 20,
            "ymax": 24,
        },
        automatic_mask=np.array(
            [
                [True, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ],
            dtype=bool,
        ),
        external_mask=np.array(
            [
                [False, True, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ],
            dtype=bool,
        ),
        contaminant_mask=np.zeros((4, 5), dtype=bool),
        fitting_mask=np.array(
            [
                [True, True, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ],
            dtype=bool,
        ),
        geometry=SimpleNamespace(
            x0=2.1,
            y0=1.9,
            q=0.8,
            theta=0.3,
        ),
        plateau=SimpleNamespace(
            radius=12.0,
            index=4,
            background_level=1.4,
            noise_sigma=0.2,
            window_start=3,
            window_stop=7,
        ),
        sky_region=SimpleNamespace(
            inner_radius=17.0,
            outer_radius=25.0,
            n_pixels=42,
            coverage=0.84,
        ),
    )

    runtime = build_image_preparation_runtime(
        products,
        observation={
            "filter": "F200W",
            "pixel_scale_arcsec": 0.03,
        },
    )

    preparation = runtime["image_preparation"]

    assert runtime["observation"]["filter"] == "F200W"
    assert preparation["cutout"]["shape"] == [4, 5]
    assert preparation["background"]["level"] == 1.5
    assert preparation["noise"]["median"] == 2.0
    assert preparation["target"]["geometry"]["q"] == 0.8
    assert preparation["plateau"]["radius"] == 12.0
    assert preparation["sky_region"]["n_pixels"] == 42
    assert preparation["masking"]["automatic_fraction"] == 0.05
    assert preparation["masking"]["external_fraction"] == 0.05
    assert preparation["masking"]["fitting_fraction"] == 0.1


def test_build_image_preparation_runtime_without_optional_products():
    products = SimpleNamespace(
        prepared_data=np.zeros((3, 3)),
        noise=np.ones((3, 3)),
        background=0.0,
        target_x=1,
        target_y=1,
        target_label=1,
        cutout_mode="fixed",
        cutout_bounds=None,
        automatic_mask=np.zeros((3, 3), dtype=bool),
        external_mask=None,
        contaminant_mask=np.zeros((3, 3), dtype=bool),
        fitting_mask=np.zeros((3, 3), dtype=bool),
        geometry=None,
        plateau=None,
        sky_region=None,
    )

    runtime = build_image_preparation_runtime(
        products,
    )

    preparation = runtime["image_preparation"]

    assert runtime["observation"] is None
    assert preparation["target"]["geometry"] is None
    assert preparation["plateau"] is None
    assert preparation["sky_region"] is None
    assert preparation["masking"]["external_fraction"] is None
