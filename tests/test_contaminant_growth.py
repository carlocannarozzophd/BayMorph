"""The growth of the contaminant masks in the sky region and the support.

A detected segment ends where the smoothed image falls below the detection
threshold, but the light of the source extends beyond it. With
``preparation.masking.contaminant_growth_pixels``, 20 px by default when the
preparation masks its detected segments, every pixel within that distance
of a masked segment is excluded from the sky region and from the likelihood
support; the target support, the geometry and the plateau are measured with
the masks as detected. A curated mask used as it is is not grown.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import baymorph.pipeline.image_preparation as image_preparation_module
from baymorph.pipeline import (
    PreparedObservation,
    load_image_preparation_products,
    pipeline_paths,
    prepare_real_image_from_config,
    resolve_config,
    save_image_preparation_products,
)
from baymorph.pipeline.config import validate_config
from baymorph.pipeline.prepared_identity import _preparation_config_identity

GROWTH = 20


def _within(mask: np.ndarray, distance: float) -> np.ndarray:
    """Return the pixels within ``distance`` of ``mask``, by brute force."""
    yy, xx = np.indices(mask.shape)
    sources = np.argwhere(mask)
    nearest = np.full(mask.shape, np.inf)

    for y, x in sources:
        nearest = np.minimum(nearest, np.hypot(yy - y, xx - x))

    return nearest <= distance


# ---------------------------------------------------------------------------
# The grown mask
# ---------------------------------------------------------------------------


def test_the_growth_takes_every_pixel_within_the_distance():
    mask = np.zeros((41, 41), dtype=bool)
    mask[20, 20] = True
    mask[5, 30:33] = True

    grown = image_preparation_module._grown_contaminant_mask(mask, 6)

    assert np.array_equal(grown, _within(mask, 6.0))
    assert grown[20, 26] and not grown[20, 27]
    assert grown[24, 24] and not grown[25, 25]


def test_no_growth_or_no_contaminant_leaves_the_mask_unchanged():
    mask = np.zeros((12, 12), dtype=bool)
    mask[3:5, 3:5] = True

    assert np.array_equal(
        image_preparation_module._grown_contaminant_mask(mask, 0),
        mask,
    )
    assert not np.any(
        image_preparation_module._grown_contaminant_mask(
            np.zeros((12, 12), dtype=bool),
            5,
        )
    )


# ---------------------------------------------------------------------------
# Configuration and identity
# ---------------------------------------------------------------------------


def _minimal_config(tmp_path: Path, *, masking=None, noise=None, data=None):
    config = {
        "object": {"name": "growth_target"},
        "data": {"image_path": str(tmp_path / "image.fits"), **(data or {})},
        "outputs": {"root": str(tmp_path / "outputs"), "overwrite": True},
    }
    preparation = {}

    if masking is not None:
        preparation["masking"] = masking

    if noise is not None:
        preparation["noise"] = noise

    if preparation:
        config["preparation"] = preparation

    return config


def test_the_default_growth_depends_on_the_mask_mode(tmp_path):
    mask_path = tmp_path / "mask.fits"
    fits.writeto(mask_path, np.zeros((10, 10), dtype=np.uint8))
    detected = resolve_config(_minimal_config(tmp_path))
    curated = {
        mode: resolve_config(
            _minimal_config(
                tmp_path,
                data={"mask_path": str(mask_path), "mask_mode": mode},
            )
        )
        for mode in ("union", "exact", "replace")
    }

    assert detected["preparation"]["masking"]["contaminant_growth_pixels"] == 20
    assert detected["preparation"]["noise"]["sky_correlation"] == "autocovariance"
    assert curated["union"]["preparation"]["masking"][
        "contaminant_growth_pixels"
    ] == 20

    for mode in ("exact", "replace"):
        assert curated[mode]["preparation"]["masking"][
            "contaminant_growth_pixels"
        ] == 0
        validate_config(curated[mode])

    declared = resolve_config(
        _minimal_config(tmp_path, masking={"contaminant_growth_pixels": 0})
    )
    assert declared["preparation"]["masking"]["contaminant_growth_pixels"] == 0


def test_a_mapping_without_the_entry_takes_the_default_of_its_mask_mode():
    growth = image_preparation_module._contaminant_growth_pixels

    assert growth({"data": {}}) == 20
    assert growth({"data": {"mask_mode": "union"}}) == 20
    assert growth({"data": {"mask_mode": "exact"}}) == 0
    assert growth({"data": {"mask_mode": "replace"}}) == 0
    assert (
        growth(
            {
                "data": {},
                "preparation": {"masking": {"contaminant_growth_pixels": 5}},
            }
        )
        == 5
    )


@pytest.mark.parametrize("growth", [-1, 2.5, True, "20", None])
def test_an_invalid_growth_is_rejected(tmp_path, growth):
    with pytest.raises(ValueError, match="contaminant_growth_pixels"):
        validate_config(
            _minimal_config(
                tmp_path,
                masking={"contaminant_growth_pixels": growth},
            )
        )


@pytest.mark.parametrize("mode", ["exact", "replace"])
def test_a_growth_with_a_curated_mask_used_as_it_is_is_rejected(tmp_path, mode):
    mask_path = tmp_path / "mask.fits"
    fits.writeto(mask_path, np.zeros((10, 10), dtype=np.uint8))

    with pytest.raises(ValueError, match="does not use"):
        validate_config(
            _minimal_config(
                tmp_path,
                masking={"contaminant_growth_pixels": GROWTH},
                data={"mask_path": str(mask_path), "mask_mode": mode},
            )
        )


@pytest.mark.parametrize(
    "noise, message",
    [
        ({"sky_correlation": "high_pass"}, "sky_correlation"),
        ({"sky_correlation": ["field_separated"]}, "sky_correlation"),
        ({"correlation": "field_separated"}, "preparation.noise"),
    ],
)
def test_an_invalid_noise_entry_is_rejected(tmp_path, noise, message):
    with pytest.raises(ValueError, match=message):
        validate_config(_minimal_config(tmp_path, noise=noise))


@pytest.mark.parametrize("growth", [0, 10, 30])
def test_the_field_separated_estimator_requires_the_validated_growth(
    tmp_path,
    growth,
):
    with pytest.raises(ValueError, match="contaminant_growth_pixels: 20"):
        validate_config(
            _minimal_config(
                tmp_path,
                masking={"contaminant_growth_pixels": growth},
                noise={"sky_correlation": "field_separated"},
            )
        )

    for masking in ({"contaminant_growth_pixels": GROWTH}, None):
        validate_config(
            _minimal_config(
                tmp_path,
                masking=masking,
                noise={"sky_correlation": "field_separated"},
            )
        )


def test_the_field_separated_estimator_is_refused_with_a_curated_mask(tmp_path):
    mask_path = tmp_path / "mask.fits"
    fits.writeto(mask_path, np.zeros((10, 10), dtype=np.uint8))

    with pytest.raises(ValueError, match="contaminant_growth_pixels: 20"):
        validate_config(
            _minimal_config(
                tmp_path,
                noise={"sky_correlation": "field_separated"},
                data={"mask_path": str(mask_path), "mask_mode": "exact"},
            )
        )


def _identity(tmp_path, preparation):
    config = {
        "object": {"name": "growth_target"},
        "data": {"image_path": "image.fits"},
        "preparation": preparation,
    }

    return _preparation_config_identity(
        config,
        image_path=tmp_path / "image.fits",
        mask_identity=None,
    )[1]


def test_the_defaults_keep_the_identity_of_older_products(tmp_path):
    older = _identity(
        tmp_path,
        {"masking": {"negative_threshold_sigma": 5.0}, "noise": {}},
    )
    declared = _identity(
        tmp_path,
        {
            "masking": {
                "negative_threshold_sigma": 5.0,
                "contaminant_growth_pixels": 0,
            },
            "noise": {"sky_correlation": "autocovariance"},
        },
    )
    grown = _identity(
        tmp_path,
        {
            "masking": {
                "negative_threshold_sigma": 5.0,
                "contaminant_growth_pixels": GROWTH,
            },
            "noise": {"sky_correlation": "autocovariance"},
        },
    )
    separated = _identity(
        tmp_path,
        {
            "masking": {
                "negative_threshold_sigma": 5.0,
                "contaminant_growth_pixels": GROWTH,
            },
            "noise": {"sky_correlation": "field_separated"},
        },
    )

    assert declared == older
    assert len({older, grown, separated}) == 3

    default = resolve_config(_minimal_config(tmp_path))["preparation"]
    without = copy.deepcopy(default)
    without["masking"]["contaminant_growth_pixels"] = 0
    stripped = copy.deepcopy(without)
    del stripped["masking"]["contaminant_growth_pixels"]

    assert _identity(tmp_path, without) == _identity(tmp_path, stripped)
    assert _identity(tmp_path, default) != _identity(tmp_path, without)


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def _image_with_a_neighbour(path: Path) -> None:
    """Write a galaxy at the centre and a compact neighbour in the sky."""
    rng = np.random.default_rng(20261006)
    size = 120
    yy, xx = np.mgrid[0:size, 0:size]
    image = 100.0 + rng.normal(0.0, 2.0, size=(size, size))
    image += 400.0 * np.exp(
        -0.5 * ((yy - 60.0) ** 2 + (xx - 60.0) ** 2) / 6.0**2
    )
    image += 300.0 * np.exp(
        -0.5 * ((yy - 22.0) ** 2 + (xx - 98.0) ** 2) / 1.5**2
    )

    fits.writeto(path, image, overwrite=True)


def _preparation_config(tmp_path: Path, growth: int | None) -> dict:
    config = {
        "object": {"name": f"growth_{growth}"},
        "data": {
            "image_path": str(tmp_path / "image.fits"),
            "target_position": {"x": 60, "y": 60},
            "cutout": {"mode": "fixed", "size": 100},
        },
        "outputs": {"root": str(tmp_path / "runs"), "overwrite": True},
    }

    if growth is not None:
        config["preparation"] = {
            "masking": {"contaminant_growth_pixels": growth}
        }

    return resolve_config(config)


@pytest.fixture
def prepared(tmp_path):
    """Prepare the same image without growth and with the default growth."""
    _image_with_a_neighbour(tmp_path / "image.fits")

    return (
        prepare_real_image_from_config(_preparation_config(tmp_path, 0)),
        prepare_real_image_from_config(_preparation_config(tmp_path, None)),
    )


def test_the_grown_masks_define_the_support_and_the_sky(prepared):
    detected, grown = prepared
    expected = _within(detected.contaminant_mask, GROWTH)

    assert np.any(detected.contaminant_mask)
    assert grown.contaminant_growth_pixels == GROWTH
    assert np.array_equal(grown.contaminant_mask, detected.contaminant_mask)
    assert np.array_equal(grown.fitting_mask, expected)
    assert not np.any(grown.sky_region.mask & expected)
    assert np.any(detected.sky_region.mask & expected)
    assert grown.sky_region.n_pixels < detected.sky_region.n_pixels


def test_the_target_geometry_and_plateau_use_the_masks_as_detected(prepared):
    detected, grown = prepared

    assert np.array_equal(grown.target_segment, detected.target_segment)
    assert grown.geometry == detected.geometry
    assert grown.plateau.radius == detected.plateau.radius
    assert detected.contaminant_growth_pixels == 0
    assert np.array_equal(detected.fitting_mask, detected.contaminant_mask)


def test_the_growth_survives_the_round_trip(tmp_path, prepared):
    _, grown = prepared
    config = _preparation_config(tmp_path, None)
    config["outputs"]["overwrite"] = False
    metadata_path = (
        pipeline_paths(config)["image_preparation"]["dir"]
        / "image_preparation_metadata.json"
    )

    loaded = load_image_preparation_products(config)

    assert loaded.contaminant_growth_pixels == GROWTH
    assert np.array_equal(loaded.fitting_mask, grown.fitting_mask)
    assert np.array_equal(loaded.sky_region.mask, grown.sky_region.mask)
    assert (
        json.loads(metadata_path.read_text())["contaminant_masking"][
            "growth_pixels"
        ]
        == GROWTH
    )


def test_products_of_another_growth_are_not_reused(tmp_path, prepared):
    config = _preparation_config(tmp_path, None)
    config["preparation"]["masking"]["contaminant_growth_pixels"] = 10
    config["outputs"]["overwrite"] = False

    with pytest.raises(RuntimeError, match="do not match"):
        load_image_preparation_products(config)


# ---------------------------------------------------------------------------
# Persisted products
# ---------------------------------------------------------------------------


def _bundle(tmp_path, *, growth, persisted_growth, fitting_mask):
    """Return a resolved config and products with two contaminant segments."""
    image = np.arange(400, dtype=float).reshape(20, 20)
    fits.writeto(tmp_path / "image.fits", image)

    resolved = resolve_config(
        {
            "object": {"name": "growth_target"},
            "data": {
                "image_path": str(tmp_path / "image.fits"),
                "target_position": {"x": 10, "y": 10},
                "cutout": {"mode": "fixed", "size": 20},
            },
            "preparation": {
                "masking": {"contaminant_growth_pixels": growth},
            },
            "outputs": {"root": str(tmp_path / "outputs"), "overwrite": True},
        }
    )
    pipeline_paths(resolved)["image_preparation"]["dir"].mkdir(
        parents=True,
        exist_ok=True,
    )

    segmentation = np.zeros((20, 20), dtype=int)
    segmentation[8:12, 8:12] = 1
    segmentation[0:2, 0:2] = 2
    segmentation[17:20, 16:20] = 3
    contaminants = np.isin(segmentation, [2, 3])
    masks = {
        "detected": contaminants.copy(),
        "grown": _within(contaminants, persisted_growth),
    }

    products = PreparedObservation(
        cutout=image.copy(),
        prepared_data=image - 1.5,
        noise=np.full((20, 20), 3.0),
        automatic_mask=segmentation > 0,
        fitting_mask=masks[fitting_mask],
        target_segment=segmentation == 1,
        contaminant_mask=contaminants,
        segmentation=segmentation,
        background=1.5,
        target_x=10,
        target_y=10,
        target_label=1,
        original_image=image.copy(),
        cutout_bounds={"xmin": 0, "xmax": 20, "ymin": 0, "ymax": 20},
        contaminant_rule="local_excess",
        contaminant_growth_pixels=persisted_growth,
    )

    return resolved, products


def test_a_support_that_readmits_the_grown_pixels_is_refused(tmp_path):
    resolved, products = _bundle(
        tmp_path,
        growth=3,
        persisted_growth=3,
        fitting_mask="detected",
    )

    with pytest.raises(RuntimeError, match="fitting mask"):
        save_image_preparation_products(resolved, products)


def test_a_growth_other_than_the_configured_one_is_refused(tmp_path):
    resolved, products = _bundle(
        tmp_path,
        growth=3,
        persisted_growth=2,
        fitting_mask="grown",
    )

    with pytest.raises(RuntimeError, match="growth of the contaminant masks"):
        save_image_preparation_products(resolved, products)


def test_a_consistent_grown_bundle_is_saved_and_reloaded(tmp_path):
    resolved, products = _bundle(
        tmp_path,
        growth=3,
        persisted_growth=3,
        fitting_mask="grown",
    )

    save_image_preparation_products(resolved, products)
    loaded = load_image_preparation_products(resolved)

    assert loaded.contaminant_growth_pixels == 3
    assert np.array_equal(loaded.fitting_mask, products.fitting_mask)
