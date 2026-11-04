"""The single product store of the new preparation.

Every stage saves one product the same way: arrays as ``.npy`` files, images
as FITS files, a record, a manifest of digests and a completion marker
written last. A product is published whole or not at all, never rewritten in
place, refused when any byte changed, and refused as stale when it was built
from other inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from baymorph.preparation.identity import sha256_hex
from baymorph.preparation.store import (
    COMPLETE_NAME,
    MANIFEST_NAME,
    RECORD_NAME,
    CorruptProductError,
    IncompleteProductError,
    ProductExistsError,
    ProductStatus,
    ProductStore,
    StaleProductError,
    StageProduct,
)

IDENTITY = sha256_hex(b"first identity")
OTHER_IDENTITY = sha256_hex(b"other identity")


def _product(identity: str = IDENTITY, **overrides) -> StageProduct:
    rng = np.random.default_rng(7)
    arguments = {
        "stage": "sky",
        "version": 1,
        "identity": identity,
        "record": {
            "background": 0.25,
            "sky_scale": 2.6,
            "pixels": 311604,
            "notes": ["level from complete annuli"],
        },
        "arrays": {
            "sky_region": rng.random((12, 10)) > 0.5,
            "profile": rng.normal(size=30),
        },
        "images": {
            "prepared_image": rng.normal(size=(12, 10)).astype(np.float32),
            "fitting_mask": (rng.random((12, 10)) > 0.8).astype(np.uint8),
        },
    }
    arguments.update(overrides)
    return StageProduct(**arguments)


def _directory_bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(directory.iterdir())
        if path.name != COMPLETE_NAME
    }


def test_round_trip_returns_the_same_product(tmp_path):
    store = ProductStore(tmp_path / "run" / "preparation")
    product = _product()

    directory = store.save(product)
    loaded = store.load("sky", identity=IDENTITY)

    assert directory == tmp_path / "run" / "preparation" / "sky"
    assert loaded.stage == "sky"
    assert loaded.version == 1
    assert loaded.identity == IDENTITY
    assert dict(loaded.record) == dict(product.record)
    assert set(loaded.arrays) == {"sky_region", "profile"}
    assert set(loaded.images) == {"prepared_image", "fitting_mask"}

    for name, array in product.arrays.items():
        np.testing.assert_array_equal(loaded.arrays[name], array)
        assert loaded.arrays[name].dtype == array.dtype

    for name, image in product.images.items():
        np.testing.assert_array_equal(loaded.images[name], image)
        assert loaded.images[name].dtype == image.dtype
        assert loaded.images[name].dtype.isnative


def test_loaded_arrays_and_record_are_read_only(tmp_path):
    store = ProductStore(tmp_path)
    store.save(_product())
    loaded = store.load("sky")

    with pytest.raises(ValueError):
        loaded.arrays["profile"][0] = 1.0

    with pytest.raises(TypeError):
        loaded.record["background"] = 0.0


def test_the_layout_has_one_file_per_array_and_image(tmp_path):
    store = ProductStore(tmp_path)
    directory = store.save(_product())

    assert sorted(path.name for path in directory.iterdir()) == sorted(
        [
            "sky_region.npy",
            "profile.npy",
            "prepared_image.fits",
            "fitting_mask.fits",
            RECORD_NAME,
            MANIFEST_NAME,
            COMPLETE_NAME,
        ]
    )

    with fits.open(directory / "prepared_image.fits") as hdul:
        assert hdul[0].header["BMSTAGE"] == "sky"
        assert hdul[0].header["BMIDENT"] == IDENTITY

    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_the_same_product_gives_the_same_bytes_in_another_namespace(tmp_path):
    first = ProductStore(tmp_path / "first").save(_product())
    second = ProductStore(tmp_path / "second").save(_product())

    assert _directory_bytes(first) == _directory_bytes(second)


def test_an_existing_product_is_never_replaced_without_overwrite(tmp_path):
    store = ProductStore(tmp_path)
    store.save(_product())
    before = _directory_bytes(store.stage_dir("sky"))

    with pytest.raises(ProductExistsError, match="outputs.overwrite"):
        store.save(_product(identity=OTHER_IDENTITY))

    assert _directory_bytes(store.stage_dir("sky")) == before


def test_overwrite_replaces_the_whole_product(tmp_path):
    store = ProductStore(tmp_path)
    store.save(_product())
    replacement = _product(
        identity=OTHER_IDENTITY,
        record={"background": 0.5},
        arrays={"profile": np.ones(4)},
        images={},
    )

    store.save(replacement, overwrite=True)
    loaded = store.load("sky", identity=OTHER_IDENTITY)

    assert dict(loaded.record) == {"background": 0.5}
    assert set(loaded.arrays) == {"profile"}
    assert dict(loaded.images) == {}
    assert sorted(path.name for path in store.stage_dir("sky").iterdir()) == sorted(
        ["profile.npy", RECORD_NAME, MANIFEST_NAME, COMPLETE_NAME]
    )
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_status_reports_missing_incomplete_stale_and_complete(tmp_path):
    store = ProductStore(tmp_path)

    assert store.status("sky", IDENTITY) is ProductStatus.MISSING

    store.save(_product())

    assert store.status("sky", IDENTITY) is ProductStatus.COMPLETE
    assert store.status("sky", OTHER_IDENTITY) is ProductStatus.STALE

    (store.stage_dir("sky") / COMPLETE_NAME).unlink()

    assert store.status("sky", IDENTITY) is ProductStatus.INCOMPLETE


def test_a_stale_product_is_refused_with_both_identities(tmp_path):
    store = ProductStore(tmp_path)
    store.save(_product())

    with pytest.raises(StaleProductError) as error:
        store.load("sky", identity=OTHER_IDENTITY)

    assert IDENTITY[:12] in str(error.value)
    assert OTHER_IDENTITY[:12] in str(error.value)


def test_an_unfinished_product_is_refused_and_kept(tmp_path):
    store = ProductStore(tmp_path)
    store.save(_product())
    (store.stage_dir("sky") / COMPLETE_NAME).unlink()

    with pytest.raises(IncompleteProductError, match="diagnosis"):
        store.load("sky")

    with pytest.raises(ProductExistsError, match="unfinished"):
        store.save(_product())

    assert (store.stage_dir("sky") / RECORD_NAME).is_file()


def test_a_missing_product_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        ProductStore(tmp_path).load("sky")


@pytest.mark.parametrize("name", ["profile.npy", "prepared_image.fits", RECORD_NAME])
def test_one_changed_byte_is_detected(tmp_path, name):
    store = ProductStore(tmp_path)
    store.save(_product())
    path = store.stage_dir("sky") / name
    payload = bytearray(path.read_bytes())
    payload[-2] ^= 0x01
    path.write_bytes(bytes(payload))

    with pytest.raises(CorruptProductError, match="integrity"):
        store.load("sky")


def test_a_changed_manifest_is_detected(tmp_path):
    store = ProductStore(tmp_path)
    store.save(_product())
    path = store.stage_dir("sky") / MANIFEST_NAME
    path.write_text(path.read_text().replace('"sky"', '"sky" '))

    with pytest.raises(CorruptProductError, match="completion marker"):
        store.load("sky")


def test_an_unexpected_or_missing_file_is_detected(tmp_path):
    store = ProductStore(tmp_path)
    store.save(_product())
    directory = store.stage_dir("sky")
    (directory / "extra.npy").write_bytes(b"x")

    with pytest.raises(CorruptProductError, match="unexpected"):
        store.load("sky")

    (directory / "extra.npy").unlink()
    (directory / "profile.npy").unlink()

    with pytest.raises(CorruptProductError, match="missing"):
        store.load("sky")


def test_a_marker_of_another_stage_is_refused(tmp_path):
    store = ProductStore(tmp_path)
    store.save(_product())
    marker_path = store.stage_dir("sky") / COMPLETE_NAME
    marker = json.loads(marker_path.read_text())
    marker["stage"] = "cutout"
    marker_path.write_text(json.dumps(marker))

    with pytest.raises(CorruptProductError):
        store.load("sky")


def test_a_failed_save_publishes_nothing_and_leaves_no_staging(tmp_path, monkeypatch):
    store = ProductStore(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("baymorph.preparation.store._write_bytes", fail)

    with pytest.raises(OSError, match="disk full"):
        store.save(_product())

    assert not store.stage_dir("sky").exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"stage": "Sky"}, ValueError),
        ({"version": 0}, ValueError),
        ({"identity": "not-a-digest"}, ValueError),
        ({"record": {"value": float("nan")}}, ValueError),
        ({"record": {"path": Path("data/image.fits")}}, TypeError),
        ({"arrays": {"record": np.zeros(2)}}, ValueError),
        ({"arrays": {"bad-name": np.zeros(2)}}, ValueError),
        ({"arrays": {"objects": np.array([1, "a"], dtype=object)}}, TypeError),
        ({"arrays": {"words": np.array(["a", "b"])}}, TypeError),
        ({"images": {"mask": np.zeros((2, 2), dtype=bool)}}, TypeError),
        ({"images": {"profile": np.zeros((2, 2))}}, ValueError),
    ],
)
def test_invalid_products_are_refused_before_anything_is_written(
    tmp_path, overrides, error
):
    with pytest.raises(error):
        _product(**overrides)

    assert list(tmp_path.iterdir()) == []


def test_the_product_holds_copies_of_the_arrays():
    values = np.zeros(3)
    product = _product(arrays={"profile": values}, images={})
    values[0] = 1.0

    assert product.arrays["profile"][0] == 0.0
