"""The curated mask of the preparation, ``data.mask_path``.

The cases are those that the reader of 0.6 had to meet: a mask of known
pixels, a box and a bad column, written as 0/1 bytes, floats, integers with
-1 flags, in an extension behind an empty primary HDU or behind a table, and
compressed; masks that are not finite, three-dimensional, of another shape or
missing; a relative path, which is relative to the project and not to the
current directory; and a mask that marks the good pixels, which is warned
about.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import baymorph.preparation.opening as opening
from baymorph.preparation.masks import (
    CURATED_STAGE,
    CuratedMask,
    curated_from_settings,
    read_curated_mask,
)
from baymorph.preparation.opening import open_image
from baymorph.preparation.schema import read_preparation_settings

SHAPE = (120, 100)


@pytest.fixture(autouse=True)
def _fresh_headers():
    """Start and end every test without headers kept for reuse."""
    opening.forget_kept_headers()
    yield
    opening.forget_kept_headers()


def _truth() -> np.ndarray:
    """Return the curated pixels: a box over a star and a bad column."""
    mask = np.zeros(SHAPE, dtype=bool)
    mask[30:45, 70:85] = True
    mask[:, 12:14] = True

    return mask


REGION = {
    "region": {"size": 70.0},
    "target_position": {"x": 50.0, "y": 65.0},
    "pixel_scale": 1.0,
}
"""A working region of 70 pixels about (50, 65), at 1 arcsec per pixel."""


def _opened(tmp_path: Path, **settings):
    """Write an image of noise, unless it exists, and open it."""
    path = tmp_path / "image.fits"

    if not path.exists():
        image = np.random.default_rng(1).normal(0.0, 1.0, SHAPE).astype(np.float32)
        fits.PrimaryHDU(image).writeto(path)

    return open_image(path, project_root=tmp_path, **settings)


def _window(opened):
    """Return the slices of the working region in the whole image."""
    region = opened.region

    return slice(region.ymin, region.ymax), slice(region.xmin, region.xmax)


def _write(path: Path, values, *, layout: str = "primary") -> Path:
    """Write a mask as the primary HDU, behind an empty primary, or a table."""
    image = np.asarray(values)

    if layout == "primary":
        fits.PrimaryHDU(image).writeto(path)
    elif layout == "extension":
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(image, name="MASK")]).writeto(
            path
        )
    else:
        table = fits.BinTableHDU.from_columns(
            [fits.Column(name="a", format="E", array=np.zeros(3))], name="TAB"
        )
        fits.HDUList(
            [fits.PrimaryHDU(), table, fits.ImageHDU(image, name="MASK")]
        ).writeto(path)

    return path


@pytest.mark.parametrize(
    "values",
    [
        _truth().astype(np.uint8),
        _truth().astype(np.float32),
        np.where(_truth(), -1, 0).astype(np.int16),
    ],
    ids=["uint8", "float32", "int16_minus_one"],
)
def test_non_zero_pixels_are_excluded(tmp_path, values) -> None:
    opened = _opened(tmp_path)
    path = _write(tmp_path / "mask.fits", values)

    curated = read_curated_mask(path, opened, project_root=tmp_path)

    assert isinstance(curated, CuratedMask)
    assert np.array_equal(curated.pixels, _truth())
    assert curated.pixels.dtype == bool
    assert not curated.pixels.flags.writeable
    assert curated.excluded_pixels == int(_truth().sum())
    assert curated.hdu_index == 0
    assert curated.warnings == ()


@pytest.mark.parametrize("layout", ["extension", "behind_a_table"])
def test_the_hdu_is_chosen_as_for_the_image(tmp_path, layout) -> None:
    opened = _opened(tmp_path)
    path = _write(tmp_path / "mask.fits", _truth().astype(np.uint8), layout=layout)

    curated = read_curated_mask(path, opened, project_root=tmp_path)

    assert np.array_equal(curated.pixels, _truth())
    assert curated.hdu_index == (1 if layout == "extension" else 2)
    assert curated.record["hdu"]["name"] == "MASK"


def test_a_compressed_mask_is_read_on_the_working_region(tmp_path) -> None:
    opened = _opened(tmp_path, **REGION)
    plain = _write(tmp_path / "mask.fits", _truth().astype(np.uint8))
    compressed = tmp_path / "mask.fits.gz"

    with plain.open("rb") as source, gzip.open(compressed, "wb") as target:
        shutil.copyfileobj(source, target)

    curated = read_curated_mask(compressed, opened, project_root=tmp_path)

    assert opened.region is not None
    assert curated.pixels.shape == opened.pixels.shape
    assert np.array_equal(curated.pixels, _truth()[_window(opened)])
    assert np.any(curated.pixels) and not np.all(curated.pixels)
    assert curated.record["region"] == opened.region.as_dict()


@pytest.mark.parametrize(
    "values, message",
    [
        (np.where(_truth(), np.nan, 0.0).astype(np.float32), "not finite"),
        (_truth().astype(np.uint8)[None, :, :], "no two-dimensional image"),
        (_truth().astype(np.uint8)[:100, :90], "shape of the whole image"),
    ],
    ids=["nan", "three_dimensional", "other_shape"],
)
def test_a_mask_that_cannot_be_used_is_refused(tmp_path, values, message) -> None:
    opened = _opened(tmp_path)
    path = _write(tmp_path / "mask.fits", values)

    with pytest.raises(ValueError, match=message):
        read_curated_mask(path, opened, project_root=tmp_path)


def test_a_missing_mask_names_the_project_folder(tmp_path) -> None:
    opened = _opened(tmp_path)

    with pytest.raises(FileNotFoundError, match="relative to the project folder"):
        read_curated_mask("data/none.fits", opened, project_root=tmp_path)


def test_a_relative_path_is_relative_to_the_project(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    elsewhere = tmp_path / "elsewhere"
    (project / "data").mkdir(parents=True)
    (elsewhere / "data").mkdir(parents=True)
    opened = _opened(project)
    _write(project / "data" / "mask.fits", _truth().astype(np.uint8))
    decoy = np.zeros(SHAPE, dtype=np.uint8)
    _write(elsewhere / "data" / "mask.fits", decoy)
    monkeypatch.chdir(elsewhere)

    curated = read_curated_mask("data/mask.fits", opened, project_root=project)

    assert np.array_equal(curated.pixels, _truth())
    assert curated.record["file"]["path"] == "data/mask.fits"


def test_a_mask_that_excludes_most_of_the_region_is_warned_about(tmp_path) -> None:
    opened = _opened(tmp_path)
    path = _write(tmp_path / "mask.fits", (~_truth()).astype(np.uint8))

    curated = read_curated_mask(path, opened, project_root=tmp_path)

    assert len(curated.warnings) == 1
    assert "must be inverted" in curated.warnings[0]
    assert curated.record["warnings"] == list(curated.warnings)


def test_the_identity_follows_the_content_and_the_region(tmp_path) -> None:
    opened = _opened(tmp_path)
    first = _write(tmp_path / "first.fits", _truth().astype(np.uint8))
    copy = tmp_path / "copy.fits"
    shutil.copyfile(first, copy)
    other = _truth().copy()
    other[0, 0] = True
    changed = _write(tmp_path / "changed.fits", other.astype(np.uint8))

    identity = read_curated_mask(first, opened, project_root=tmp_path).identity
    region = _opened(tmp_path, **REGION)

    assert read_curated_mask(copy, opened, project_root=tmp_path).identity == identity
    assert (
        read_curated_mask(changed, opened, project_root=tmp_path).identity != identity
    )
    assert read_curated_mask(first, region, project_root=tmp_path).identity != identity


def test_the_record_is_json_serializable(tmp_path) -> None:
    opened = _opened(tmp_path)
    path = _write(tmp_path / "mask.fits", _truth().astype(np.uint8))

    record = json.loads(
        json.dumps(dict(read_curated_mask(path, opened, project_root=tmp_path).record))
    )

    assert record["stage"] == CURATED_STAGE
    assert record["excluded_pixels"] == int(_truth().sum())
    assert record["rule"] == "non-zero pixels are excluded"


def test_a_configuration_without_a_mask_reads_none(tmp_path) -> None:
    opened = _opened(tmp_path)
    settings = read_preparation_settings(
        {
            "object": {"name": "galaxy"},
            "data": {"image_path": "image.fits", "target_position": "center"},
            "outputs": {"root": "runs"},
        }
    )

    assert curated_from_settings(settings, opened, project_root=tmp_path) is None


def test_a_configuration_with_a_mask_reads_it(tmp_path) -> None:
    opened = _opened(tmp_path)
    _write(tmp_path / "mask.fits", _truth().astype(np.uint8))
    settings = read_preparation_settings(
        {
            "object": {"name": "galaxy"},
            "data": {
                "image_path": "image.fits",
                "target_position": "center",
                "mask_path": "mask.fits",
            },
            "outputs": {"root": "runs"},
        }
    )

    curated = curated_from_settings(settings, opened, project_root=tmp_path)

    assert np.array_equal(curated.pixels, _truth())


def test_the_reader_takes_the_product_of_the_opening(tmp_path) -> None:
    path = _write(tmp_path / "mask.fits", _truth().astype(np.uint8))

    with pytest.raises(TypeError, match="OpenedObservation"):
        read_curated_mask(path, object(), project_root=tmp_path)
