"""The target stage of the preparation (AD-055).

The target stage checks the target position against the image: inside the
image and the working region, on a pixel with data. The selection of the
target segment, which the first stage that detects sources calls, takes the
segment at the position, or the nearest one within 8 pixels with a warning,
stops beyond, and names a brighter source that the deblending split from the
target segment. These tests check each rule on maps whose answer they build,
against a search of every pixel, and on a deblended scene of a galaxy with a
star beside its nucleus.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from photutils.segmentation import deblend_sources, detect_sources
from scipy.ndimage import gaussian_filter

import baymorph.preparation.opening as opening
from baymorph.preparation.opening import open_image, seed_position
from baymorph.preparation.store import ProductStore
from baymorph.preparation.target import (
    MAX_SEED_DISTANCE,
    TARGET_STAGE,
    resolve_target,
    select_target_segment,
)


@pytest.fixture(autouse=True)
def _fresh_headers():
    """Start and end every test without headers kept for reuse."""
    opening.forget_kept_headers()
    yield
    opening.forget_kept_headers()


def _opened(tmp_path: Path, data=None, name: str = "image.fits", **settings):
    """Open an image of ``data``, 40 rows of 50 ones by default."""
    data = np.ones((40, 50), np.float32) if data is None else data
    path = tmp_path / name
    fits.PrimaryHDU(np.asarray(data)).writeto(path)

    return open_image(path, project_root=tmp_path, **settings)


def _region(tmp_path: Path, data, x: float, y: float):
    """Open a 300-pixel image within a region of 100 pixels about (x, y)."""
    return _opened(
        tmp_path,
        data,
        region={"size": 20.0},
        target_position={"x": x, "y": y},
        pixel_scale=0.2,
    )


# --- The target position ------------------------------------------------------


def test_a_declared_position_is_checked_and_kept(tmp_path) -> None:
    target = resolve_target(_opened(tmp_path), {"x": 10.2, "y": 20.7})

    assert target.position == (10.2, 20.7)
    assert (target.column, target.row) == (10, 21)
    assert target.source == "declared"
    assert target.record["declared"] == {"x": 10.2, "y": 20.7}
    assert target.record["image"] == {"width": 50, "height": 40}
    assert target.record["region"] is None


def test_center_is_the_centre_of_the_whole_image(tmp_path) -> None:
    target = resolve_target(_opened(tmp_path), "center")

    assert target.position == (24.5, 19.5)
    # A position halfway between two pixels goes to the even one.
    assert (target.column, target.row) == (24, 20)
    assert target.source == "image_centre"
    assert target.record["declared"] == "center"
    assert seed_position("center", (40, 50)) == (24.5, 19.5)
    assert seed_position(None, (40, 50)) is None


def test_center_with_a_working_region_is_the_centre_of_the_whole_image(
    tmp_path,
) -> None:
    opened = _opened(
        tmp_path,
        np.ones((300, 300), np.float32),
        region={"size": 20.0},
        target_position="center",
        pixel_scale=0.2,
    )

    target = resolve_target(opened, "center")

    assert target.position == (149.5, 149.5)
    assert target.record["region"] == {
        "xmin": 100,
        "xmax": 200,
        "ymin": 100,
        "ymax": 200,
    }


def test_a_missing_target_position_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"data\.target_position is required"):
        resolve_target(_opened(tmp_path), None)


@pytest.mark.parametrize(
    "value",
    [
        "centre",
        {"x": 1.0},
        {"x": float("nan"), "y": 1.0},
        {"x": True, "y": 1.0},
        [1.0, 2.0],
        {"ra": 195.1, "dec": 27.9},
    ],
)
def test_a_malformed_target_position_is_refused(tmp_path, value) -> None:
    with pytest.raises(ValueError, match=r"data\.target_position must be"):
        resolve_target(_opened(tmp_path), value)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (5000.0, 5000.0),
        (-50.0, 300.0),
        (-0.51, 10.0),
        (599.51, 10.0),
        (10.0, 600.0),
    ],
)
def test_a_target_outside_the_image_stops_at_the_target_stage(tmp_path, x, y) -> None:
    # The 0.6 preparation prepared the nearest star for (5000, 5000) on a
    # 600 x 600 image, with a fixed and with an automatic cutout.
    opened = _opened(tmp_path, np.ones((600, 600), np.float32))

    with pytest.raises(ValueError) as error:
        resolve_target(opened, {"x": x, "y": y})

    message = str(error.value)
    assert message.startswith("data.target_position")
    assert "lies outside the image of 600 x 600 pixels" in message
    assert "counts from one" in message


def test_the_edges_of_the_image_belong_to_its_edge_pixels(tmp_path) -> None:
    opened = _opened(tmp_path, np.ones((600, 600), np.float32))

    low = resolve_target(opened, {"x": -0.5, "y": -0.5})
    high = resolve_target(opened, {"x": 599.5, "y": 599.5})

    assert (low.column, low.row) == (0, 0)
    assert (high.column, high.row) == (599, 599)


def test_a_target_outside_the_working_region_is_refused(tmp_path) -> None:
    opened = _region(tmp_path, np.ones((300, 300), np.float32), 60.0, 60.0)

    assert opened.region.as_dict() == {"xmin": 10, "xmax": 110, "ymin": 10, "ymax": 110}

    edge = resolve_target(opened, {"x": 109.5, "y": 9.5})

    assert (edge.column, edge.row) == (109, 10)

    for x, y in ((110.0, 60.0), (200.0, 60.0), (60.0, 9.49)):
        with pytest.raises(ValueError, match="outside the working region x 10-110"):
            resolve_target(opened, {"x": x, "y": y})


def test_a_target_on_a_pixel_without_data_is_refused(tmp_path) -> None:
    data = np.ones((40, 50), np.float32)
    data[21, 10] = np.nan
    data[0:12, 30:50] = 0.0  # 240 exact zeros: an area without data
    opened = _opened(tmp_path, data)

    with pytest.raises(ValueError, match=r"pixel \(10, 21\), which holds no data"):
        resolve_target(opened, {"x": 10.2, "y": 20.7})

    with pytest.raises(ValueError, match="which holds no data"):
        resolve_target(opened, {"x": 40.0, "y": 5.0})

    assert resolve_target(opened, {"x": 10.2, "y": 20.4}).row == 20


def test_a_pixel_without_data_is_found_inside_the_working_region(tmp_path) -> None:
    data = np.ones((300, 300), np.float32)
    data[60, 70] = np.nan
    opened = _region(tmp_path, data, 70.0, 60.0)

    assert (opened.region.xmin, opened.region.ymin) == (20, 10)

    with pytest.raises(ValueError, match=r"pixel \(70, 60\), which holds no data"):
        resolve_target(opened, {"x": 70.0, "y": 60.0})

    assert resolve_target(opened, {"x": 71.0, "y": 60.0}).column == 71


def test_the_target_stage_takes_the_product_of_the_opening() -> None:
    with pytest.raises(TypeError, match="OpenedObservation"):
        resolve_target({"pixels": None}, "center")


def _project(root: Path, value: float = 1.0) -> Path:
    """Write a project with a 40 x 50 image in data/."""
    (root / "data").mkdir(parents=True)
    image = np.full((40, 50), value, np.float32)
    fits.PrimaryHDU(image).writeto(root / "data" / "image.fits")

    return root


def test_the_identity_follows_the_pixels_and_the_declared_position(tmp_path) -> None:
    root = _project(tmp_path / "project")
    plain = open_image("data/image.fits", project_root=root)
    declared = open_image(
        "data/image.fits", project_root=root, calibration={"gain": 2.5}
    )
    position = {"x": 10.0, "y": 20.0}
    first = resolve_target(plain, position)

    assert resolve_target(plain, dict(position)).identity == first.identity
    # The declared calibration changes the opening, not the pixels.
    assert resolve_target(declared, position).identity == first.identity
    assert resolve_target(plain, {"x": 11.0, "y": 20.0}).identity != first.identity
    assert (
        resolve_target(plain, "center").identity
        != resolve_target(plain, {"x": 24.5, "y": 19.5}).identity
    )

    moved = tmp_path / "elsewhere" / "project"
    shutil.copytree(root, moved)
    other = _project(tmp_path / "other", value=2.0)

    assert (
        resolve_target(open_image("data/image.fits", project_root=moved), position)
    ).identity == first.identity
    assert (
        resolve_target(open_image("data/image.fits", project_root=other), position)
    ).identity != first.identity


def test_the_record_is_json_and_the_product_round_trips(tmp_path) -> None:
    opened = _opened(tmp_path)
    target = resolve_target(opened, {"x": 10.0, "y": 20.0})

    record = json.loads(json.dumps(dict(target.record)))

    assert record["stage"] == TARGET_STAGE
    assert record["source"] == "declared"
    assert record["position"] == {"x": 10.0, "y": 20.0}
    assert record["pixel"] == {"x": 10, "y": 20}
    assert record["image_identity"] == opened.image_identity
    assert target.image_identity == opened.image_identity

    store = ProductStore(tmp_path / "products")
    store.save(target.product())
    loaded = store.load(TARGET_STAGE, identity=target.identity)

    assert json.loads(json.dumps(dict(loaded.record))) == record


# --- The target segment -------------------------------------------------------


def _map() -> np.ndarray:
    """Return 40 x 60 labels: a square, segment 1, and a bar, segment 2."""
    labels = np.zeros((40, 60), np.int32)
    labels[10:20, 10:20] = 1
    labels[25:30, 40:50] = 2

    return labels


def test_a_seed_on_a_segment_selects_it() -> None:
    selection = select_target_segment(_map(), (15.2, 14.8))

    assert selection.label == 1
    assert selection.on_source is True
    assert selection.distance == 0.0
    assert selection.nearest_pixel == (15, 15)
    assert selection.segment.pixels == 100
    assert selection.segment.centre == (14.5, 14.5)
    assert selection.segment.peak is None
    assert selection.max_distance == MAX_SEED_DISTANCE == 8.0
    assert selection.warnings == ()
    assert selection.parent is None
    assert selection.as_dict()["siblings"] is None


def test_a_seed_near_a_segment_selects_it_with_a_warning() -> None:
    selection = select_target_segment(_map(), (25.0, 15.0))

    assert selection.label == 1
    assert selection.on_source is False
    assert selection.distance == 6.0
    assert selection.nearest_pixel == (19, 15)

    [warning] = selection.warnings

    assert warning.startswith("No source was detected at the target position (25, 15)")
    assert "segment 1 (100 pixels, centred at (14.5, 14.5))" in warning
    assert "6.00 pixels away, within 8 pixels" in warning


def test_a_seed_far_from_every_segment_stops_naming_the_nearest() -> None:
    # The map starts at the pixel (100, 200) of the whole image.
    with pytest.raises(ValueError) as error:
        select_target_segment(_map(), (130.0, 215.0), origin=(100, 200))

    message = str(error.value)
    assert "No source was detected at the target position (130, 215)" in message
    assert "segment 1 (100 pixels, centred at (114.5, 214.5))" in message
    assert "nearest pixel (119, 215) lies 11.00 pixels away" in message
    assert "beyond the 8 pixels" in message
    assert "data.target_position" in message


@pytest.mark.parametrize(("x", "accepted"), [(27.0, True), (27.01, False)])
def test_the_largest_distance_is_inclusive(x, accepted) -> None:
    if accepted:
        assert select_target_segment(_map(), (x, 15.0)).distance == 8.0
    else:
        with pytest.raises(ValueError, match="8.01 pixels away, beyond the 8"):
            select_target_segment(_map(), (x, 15.0))


def test_a_largest_distance_of_zero_takes_only_a_seed_on_a_segment() -> None:
    assert select_target_segment(_map(), (19.0, 15.0), max_distance=0.0).label == 1

    with pytest.raises(ValueError, match="1.00 pixels away, beyond the 0 pixels"):
        select_target_segment(_map(), (20.0, 15.0), max_distance=0.0)


def test_segments_at_the_same_distance_are_named() -> None:
    labels = np.zeros((21, 21), np.int32)
    labels[10, 5] = 3
    labels[10, 15] = 4
    labels[5, 10] = 7

    selection = select_target_segment(labels, (10.0, 10.0))

    # All three lie 5 pixels away; the row of segment 7 comes first.
    assert selection.label == 7
    assert selection.distance == 5.0
    assert selection.equally_near == (3, 4)
    assert "Segments 3, 4 lie at the same distance" in selection.warnings[0]


def _brute_force(labels: np.ndarray, x: float, y: float) -> tuple[int, float]:
    """Return the label and distance of the nearest labelled pixel, by search."""
    rows, columns = np.nonzero(labels)
    squared = (columns - x) ** 2 + (rows - y) ** 2
    nearest = int(np.argmin(squared))

    return int(labels[rows[nearest], columns[nearest]]), math.sqrt(squared[nearest])


def test_the_nearest_segment_is_the_nearest_of_every_pixel() -> None:
    rng = np.random.default_rng(20261105)
    checked = 0

    for _ in range(200):
        height, width = (int(value) for value in rng.integers(4, 90, size=2))
        labels = np.zeros((height, width), np.int32)

        for label in range(1, int(rng.integers(2, 15))):
            row, column = (int(value) for value in rng.integers(0, (height, width)))
            rows, columns = (int(value) for value in rng.integers(1, 7, size=2))
            labels[row : row + rows, column : column + columns] = label

        for _ in range(12):
            x = float(rng.uniform(-0.5, width - 0.5))
            y = float(rng.uniform(-0.5, height - 0.5))

            if rng.random() < 0.3:
                # Halfway positions give pixels at equal distances.
                x = min(math.floor(x) + 0.5, width - 0.5)

            selection = select_target_segment(labels, (x, y), max_distance=1.0e6)
            pixel = labels[min(round(y), height - 1), min(round(x), width - 1)]

            if pixel:
                assert selection.label == pixel
                assert selection.on_source is True
                continue

            label, distance = _brute_force(labels, x, y)

            assert selection.label == label
            assert selection.distance == pytest.approx(distance, abs=1e-12)
            checked += 1

    assert checked > 1000


def test_the_nearest_segment_is_found_fast_on_a_large_map() -> None:
    # 16384 segments of 4 x 4 pixels on a 2048 x 2048 map, none within 600
    # pixels of the seed: the 0.6 search took minutes on such a map.
    cell = np.zeros((16, 16), np.int32)
    cell[:4, :4] = 1
    labels = np.kron(
        np.arange(1, 128 * 128 + 1, dtype=np.int32).reshape(128, 128), cell
    )
    rows, columns = np.ogrid[:2048, :2048]
    labels[(columns - 1000.3) ** 2 + (rows - 1000.7) ** 2 < 600.0**2] = 0
    seed = (1000.3, 1000.7)

    start = time.perf_counter()

    with pytest.raises(ValueError, match="beyond the 8 pixels"):
        select_target_segment(labels, seed)

    selection = select_target_segment(labels, seed, max_distance=1.0e4)
    elapsed = time.perf_counter() - start
    label, distance = _brute_force(labels, *seed)

    assert selection.label == label
    assert selection.distance == pytest.approx(distance, abs=1e-9)
    assert elapsed < 2.0


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (np.zeros((4, 4), float), "integer labels"),
        (np.zeros((4, 4), bool), "integer labels"),
        (np.full((4, 4), -1, np.int32), "negative labels"),
        (np.zeros((2, 4, 4), np.int32), "two-dimensional"),
        (np.zeros((0, 4), np.int32), "two-dimensional"),
    ],
)
def test_an_invalid_map_is_refused(labels, message) -> None:
    with pytest.raises(ValueError, match=message):
        select_target_segment(labels, (1.0, 1.0))


def test_a_map_without_sources_is_refused() -> None:
    with pytest.raises(ValueError, match=r"No source was detected in the map x 5-15"):
        select_target_segment(np.zeros((20, 10), np.int32), (8.0, 9.0), origin=(5, 7))


@pytest.mark.parametrize("seed", [(-0.6, 5.0), (60.0, 5.0), (5.0, 40.0)])
def test_a_seed_outside_the_map_is_refused(seed) -> None:
    with pytest.raises(ValueError, match="lies outside the map of detected sources"):
        select_target_segment(_map(), seed)


@pytest.mark.parametrize(
    "seed", [(np.nan, 1.0), (1.0, np.inf), (1.0,), "ab", (True, 1.0), None]
)
def test_an_invalid_seed_is_refused(seed) -> None:
    with pytest.raises(ValueError, match="two finite coordinates"):
        select_target_segment(_map(), seed)


@pytest.mark.parametrize("origin", [(-1, 0), (0.5, 0), (True, 0), (0,), "00", None])
def test_an_invalid_origin_is_refused(origin) -> None:
    with pytest.raises(ValueError, match="origin of a map"):
        select_target_segment(_map(), (15.0, 15.0), origin=origin)


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, True, "8"])
def test_an_invalid_largest_distance_is_refused(value) -> None:
    with pytest.raises(ValueError, match="max_distance"):
        select_target_segment(_map(), (15.0, 15.0), max_distance=value)


# --- A brighter source of the same detection ----------------------------------


def _family() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return two detections, the first split by the deblending in three.

    Segments 1, 2 and 3 come from detection 1 and hold fluxes 140, 368 and
    50; segment 4, alone in detection 2, holds 2500. The brightest pixel of
    segment 2 is (16, 15).
    """
    parents = np.zeros((30, 40), np.int32)
    parents[5:25, 5:25] = 1
    parents[5:10, 30:35] = 2
    labels = np.zeros_like(parents)
    labels[5:25, 5:12] = 1
    labels[5:25, 12:20] = 2
    labels[5:25, 20:25] = 3
    labels[5:10, 30:35] = 4
    flux = np.zeros((30, 40))
    flux[labels == 1] = 1.0
    flux[labels == 2] = 2.0
    flux[labels == 3] = 0.5
    flux[labels == 4] = 100.0
    flux[15, 16] = 50.0

    return labels, parents, flux


def test_a_brighter_source_of_the_same_detection_is_named() -> None:
    labels, parents, flux = _family()

    selection = select_target_segment(labels, (8.0, 15.0), parents=parents, flux=flux)

    assert selection.label == 1
    assert selection.parent == 1
    assert selection.siblings == 3
    assert selection.segment.flux == pytest.approx(140.0)
    assert selection.brighter.label == 2
    assert selection.brighter.flux == pytest.approx(368.0)
    assert selection.brighter.peak == (16, 15)
    assert selection.brighter_count == 1

    [warning] = selection.warnings

    assert "split the target segment 1 from a brighter source" in warning
    assert "segment 2 (160 pixels, brightest at (16, 15)), with 2.63 times" in warning

    record = json.loads(json.dumps(selection.as_dict()))

    assert record["siblings"]["brighter"]["label"] == 2
    assert record["siblings"]["brighter"]["peak"] == {"x": 16, "y": 15}
    assert record["segment"]["flux"] == pytest.approx(140.0)


def test_the_brightest_of_several_brighter_sources_is_named() -> None:
    labels, parents, flux = _family()

    selection = select_target_segment(labels, (22.0, 15.0), parents=parents, flux=flux)

    assert selection.label == 3
    assert selection.brighter.label == 2
    assert selection.brighter_count == 2
    assert "2 segments of that detection are brighter" in selection.warnings[0]


def test_the_brightest_segment_and_a_lone_source_raise_no_warning() -> None:
    labels, parents, flux = _family()

    brightest = select_target_segment(labels, (15.0, 15.0), parents=parents, flux=flux)
    alone = select_target_segment(labels, (32.0, 7.0), parents=parents, flux=flux)

    assert brightest.label == 2
    assert brightest.brighter is None
    assert brightest.warnings == ()
    # Segment 4 is brighter than every other, but it is another detection.
    assert alone.label == 4
    assert alone.siblings == 1
    assert alone.warnings == ()


def test_a_seed_near_a_split_segment_gets_both_warnings() -> None:
    labels, parents, flux = _family()

    selection = select_target_segment(labels, (3.0, 15.0), parents=parents, flux=flux)

    assert selection.label == 1
    assert selection.distance == 2.0
    assert len(selection.warnings) == 2
    assert selection.warnings[0].startswith("No source was detected")
    assert "brightest at (5, 5)" in selection.warnings[0]
    assert "split the target segment 1" in selection.warnings[1]


def test_a_target_segment_without_positive_flux_is_compared_by_its_sum() -> None:
    labels, parents, flux = _family()
    flux[labels == 1] = -1.0

    selection = select_target_segment(labels, (8.0, 15.0), parents=parents, flux=flux)

    assert "a flux of 368, against -140 for the target segment" in (
        selection.warnings[0]
    )


def test_the_check_of_the_siblings_needs_consistent_maps() -> None:
    labels, parents, flux = _family()
    seed = (8.0, 15.0)

    with pytest.raises(ValueError, match="given together"):
        select_target_segment(labels, seed, parents=parents)

    with pytest.raises(ValueError, match="given together"):
        select_target_segment(labels, seed, flux=flux)

    with pytest.raises(ValueError, match="must have the shape of the map of segments"):
        select_target_segment(labels, seed, parents=parents[:-1], flux=flux)

    with pytest.raises(ValueError, match="must have the shape of the map of segments"):
        select_target_segment(labels, seed, parents=parents, flux=flux[:, :-1])

    with pytest.raises(ValueError, match="numeric"):
        select_target_segment(labels, seed, parents=parents, flux=flux > 0)

    broken = parents.copy()
    broken[5:25, 5:8] = 0

    with pytest.raises(ValueError, match="inside one of its sources"):
        select_target_segment(labels, (10.0, 15.0), parents=broken, flux=flux)

    on_target = flux.copy()
    on_target[6, 6] = np.nan

    with pytest.raises(ValueError, match="it is not on segment 1"):
        select_target_segment(labels, seed, parents=parents, flux=on_target)

    on_sibling = flux.copy()
    on_sibling[6, 22] = np.nan

    with pytest.raises(ValueError, match="segments of the detection 1"):
        select_target_segment(labels, seed, parents=parents, flux=on_sibling)

    off_sources = flux.copy()
    off_sources[0, 0] = np.nan

    assert (
        select_target_segment(
            labels, seed, parents=parents, flux=off_sources
        ).brighter.label
        == 2
    )


def _galaxy_and_star() -> np.ndarray:
    """Return an exponential disc with a star 25 pixels from its centre.

    The disc has R_e = 20 pixels, q = 0.7 and a centre of 640 at (100, 100);
    the star, at (125, 100), peaks at 20 times that, with a FWHM of 2.7
    pixels; the noise has sigma 3 and the sky is zero. The 0.6 preparation
    made the star the target of such a scene without a word.
    """
    y, x = np.indices((201, 201), dtype=float)
    angle = math.radians(20.0)
    along = (x - 100.0) * math.cos(angle) + (y - 100.0) * math.sin(angle)
    across = -(x - 100.0) * math.sin(angle) + (y - 100.0) * math.cos(angle)
    disc = 640.0 * np.exp(-np.hypot(along, across / 0.7) / (20.0 / 1.678))
    width = 2.7 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    star = 20.0 * 640.0 * np.exp(-((x - 125.0) ** 2 + (y - 100.0) ** 2) / width**2 / 2)
    noise = np.random.default_rng(21).normal(0.0, 3.0, x.shape)

    return disc + star + noise


def test_a_seed_on_a_deblended_star_names_the_galaxy() -> None:
    smooth = gaussian_filter(_galaxy_and_star(), 2.0)
    noise = 3.0 / (4.0 * math.sqrt(math.pi))  # sigma of the smoothed noise
    parents = detect_sources(smooth, 3.0 * noise, n_pixels=20)
    labels = deblend_sources(
        smooth, parents, n_pixels=20, n_levels=32, contrast=0.001, progress_bar=False
    )
    galaxy, star = int(labels.data[100, 100]), int(labels.data[100, 125])

    # One detection, which the deblending split into the galaxy and the star.
    assert parents.data[100, 100] == parents.data[100, 125] > 0
    assert 0 < star != galaxy > 0

    on_star = select_target_segment(
        labels.data, (125.0, 100.0), parents=parents.data, flux=smooth
    )
    on_galaxy = select_target_segment(
        labels.data, (100.0, 100.0), parents=parents.data, flux=smooth
    )

    assert on_star.label == star
    assert on_star.segment.peak == (125, 100)
    assert on_star.brighter.label == galaxy
    assert on_star.brighter.peak == (100, 100)
    assert "split the target segment" in on_star.warnings[0]
    assert on_galaxy.label == galaxy
    assert on_galaxy.warnings == ()
