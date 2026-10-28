"""Tests for the pixels without data of an image (AD-053)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import baymorph.pipeline.image_preparation as image_preparation
import baymorph.pipeline.opening as opening
import baymorph.pipeline.psf_construction as psf_construction
from baymorph import ObservationAdvisor
from baymorph.core.detection import detect_sources_segmentation
from baymorph.core.image import ImageData
from baymorph.core.mask import NO_DATA_MIN_ZERO_PIXELS, no_data_mask
from baymorph.pipeline import (
    load_image_preparation_products,
    pipeline_paths,
    prepare_real_image_from_config,
    resolve_config,
)
from baymorph.likelihood.vecchia_gaussian import (
    VecchiaCorrelatedGaussianImageLikelihood,
)
from baymorph.pipeline.config_maker import image_sky_gives_the_gain
from baymorph.pipeline.empirical_psf_builder import BuilderSettings, far_sky
from baymorph.pipeline.noise_calibration import build_correlated_noise_calibration
from baymorph.pipeline.psf_candidates import (
    load_psf_candidate_screen,
    run_psf_candidate_selection_from_config,
    save_psf_candidate_screen,
    save_psf_candidates,
    save_psf_candidates_provenance,
)


def _galaxies(shape=(160, 200), *, seed=3):
    """Return noise with two round sources, at (60, 80) and (140, 70)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(shape)
    image = rng.normal(0.0, 1.0, shape)

    for x0, y0, peak in ((60.0, 80.0, 60.0), (140.0, 70.0, 40.0)):
        image += peak * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 3.0**2))

    return image


# --- no_data_mask ---------------------------------------------------------


def test_non_finite_pixels_hold_no_data_in_any_image():
    image = np.ones((20, 20))
    image[3, 4] = np.nan
    image[5, 6] = np.inf
    image[7, 8] = -np.inf

    for floating_point in (True, False):
        mask = no_data_mask(image, floating_point=floating_point)

        assert mask.dtype == bool
        assert set(zip(*np.nonzero(mask))) == {(3, 4), (5, 6), (7, 8)}


def test_a_large_area_of_zeros_holds_no_data_only_in_floating_point():
    image = np.ones((40, 40))
    image[:10, :10] = 0.0

    assert np.array_equal(no_data_mask(image, floating_point=True), image == 0.0)
    assert not np.any(no_data_mask(image, floating_point=False))


def test_an_area_of_zeros_counts_from_one_hundred_connected_pixels():
    image = np.ones((40, 40))
    image[0:9, 0:11] = 0.0  # 99 pixels
    image[20:30, 20:30] = 0.0  # 100 pixels
    image[35, ::2] = 0.0  # 20 scattered zeros

    mask = no_data_mask(image, floating_point=True)

    assert NO_DATA_MIN_ZERO_PIXELS == 100
    assert not np.any(mask[0:9, 0:11])
    assert np.all(mask[20:30, 20:30])
    assert np.count_nonzero(mask) == 100


def test_zeros_touching_only_at_corners_are_separate_areas():
    image = np.ones((30, 30))
    image[np.arange(30), np.arange(30)] = 0.0
    image[np.arange(29), np.arange(1, 30)] = 0.0
    image[np.arange(1, 30), np.arange(29)] = 0.0

    mask = no_data_mask(image, floating_point=True, min_zero_pixels=30)

    assert np.all(mask[image == 0.0])

    image = np.ones((30, 30))
    image[np.arange(30), np.arange(30)] = 0.0

    assert not np.any(no_data_mask(image, floating_point=True, min_zero_pixels=2))


def test_negative_zero_is_a_zero():
    image = np.ones((20, 20))
    image[:10, :10] = -0.0

    assert np.count_nonzero(no_data_mask(image, floating_point=True)) == 100


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"data": np.ones(5)}, "two-dimensional numeric"),
        ({"data": np.array([["a", "b"]])}, "two-dimensional numeric"),
        ({"data": np.ones((5, 5)), "min_zero_pixels": 0}, "positive integer"),
        ({"data": np.ones((5, 5)), "min_zero_pixels": 1.5}, "positive integer"),
        ({"data": np.ones((5, 5)), "min_zero_pixels": True}, "positive integer"),
    ],
)
def test_no_data_mask_rejects_invalid_input(kwargs, message):
    with pytest.raises(ValueError, match=message):
        no_data_mask(floating_point=True, **kwargs)


# --- ImageData.floating_point ----------------------------------------------


@pytest.mark.parametrize(
    "dtype, expected", [(np.float32, True), (np.float64, True), (np.int16, False)]
)
def test_an_opened_image_says_how_its_pixels_were_stored(tmp_path, dtype, expected):
    path = tmp_path / "image.fits"
    fits.PrimaryHDU(np.ones((8, 8), dtype=dtype)).writeto(path)

    image = ImageData.from_fits(path)

    assert image.data.dtype == float
    assert image.floating_point is expected


def test_a_tile_compressed_float_image_is_floating_point(tmp_path):
    path = tmp_path / "image.fits.fz"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.CompImageHDU(np.ones((16, 16), dtype=np.float32), name="SCI"),
        ]
    ).writeto(path)

    assert ImageData.from_fits(path).floating_point


def test_an_image_in_memory_is_floating_point_by_its_array():
    assert ImageData(np.ones((4, 4))).floating_point
    assert not ImageData(np.ones((4, 4), dtype=np.int32)).floating_point


# --- detection with excluded pixels -----------------------------------------


def test_a_detection_without_excluded_pixels_is_the_detection_without_a_mask():
    image = _galaxies()

    plain = detect_sources_segmentation(image)

    for mask in (None, np.zeros(image.shape, dtype=bool)):
        masked = detect_sources_segmentation(image, mask=mask)

        assert masked.excluded is None
        assert np.array_equal(masked.detection_image, plain.detection_image)
        assert np.array_equal(masked.segmentation.data, plain.segmentation.data)
        assert masked.threshold_level == plain.threshold_level


def test_pixels_without_data_take_no_part_in_the_detection():
    image = _galaxies()
    reference = detect_sources_segmentation(image)

    empty = np.zeros(image.shape, dtype=bool)
    empty[:, 170:] = True
    empty[:30, :40] = True
    image[:, 170:] = np.nan
    image[:30, :40] = 0.0

    detection = detect_sources_segmentation(image, mask=empty)

    assert np.array_equal(detection.excluded, empty)
    assert np.all(np.isfinite(detection.detection_image))
    assert np.all(detection.detection_image[empty] == detection.background_level)
    assert not np.any(detection.mask[empty])
    assert abs(detection.threshold_level - reference.threshold_level) < 0.05

    # The threshold is measured on fewer pixels, so a segment may differ by a
    # pixel or two at its border.
    for x0, y0 in ((60, 80), (140, 70)):
        found = detection.segmentation.data
        found = found == found[y0, x0]
        expected = reference.segmentation.data
        expected = expected == expected[y0, x0]
        assert np.count_nonzero(found & expected) > 0.98 * np.count_nonzero(
            found | expected
        )

    # The zeros and the NaN do not lower the smoothed image next to them.
    edge = detection.detection_image[:, 160:170]
    assert abs(np.mean(edge) - detection.background_level) < 0.1

    table = detection.catalog.to_table()
    assert np.all(np.isfinite(table["segment_flux"]))


def test_unmasked_non_finite_pixels_stop_the_detection():
    image = _galaxies()
    image[10, 10] = np.nan

    with pytest.raises(ValueError, match="1 non-finite pixels"):
        detect_sources_segmentation(image)

    mask = np.zeros(image.shape, dtype=bool)
    mask[0, 0] = True

    with pytest.raises(ValueError, match="does not exclude"):
        detect_sources_segmentation(image, mask=mask)


def test_a_detection_mask_must_match_and_leave_pixels():
    image = _galaxies()

    with pytest.raises(ValueError, match="does not have the shape"):
        detect_sources_segmentation(image, mask=np.zeros((3, 3), dtype=bool))

    with pytest.raises(ValueError, match="excludes every pixel"):
        detect_sources_segmentation(image, mask=np.ones(image.shape, dtype=bool))


# --- preparation ------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_openings():
    """Start and end every test without observations kept for reuse."""
    opening.forget_opened_observations()
    yield
    opening.forget_opened_observations()


def _galaxy(blanks=()):
    """Return a galaxy at (170, 90) and a neighbour on a noisy sky, 200 x 260.

    Each blank is ``(value, rows, columns)``, written over the image.
    """
    rng = np.random.default_rng(20261020)
    yy, xx = np.mgrid[0:200, 0:260]
    image = 100.0 + rng.normal(0.0, 2.0, size=(200, 260))
    image += 400.0 * np.exp(-0.5 * ((yy - 90.0) ** 2 + (xx - 170.0) ** 2) / 6.0**2)
    image += 300.0 * np.exp(-0.5 * ((yy - 52.0) ** 2 + (xx - 208.0) ** 2) / 1.5**2)

    for value, rows, columns in blanks:
        image[rows, columns] = value

    return image


OUTSIDE = (
    (np.nan, slice(0, 200), slice(0, 40)),
    (0.0, slice(170, 200), slice(40, 120)),
)
"""Blanks outside the fixed cutout x 120-220, y 40-140 of the galaxy."""

INSIDE = (np.nan, slice(45, 55), slice(125, 135))
"""A blank of 100 pixels inside the cutout, away from the galaxy."""


def _config(tmp_path: Path, name: str, image, *, mode="fixed", mask=None) -> dict:
    """Write ``image`` and return the configuration of a cutout of the galaxy."""
    image_path = tmp_path / f"{name}.fits"
    fits.writeto(image_path, np.asarray(image, dtype=np.float32))
    cutout = (
        {"mode": "fixed", "size": 100}
        if mode == "fixed"
        else {"mode": "auto", "min_size": 60, "max_size": 150}
    )
    data = {
        "image_path": str(image_path),
        "pixel_scale": 0.1,
        "target_position": {"x": 170, "y": 90},
        "cutout": cutout,
    }

    if mask is not None:
        mask_path = tmp_path / f"{name}_mask.fits"
        fits.writeto(mask_path, mask.astype(np.uint8))
        data["mask_path"] = str(mask_path)

    return resolve_config(
        {
            "object": {"name": "galaxy"},
            "data": data,
            "outputs": {"root": str(tmp_path / name), "overwrite": False},
        }
    )


PRODUCTS = (
    "cutout",
    "prepared_data",
    "noise",
    "fitting_mask",
    "target_segment",
    "contaminant_mask",
    "segmentation",
)


def test_pixels_without_data_outside_a_fixed_cutout_change_nothing(tmp_path):
    clean = prepare_real_image_from_config(_config(tmp_path, "clean", _galaxy()))
    config = _config(tmp_path, "blank", _galaxy(OUTSIDE))
    blank = prepare_real_image_from_config(config)

    for name in PRODUCTS:
        assert np.array_equal(getattr(blank, name), getattr(clean, name)), name

    assert blank.background == clean.background
    assert np.isnan(blank.original_image[0, 0])

    loaded = load_image_preparation_products(config)

    assert np.array_equal(loaded.original_image, blank.original_image, equal_nan=True)


def test_pixels_without_data_outside_an_automatic_cutout_change_nothing(tmp_path):
    # The region read holds the blanks, but the automatic cutout does not.
    clean = prepare_real_image_from_config(
        _config(tmp_path, "clean", _galaxy(), mode="auto"), save=False
    )
    blank = prepare_real_image_from_config(
        _config(tmp_path, "blank", _galaxy(OUTSIDE), mode="auto"), save=False
    )

    assert blank.cutout_bounds == clean.cutout_bounds
    assert blank.cutout_bounds["xmin"] >= 40

    for name in PRODUCTS:
        assert np.array_equal(getattr(blank, name), getattr(clean, name)), name


@pytest.mark.parametrize("value", [np.nan, np.inf, 0.0])
def test_pixels_without_data_inside_the_cutout_stop_the_preparation(tmp_path, value):
    _, rows, columns = INSIDE
    config = _config(tmp_path, "inside", _galaxy([(value, rows, columns)]))

    with pytest.raises(ValueError) as error:
        prepare_real_image_from_config(config, save=False)

    message = str(error.value)

    assert "The cutout, x 120-220 and y 40-140 of the image" in message
    assert "holds 100 pixels without data" in message
    assert "within x 125-135 and y 45-55" in message
    assert "data.cutout.size" in message
    assert "data.mask_path" in message


def test_zeros_inside_the_cutout_of_an_integer_image_are_data(tmp_path):
    # Counts on a sky of zero, where an area of zeros is a measurement.
    image = _galaxy() - 100.0
    image[INSIDE[1], INSIDE[2]] = 0.0
    image = np.round(image).astype(np.int32)
    image_path = tmp_path / "stored_as_integers.fits"
    fits.writeto(image_path, image)
    config = _config(tmp_path, "integer", _galaxy())
    config["data"]["image_path"] = str(image_path)

    prepared = prepare_real_image_from_config(config, save=False)

    assert np.all(prepared.cutout[5:15, 5:15] == 0.0)
    assert not np.any(prepared.fitting_mask[5:15, 5:15])


def test_a_curated_mask_lets_pixels_without_data_into_the_cutout(tmp_path):
    value, rows, columns = INSIDE
    image = _galaxy([INSIDE, (0.0, slice(120, 132), slice(190, 202))])
    curated = np.zeros(image.shape, dtype=bool)
    curated[44:56, 124:136] = True
    curated[120:132, 190:202] = True
    config = _config(tmp_path, "curated", image, mask=curated)

    prepared = prepare_real_image_from_config(config)
    nan_block = (slice(5, 15), slice(5, 15))
    zero_block = (slice(80, 92), slice(70, 82))

    assert np.all(np.isnan(prepared.prepared_data[nan_block]))
    assert np.all(
        np.isfinite(
            np.delete(
                prepared.prepared_data.ravel(),
                np.flatnonzero(np.isnan(prepared.prepared_data)),
            )
        )
    )
    assert np.count_nonzero(np.isnan(prepared.prepared_data)) == 100
    assert np.all(prepared.fitting_mask[nan_block])
    assert np.all(prepared.fitting_mask[zero_block])
    assert not np.any(prepared.sky_region.mask[nan_block])
    assert not np.any(prepared.sky_region.mask[zero_block])
    assert not np.any(prepared.segmentation[nan_block])
    assert not np.any(prepared.segmentation[zero_block])
    assert np.isfinite(prepared.background)
    assert abs(prepared.background - 100.0) < 1.0

    loaded = load_image_preparation_products(config)

    assert np.array_equal(loaded.prepared_data, prepared.prepared_data, equal_nan=True)
    assert np.array_equal(loaded.fitting_mask, prepared.fitting_mask)


def test_the_noise_and_the_likelihood_ignore_pixels_without_data_under_the_mask(
    tmp_path,
):
    curated = np.zeros((200, 260), dtype=bool)
    curated[44:56, 124:136] = True
    config = _config(tmp_path, "curated", _galaxy([INSIDE]), mask=curated)
    prepared = prepare_real_image_from_config(config)

    calibration = build_correlated_noise_calibration(config, save=False)
    likelihood = VecchiaCorrelatedGaussianImageLikelihood(
        data=prepared.prepared_data,
        noise=prepared.noise,
        correlation_kernel=calibration.correlation_kernel,
        mask=prepared.fitting_mask,
        max_neighbors=8,
        effective_gain=1.0,
    )
    yy, xx = np.indices(prepared.prepared_data.shape)
    model = 400.0 * np.exp(-0.5 * ((yy - 50.0) ** 2 + (xx - 50.0) ** 2) / 6.0**2)

    assert np.isfinite(likelihood.log_likelihood(model, source=model))


def test_a_saved_cutout_may_hold_non_finite_pixels_only_under_the_curated_mask(
    tmp_path,
):
    curated = np.zeros((200, 260), dtype=bool)
    curated[44:56, 124:136] = True
    config = _config(tmp_path, "curated", _galaxy([INSIDE]), mask=curated)
    prepared = prepare_real_image_from_config(config, save=False)

    broken = replace(prepared, external_mask=np.zeros_like(prepared.external_mask))

    with pytest.raises(ValueError, match="1?00 non-finite pixels"):
        image_preparation._validate_products_for_persistence(broken)


# --- stars of the PSF -------------------------------------------------------


def _star_field(blanks=()):
    """Return 49 isolated Gaussian stars of 1.5 pixels on a flat sky."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:300, 0:300]
    image = 50.0 + rng.normal(0.0, 0.5, size=(300, 300))

    for index in range(49):
        row, column = divmod(index, 7)
        x = 30.0 + 40.0 * column + rng.uniform(-0.5, 0.5)
        y = 30.0 + 40.0 * row + rng.uniform(-0.5, 0.5)
        flux = 2000.0 * (1.0 + 0.1 * index)
        image += (
            flux
            / (2.0 * np.pi * 1.5**2)
            * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / 1.5**2)
        )

    for value, rows, columns in blanks:
        image[rows, columns] = value

    return image


STAR_BLANKS = (
    # Touches the stars of the first column, at x about 30.
    (np.nan, slice(0, 300), slice(0, 26)),
    # Touches the star at about (150, 270), on the last row.
    (0.0, slice(274, 300), slice(120, 180)),
)


def _psf_config(tmp_path: Path, name: str, image) -> dict:
    """Write ``image`` and return a configuration whose builder fits it."""
    image_path = tmp_path / f"{name}.fits"
    fits.writeto(image_path, np.asarray(image, dtype=np.float32))

    return resolve_config(
        {
            "object": {"name": name},
            "data": {
                "image_path": str(image_path),
                "pixel_scale": 0.1,
                "target_position": {"x": 150.0, "y": 150.0},
                "cutout": {"mode": "fixed", "size": 50},
            },
            "psf": {
                "empirical": {
                    "candidate_selection": {
                        "target_n_candidates": 49,
                        "min_area": 5,
                        "max_area": 400,
                        "min_peak": 5,
                        "max_peak": 1.0e6,
                        "min_peak_snr": 5.0,
                        "edge_margin": 12,
                        "galaxy_exclusion_box": {
                            "xmin": 145.0,
                            "xmax": 155.0,
                            "ymin": 145.0,
                            "ymax": 155.0,
                        },
                        "min_half_size": 4,
                        "max_half_size": 8,
                    },
                    "construction": {
                        "sky_inner_radius": 14.0,
                        "sky_outer_radius": 22.0,
                        "mask_growth_pixels": 2,
                        "support_half_size": 10,
                    },
                }
            },
            "outputs": {"root": str(tmp_path / name), "overwrite": False},
        }
    )


def test_psf_candidates_touching_pixels_without_data_are_left_out(tmp_path):
    clean = run_psf_candidate_selection_from_config(
        _psf_config(tmp_path, "clean", _star_field())
    )
    blank = run_psf_candidate_selection_from_config(
        _psf_config(tmp_path, "blank", _star_field(STAR_BLANKS))
    )
    no_data = no_data_mask(_star_field(STAR_BLANKS), floating_point=True)

    def touches(candidate):
        half = candidate.half_size + 2
        x, y = int(round(candidate.x)), int(round(candidate.y))
        window = no_data[
            max(y - half, 0) : y + half + 1, max(x - half, 0) : x + half + 1
        ]
        return np.any(window)

    kept = sorted((c.x, c.y) for c in clean if not touches(c))

    # The threshold is measured on fewer pixels, so a centroid may move by a
    # few hundredths of a pixel.
    assert len(kept) == len(clean) - 9
    assert np.allclose(sorted((c.x, c.y) for c in blank), kept, rtol=0.0, atol=0.05)


def test_psf_stars_leave_out_pixels_without_data(tmp_path, monkeypatch):
    monkeypatch.setattr(
        psf_construction, "save_empirical_psf_diagnostics", lambda *a, **k: None
    )
    config = _psf_config(tmp_path, "blank", _star_field(STAR_BLANKS))
    run_psf_candidate_selection_from_config(config)

    psf = psf_construction.run_psf_construction_from_config(config)
    metadata = json.loads(Path(pipeline_paths(config)["psf"]["metadata"]).read_text())

    assert np.all(np.isfinite(psf))
    assert len(metadata["stars"]) >= 6
    assert all(abs(star["sky"] - 50.0) < 0.2 for star in metadata["stars"])


def test_a_star_whose_cutout_holds_pixels_without_data_is_rejected(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        psf_construction, "save_empirical_psf_diagnostics", lambda *a, **k: None
    )
    clean = _psf_config(tmp_path, "clean", _star_field())
    candidates = run_psf_candidate_selection_from_config(clean)
    first = candidates[0]
    x, y = int(round(first.x)), int(round(first.y))
    blank = _star_field([(np.nan, slice(y + 2, y + 3), slice(x + 2, x + 3))])
    config = _psf_config(tmp_path, "blank", blank)
    # The candidates of the clean field, with the record of their screen, as
    # if found before the blank appeared.
    save_psf_candidates(config, candidates)
    save_psf_candidate_screen(config, load_psf_candidate_screen(clean))
    save_psf_candidates_provenance(config)

    psf_construction.run_psf_construction_from_config(config)
    metadata = json.loads(Path(pipeline_paths(config)["psf"]["metadata"]).read_text())
    rejected = metadata["rejected_candidates"]

    assert [(c["x"], c["y"], c["reason"]) for c in rejected] == [
        (first.x, first.y, "pixels_without_data")
    ]


def test_the_sky_ring_of_a_star_leaves_out_pixels_without_data():
    # Zeros fill most of the sky ring, so the clipping does not reject them.
    image = _star_field()
    image[:, 0:36] = 0.0
    labels = np.zeros(image.shape, dtype=int)
    settings = BuilderSettings(
        sky_inner_radius=14.0, sky_outer_radius=22.0, support_half_size=10
    )
    no_data = no_data_mask(image, floating_point=True)

    biased, _ = far_sky(image, labels, 30.0, 150.0, settings)
    sky, _ = far_sky(image, labels, 30.0, 150.0, settings, no_data)

    assert biased < 1.0
    assert abs(sky - 50.0) < 0.2


# --- whole-image diagnostics ------------------------------------------------


def test_the_advisor_inspects_an_image_with_pixels_without_data(tmp_path):
    image = _galaxy(OUTSIDE)
    image_path = tmp_path / "blank.fits"
    fits.writeto(image_path, image.astype(np.float32))
    clean_path = tmp_path / "clean.fits"
    fits.writeto(clean_path, _galaxy().astype(np.float32))

    inspection = ObservationAdvisor(
        image_path, target_position={"x": 170.0, "y": 90.0}, online=False
    ).inspect()
    clean = ObservationAdvisor(
        clean_path, target_position={"x": 170.0, "y": 90.0}, online=False
    ).inspect()
    no_data = no_data_mask(image, floating_point=True)

    assert inspection.no_data_fraction == pytest.approx(np.mean(no_data))
    assert inspection.minimum > 50.0
    assert abs(inspection.median - clean.median) < 0.5
    assert abs(inspection.background - clean.background) < 0.5
    assert abs(inspection.noise_sigma - clean.noise_sigma) < 0.1
    assert inspection.target_area_pixels == pytest.approx(
        clean.target_area_pixels, rel=0.05
    )


def test_areas_of_zeros_do_not_decide_the_gain_check():
    rng = np.random.default_rng(5)
    sky = 100.0 + rng.normal(0.0, 2.0, size=(100, 100))
    blank = sky.copy()
    blank[:, :60] = 0.0

    assert image_sky_gives_the_gain(sky)
    assert image_sky_gives_the_gain(blank)
    assert not image_sky_gives_the_gain(blank, floating_point=False)
