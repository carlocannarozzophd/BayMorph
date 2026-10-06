"""The contaminant rule: which detected segments are masked.

The production detection compares a smoothed image with one threshold above
the background of the whole image. Over the wings of the target, the light
of the target lifts peaks of the noise above that threshold, and masking
them as contaminants removes positive noise from the likelihood. The rule
``local_excess`` masks a segment only when it still holds a source with the
threshold raised by the smooth light of the target.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

import baymorph.pipeline.image_preparation as image_preparation_module
from baymorph.core.detection import (
    DetectionResult,
    detect_sources_segmentation,
    segments_above_level,
)
from baymorph.pipeline import (
    PreparedObservation,
    load_image_preparation_products,
    pipeline_paths,
    resolve_config,
    save_image_preparation_products,
)
from baymorph.pipeline.config import validate_config
from baymorph.pipeline.prepared_identity import _preparation_config_identity


def _square(shape, x, y, half):
    """Return a boolean square of side ``2 half + 1`` centred on (x, y)."""
    mask = np.zeros(shape, dtype=bool)
    mask[y - half:y + half + 1, x - half:x + half + 1] = True
    return mask


# ---------------------------------------------------------------------------
# segments_above_level
# ---------------------------------------------------------------------------


def test_a_segment_holds_a_source_when_enough_connected_pixels_clear_the_level():
    shape = (40, 40)
    image = np.zeros(shape)
    segmentation = np.zeros(shape, dtype=int)
    source = _square(shape, 10, 10, 3)
    segmentation[source] = 1
    image[source] = 2.0

    holds = segments_above_level(
        image,
        segmentation,
        np.array([1]),
        level=np.zeros(shape),
        excess=1.0,
        n_pixels=20,
    )

    assert holds.tolist() == [True]

    raised = segments_above_level(
        image,
        segmentation,
        np.array([1]),
        level=np.full(shape, 1.5),
        excess=1.0,
        n_pixels=20,
    )

    assert raised.tolist() == [False]


def test_too_few_or_disconnected_pixels_do_not_hold_a_source():
    shape = (40, 40)
    image = np.zeros(shape)
    segmentation = np.zeros(shape, dtype=int)

    small = _square(shape, 10, 10, 1)
    segmentation[small] = 1
    image[small] = 5.0

    checker = _square(shape, 28, 28, 6)
    segmentation[checker] = 2
    yy, xx = np.indices(shape)
    scattered = checker & ((xx % 3 == 0) & (yy % 3 == 0))
    image[scattered] = 5.0

    holds = segments_above_level(
        image,
        segmentation,
        np.array([1, 2]),
        level=np.zeros(shape),
        excess=1.0,
        n_pixels=5,
    )

    assert np.count_nonzero(scattered) >= 5
    assert holds.tolist() == [True, False]


def test_pixels_outside_the_segment_do_not_count():
    shape = (30, 30)
    image = np.zeros(shape)
    segmentation = np.zeros(shape, dtype=int)
    segmentation[_square(shape, 10, 10, 1)] = 1
    image[_square(shape, 10, 10, 4)] = 5.0

    holds = segments_above_level(
        image,
        segmentation,
        np.array([1]),
        level=np.zeros(shape),
        excess=1.0,
        n_pixels=10,
    )

    assert holds.tolist() == [False]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"excess": 0.0}, "excess"),
        ({"excess": np.nan}, "excess"),
        ({"n_pixels": 0}, "n_pixels"),
        ({"labels": np.array([3])}, "not a segment"),
        ({"level": np.zeros((5, 5))}, "same shape"),
        ({"level": np.full((20, 20), np.inf)}, "finite"),
    ],
)
def test_invalid_inputs_are_rejected(change, message):
    shape = (20, 20)
    segmentation = np.zeros(shape, dtype=int)
    segmentation[_square(shape, 5, 5, 2)] = 1
    arguments = {
        "labels": np.array([1]),
        "level": np.zeros(shape),
        "excess": 1.0,
        "n_pixels": 5,
    }
    arguments.update(change)

    with pytest.raises(ValueError, match=message):
        segments_above_level(
            np.zeros(shape),
            segmentation,
            arguments["labels"],
            level=arguments["level"],
            excess=arguments["excess"],
            n_pixels=arguments["n_pixels"],
        )


def test_detection_records_its_image_background_and_threshold():
    rng = np.random.default_rng(0)
    image = rng.normal(size=(64, 64))
    image[30:36, 30:36] += 40.0

    detection = detect_sources_segmentation(image, n_sigma=3.0, n_pixels=5)

    assert detection.detection_image.shape == image.shape
    assert detection.threshold_level > detection.background_level
    smoothed_scatter = float(np.std(detection.detection_image[:20, :20]))
    excess = detection.threshold_level - detection.background_level
    assert excess == pytest.approx(3.0 * smoothed_scatter, rel=0.3)


# ---------------------------------------------------------------------------
# The rule in the preparation
# ---------------------------------------------------------------------------


def _galaxy_with_segments(shape=(121, 121)):
    """Return a target detection over a smooth galaxy with two segments.

    The detection image is a noiseless exponential profile. Segment 2 is a
    bump of 0.6 times the excess of the threshold over the wings, as a peak
    of the noise lifted by the galaxy; segment 3 is a neighbour of five times
    the excess. Both lie above the threshold of the whole image.
    """
    yy, xx = np.indices(shape, dtype=float)
    centre = (shape[0] - 1) / 2
    radius = np.hypot(xx - centre, yy - centre)
    galaxy = 20.0 * np.exp(-radius / 12.0)
    excess = 1.0
    detection_image = galaxy.copy()

    segmentation = np.zeros(shape, dtype=int)
    target = radius <= 18.0
    segmentation[target] = 1

    noise_peak = _square(shape, int(centre) + 24, int(centre), 3)
    neighbour = _square(shape, int(centre) - 40, int(centre) + 30, 3)
    segmentation[noise_peak] = 2
    segmentation[neighbour] = 3
    detection_image[noise_peak] += 0.6 * excess
    detection_image[neighbour] += 5.0 * excess

    assert np.all(detection_image[noise_peak] > excess)

    detection = DetectionResult(
        segmentation=SimpleNamespace(data=segmentation),
        mask=segmentation > 0,
        detection_image=detection_image,
        background_level=0.0,
        threshold_level=excess,
    )
    target_detection = image_preparation_module._TargetDetection(
        detection=detection,
        target_label=1,
        target_segment=segmentation == 1,
        contaminant_mask=(segmentation > 0) & (segmentation != 1),
    )

    return galaxy, target_detection, segmentation


def test_local_excess_releases_a_noise_peak_and_keeps_a_neighbour():
    image, target_detection, segmentation = _galaxy_with_segments()

    selected = image_preparation_module._apply_contaminant_rule(
        image,
        target_detection,
        rule="local_excess",
        detection_cfg={"n_pixels": 20},
    )

    assert selected.released_labels == (2,)
    assert np.array_equal(selected.contaminant_mask, segmentation == 3)
    assert np.array_equal(
        target_detection.contaminant_mask,
        (segmentation == 2) | (segmentation == 3),
    )


def test_all_segments_masks_every_segment():
    image, target_detection, _ = _galaxy_with_segments()

    selected = image_preparation_module._apply_contaminant_rule(
        image,
        target_detection,
        rule="all_segments",
        detection_cfg={},
    )

    assert selected is target_detection
    assert selected.released_labels == ()


def test_an_unknown_rule_is_rejected():
    image, target_detection, _ = _galaxy_with_segments()

    with pytest.raises(ValueError, match="Unknown contaminant rule"):
        image_preparation_module._apply_contaminant_rule(
            image,
            target_detection,
            rule="neighbours_only",
            detection_cfg={"n_pixels": 20},
        )


def test_the_level_is_the_smooth_light_of_the_target():
    image, target_detection, segmentation = _galaxy_with_segments()

    level = image_preparation_module._target_light_level(image, target_detection)
    unmasked = segmentation == 0

    # Annuli one pixel wide: the level differs from the profile by about the
    # change of the profile across one pixel, 0.37 at the edge of the target
    # segment, well below the excess of the threshold, 1.
    assert np.allclose(level[unmasked], image[unmasked], atol=0.3)
    assert np.allclose(level[segmentation == 2], image[segmentation == 2], atol=0.2)


def test_on_a_noisy_galaxy_the_rule_masks_fewer_segments_and_keeps_a_star():
    rng = np.random.default_rng(7)
    shape = (241, 241)
    yy, xx = np.indices(shape, dtype=float)
    centre = 120.0
    radius = np.hypot(xx - centre, (yy - centre) / 0.85)
    galaxy = 60.0 * np.exp(-7.67 * ((radius / 20.0) ** 0.25 - 1.0))
    star = 3000.0 * np.exp(-0.5 * ((xx - 200.0) ** 2 + (yy - 40.0) ** 2) / 1.5**2)
    image = galaxy + star + rng.normal(scale=3.0, size=shape)

    detected = image_preparation_module._detect_target_in_image(
        image,
        target_x=centre,
        target_y=centre,
        detection_cfg={"n_sigma": 3.0, "n_pixels": 20, "n_levels": 32},
        context="test",
    )
    selected = image_preparation_module._apply_contaminant_rule(
        image,
        detected,
        rule="local_excess",
        detection_cfg={"n_pixels": 20},
    )
    labels = np.asarray(detected.detection.segmentation.data)
    star_label = int(labels[40, 200])

    assert star_label not in (0, detected.target_label)
    assert star_label not in selected.released_labels
    assert selected.contaminant_mask[40, 200]
    assert np.all(detected.contaminant_mask[selected.contaminant_mask])
    released_pixels = np.isin(labels, selected.released_labels)
    assert not np.any(selected.contaminant_mask[released_pixels])


# ---------------------------------------------------------------------------
# Configuration, identity and persisted products
# ---------------------------------------------------------------------------


def _minimal_config(tmp_path, **masking):
    config = {
        "object": {"name": "rule_target"},
        "data": {"image_path": str(tmp_path / "image.fits")},
        "outputs": {"root": str(tmp_path / "outputs"), "overwrite": True},
    }

    if masking:
        config["preparation"] = {"masking": masking}

    return config


def test_the_default_rule_is_local_excess(tmp_path):
    resolved = resolve_config(_minimal_config(tmp_path))

    assert resolved["preparation"]["masking"]["contaminant_rule"] == "local_excess"
    assert image_preparation_module._contaminant_rule(resolved) == "local_excess"


def test_an_unknown_configured_rule_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="contaminant_rule"):
        validate_config(_minimal_config(tmp_path, contaminant_rule="every"))


def test_the_rule_of_every_segment_keeps_the_identity_of_older_products(tmp_path):
    def identity(masking):
        config = {
            "object": {"name": "rule_target"},
            "data": {"image_path": "image.fits"},
            "preparation": {"masking": masking},
        }
        return _preparation_config_identity(
            config,
            image_path=tmp_path / "image.fits",
            mask_identity=None,
        )[1]

    older = identity({"negative_threshold_sigma": 5.0})
    declared = identity(
        {"negative_threshold_sigma": 5.0, "contaminant_rule": "all_segments"}
    )
    local = identity(
        {"negative_threshold_sigma": 5.0, "contaminant_rule": "local_excess"}
    )

    assert declared == older
    assert local != older


def _bundle(tmp_path, *, rule, released, contaminant_labels):
    """Return a resolved config and products with three labelled segments."""
    image = np.arange(100, dtype=float).reshape(10, 10)
    fits.writeto(tmp_path / "image.fits", image)

    resolved = resolve_config(
        {
            "object": {"name": "rule_target"},
            "data": {
                "image_path": str(tmp_path / "image.fits"),
                "target_position": {"x": 5, "y": 5},
                "cutout": {"mode": "fixed", "size": 10},
            },
            # The products below hold the segments as detected, not grown.
            "preparation": {
                "masking": {
                    "contaminant_rule": rule,
                    "contaminant_growth_pixels": 0,
                }
            },
            "outputs": {"root": str(tmp_path / "outputs"), "overwrite": True},
        }
    )
    pipeline_paths(resolved)["image_preparation"]["dir"].mkdir(
        parents=True,
        exist_ok=True,
    )

    segmentation = np.zeros((10, 10), dtype=int)
    segmentation[4:7, 4:7] = 1
    segmentation[0:2, 0:2] = 2
    segmentation[8:10, 8:10] = 3
    contaminants = np.isin(segmentation, contaminant_labels)

    products = PreparedObservation(
        cutout=image.copy(),
        prepared_data=image - 1.5,
        noise=np.full((10, 10), 3.0),
        automatic_mask=segmentation > 0,
        fitting_mask=contaminants.copy(),
        target_segment=segmentation == 1,
        contaminant_mask=contaminants,
        segmentation=segmentation,
        background=1.5,
        target_x=5,
        target_y=5,
        target_label=1,
        original_image=image.copy(),
        cutout_bounds={"xmin": 0, "xmax": 10, "ymin": 0, "ymax": 10},
        contaminant_rule=rule,
        released_labels=released,
    )

    return resolved, products


def test_released_segments_survive_the_round_trip(tmp_path):
    resolved, products = _bundle(
        tmp_path,
        rule="local_excess",
        released=(3,),
        contaminant_labels=[2],
    )
    save_image_preparation_products(resolved, products)

    loaded = load_image_preparation_products(resolved)
    metadata_path = (
        pipeline_paths(resolved)["image_preparation"]["dir"]
        / "image_preparation_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())

    assert loaded.contaminant_rule == "local_excess"
    assert loaded.released_labels == (3,)
    assert np.array_equal(loaded.contaminant_mask, products.contaminant_mask)
    assert metadata["contaminant_masking"] == {
        "rule": "local_excess",
        "released_labels": [3],
        "released_pixels": 4,
        "growth_pixels": 0,
    }


@pytest.mark.parametrize(
    "rule, released, contaminant_labels, message",
    [
        ("local_excess", (1,), [2, 3], "released segments"),
        ("local_excess", (3, 3), [2], "released segments"),
        ("local_excess", (3,), [2, 3], "contaminant mask"),
        ("all_segments", (3,), [2], "released segments"),
        ("all_segments", (), [2], "contaminant mask"),
    ],
)
def test_inconsistent_released_segments_are_refused(
    tmp_path,
    rule,
    released,
    contaminant_labels,
    message,
):
    resolved, products = _bundle(
        tmp_path,
        rule=rule,
        released=released,
        contaminant_labels=contaminant_labels,
    )

    with pytest.raises(RuntimeError, match=message):
        save_image_preparation_products(resolved, products)


def test_a_rule_other_than_the_configured_one_is_refused(tmp_path):
    resolved, products = _bundle(
        tmp_path,
        rule="local_excess",
        released=(),
        contaminant_labels=[2, 3],
    )
    resolved["preparation"]["masking"]["contaminant_rule"] = "all_segments"

    with pytest.raises(RuntimeError, match="contaminant rule"):
        save_image_preparation_products(resolved, products)
