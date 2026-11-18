"""Which detected segments hold a source above the light of the target (AD-050).

The light level is the median of the smoothed image in elliptical annuli about
the target position; a segment holds a source when enough connected pixels
exceed it by the excess of the detection threshold. These tests check the
level against a direct computation, the test of a segment against its rule and
against the function of 0.6, and the release on a galaxy that the deblending
splits: its own pieces and the peaks of the noise on its wings are released,
a neighbour on its envelope and the sources far from it are kept, and the
second pass measures what it states. They also check the two options of the
release: the level of the first pass with the pieces of the target's own
detection in its medians, on a flat disc whose target segment is a small
piece of it, and the plane of the far sky as the level beyond the last
complete annulus, on a source in the corner of an image.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.special import gammaincinv

import baymorph.preparation.contaminants as contaminants_module
from baymorph.core.detection import segments_above_level as segments_above_level_06
from baymorph.preparation.contaminants import (
    EVERY_OTHER_SEGMENT,
    OTHER_DETECTIONS,
    Release,
    light_level,
    release_segments,
    segments_above_level,
)
from baymorph.preparation.detection import Detection, detect, plane_image
from baymorph.preparation.profile import (
    Ellipse,
    ReferenceSky,
    complete_radius,
    ellipse_of_segment,
)
from baymorph.preparation.target import select_target_segment

SIDE = 501
CENTRE = (250.27, 249.87)
NEIGHBOUR = (250.27 + 60.0 * math.cos(2.0), 249.87 + 60.0 * math.sin(2.0))
FAR = ((40.0, 45.0), (455.0, 470.0))


def _square(shape, x, y, half):
    """Return a boolean square of side ``2 half + 1`` centred on (x, y)."""
    mask = np.zeros(shape, dtype=bool)
    mask[y - half : y + half + 1, x - half : x + half + 1] = True

    return mask


def _blob(shape, x, y, peak, width):
    """Return a round Gaussian source of ``peak`` at (x, y)."""
    rows, columns = np.indices(shape, dtype=float)

    return peak * np.exp(-((columns - x) ** 2 + (rows - y) ** 2) / (2 * width**2))


@pytest.fixture(scope="module")
def scene():
    """A Sérsic galaxy of index 4 split by the deblending, with neighbours.

    The galaxy has 15 noise scales at R_e = 40 px and an axis ratio of 0.67;
    a compact neighbour lies on its envelope at 1.5 R_e, and two lie far from
    it. The noise is white, of unit scale.
    """
    shape = (SIDE, SIDE)
    radius, _ = Ellipse(x0=CENTRE[0], y0=CENTRE[1], q=0.67, theta=0.5).coordinates(
        shape
    )
    b = gammaincinv(8.0, 0.5)
    galaxy = 15.0 * np.exp(-b * ((radius / 40.0) ** 0.25 - 1.0))
    image = galaxy + _blob(shape, *NEIGHBOUR, 200.0, 2.0)

    for x, y in FAR:
        image = image + _blob(shape, x, y, 100.0, 2.0)

    image = image + np.random.default_rng(5).normal(0.0, 1.0, shape)
    detection = detect(image)
    selection = select_target_segment(
        detection.labels, CENTRE, parents=detection.parents, flux=detection.smoothed
    )

    return {"image": image, "detection": detection, "label": selection.label}


def _label_at(detection, position):
    """Return the label of the segment at a position."""
    return int(detection.labels[int(round(position[1])), int(round(position[0]))])


# --- The light level ---------------------------------------------------------------


def test_the_light_level_is_the_median_of_each_annulus() -> None:
    rng = np.random.default_rng(3)
    shape = (57, 64)
    data = rng.normal(size=shape)
    masked = rng.random(shape) < 0.3
    ellipse = Ellipse(x0=30.4, y0=25.7, q=0.6, theta=0.4)

    level = light_level(data, ellipse, masked)
    radius, _ = ellipse.coordinates(shape)
    index = np.floor(radius).astype(int)
    checked = 0

    for number in range(index.max() + 1):
        members = (index == number) & ~masked

        if np.any(members):
            np.testing.assert_allclose(
                level[index == number], np.median(data[members]), rtol=0, atol=1e-15
            )
            checked += 1

    assert checked > 30


def test_an_empty_annulus_takes_the_value_interpolated_from_its_neighbours() -> None:
    shape = (41, 41)
    ellipse = Ellipse(x0=20.0, y0=20.0, q=1.0, theta=0.0)
    radius, _ = ellipse.coordinates(shape)
    index = np.floor(radius).astype(int)
    data = (10.0 - radius).clip(0.0)
    masked = (index == 5) | (index == 6)

    level = light_level(data, ellipse, masked)
    four = np.median(data[index == 4])
    seven = np.median(data[index == 7])

    np.testing.assert_allclose(level[index == 5], four + (seven - four) / 3.0)
    np.testing.assert_allclose(level[index == 6], four + 2.0 * (seven - four) / 3.0)


def test_beyond_the_last_annulus_with_pixels_the_level_keeps_its_value() -> None:
    shape = (41, 41)
    ellipse = Ellipse(x0=20.0, y0=20.0, q=1.0, theta=0.0)
    radius, _ = ellipse.coordinates(shape)
    data = np.ones(shape)
    data[radius >= 12.0] = 5.0
    masked = radius >= 12.0

    level = light_level(data, ellipse, masked)

    assert np.all(level[radius >= 12.0] == 1.0)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"masked": np.ones((9, 9), bool)}, "no unmasked pixel"),
        ({"data": np.full((9, 9), np.nan)}, "finite"),
        ({"masked": np.zeros((5, 5), bool)}, "shape"),
    ],
)
def test_the_light_level_refuses_what_it_cannot_read(change, message) -> None:
    arguments = {"data": np.zeros((9, 9)), "masked": np.zeros((9, 9), bool)}
    arguments.update(change)

    with pytest.raises(ValueError, match=message):
        light_level(
            arguments["data"],
            Ellipse(x0=4.0, y0=4.0, q=1.0, theta=0.0),
            arguments["masked"],
        )


# --- The test of a segment -----------------------------------------------------------


def test_a_segment_holds_a_source_when_enough_connected_pixels_clear_the_level():
    shape = (40, 40)
    image = np.zeros(shape)
    labels = np.zeros(shape, dtype=int)
    source = _square(shape, 10, 10, 3)
    labels[source] = 1
    image[source] = 2.0

    holds, footprint = segments_above_level(
        image, labels, np.array([1]), level=np.zeros(shape), excess=1.0, n_pixels=20
    )

    assert holds.tolist() == [True]
    assert np.array_equal(footprint, source)

    holds, footprint = segments_above_level(
        image,
        labels,
        np.array([1]),
        level=np.full(shape, 1.5),
        excess=1.0,
        n_pixels=20,
    )

    assert holds.tolist() == [False]
    assert not np.any(footprint)


def test_the_footprint_is_the_part_of_the_segment_above_the_level() -> None:
    shape = (40, 40)
    image = np.zeros(shape)
    labels = np.zeros(shape, dtype=int)
    segment = _square(shape, 20, 20, 10)
    core = _square(shape, 15, 20, 3)
    labels[segment] = 4
    image[segment] = 0.5
    image[core] = 3.0

    holds, footprint = segments_above_level(
        image, labels, np.array([4]), level=np.zeros(shape), excess=1.0, n_pixels=20
    )

    assert holds.tolist() == [True]
    assert np.array_equal(footprint, core)


def test_too_few_or_disconnected_pixels_do_not_hold_a_source() -> None:
    shape = (40, 40)
    image = np.zeros(shape)
    labels = np.zeros(shape, dtype=int)
    small = _square(shape, 10, 10, 1)
    labels[small] = 1
    image[small] = 5.0
    checker = _square(shape, 28, 28, 6)
    labels[checker] = 2
    rows, columns = np.indices(shape)
    scattered = checker & (columns % 3 == 0) & (rows % 3 == 0)
    image[scattered] = 5.0

    holds, footprint = segments_above_level(
        image, labels, np.array([1, 2]), level=np.zeros(shape), excess=1.0, n_pixels=5
    )

    assert np.count_nonzero(scattered) >= 5
    assert holds.tolist() == [True, False]
    assert np.array_equal(footprint, small)


def test_pixels_outside_the_segment_do_not_count() -> None:
    shape = (30, 30)
    image = np.zeros(shape)
    labels = np.zeros(shape, dtype=int)
    labels[_square(shape, 10, 10, 1)] = 1
    image[_square(shape, 10, 10, 4)] = 5.0

    holds, footprint = segments_above_level(
        image, labels, np.array([1]), level=np.zeros(shape), excess=1.0, n_pixels=10
    )

    assert holds.tolist() == [False]
    assert not np.any(footprint)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"excess": 0.0}, "excess"),
        ({"excess": np.nan}, "excess"),
        ({"n_pixels": 0}, "n_pixels"),
        ({"tested": np.array([3])}, "not a segment"),
        ({"level": np.zeros((5, 5))}, "same shape"),
        ({"level": np.full((20, 20), np.inf)}, "finite"),
        ({"tested": np.array([[1]])}, "one-dimensional"),
    ],
)
def test_invalid_inputs_are_rejected(change, message) -> None:
    shape = (20, 20)
    labels = np.zeros(shape, dtype=int)
    labels[_square(shape, 5, 5, 2)] = 1
    arguments = {
        "tested": np.array([1]),
        "level": np.zeros(shape),
        "excess": 1.0,
        "n_pixels": 5,
    }
    arguments.update(change)

    with pytest.raises(ValueError, match=message):
        segments_above_level(
            np.zeros(shape),
            labels,
            arguments["tested"],
            level=arguments["level"],
            excess=arguments["excess"],
            n_pixels=arguments["n_pixels"],
        )


def test_the_test_agrees_with_the_rule_of_0_6(scene) -> None:
    detection = scene["detection"]
    labels = detection.labels
    target = labels == scene["label"]
    others = np.unique(labels[(labels > 0) & ~target])
    ellipse = ellipse_of_segment(scene["image"], target, x0=CENTRE[0], y0=CENTRE[1])
    level = light_level(
        detection.smoothed, ellipse, ((labels > 0) & ~target) | detection.excluded
    )

    holds, footprint = segments_above_level(
        detection.smoothed,
        labels,
        others,
        level=level,
        excess=detection.threshold,
        n_pixels=20,
    )
    expected = segments_above_level_06(
        detection.smoothed,
        labels,
        others,
        level=level,
        excess=detection.threshold,
        n_pixels=20,
    )

    assert holds.tolist() == expected.tolist()
    assert 0 < np.count_nonzero(holds) < others.size
    assert np.all(np.isin(labels[footprint], others[holds]))
    assert np.all(
        detection.smoothed[footprint] - level[footprint] > detection.threshold
    )


def test_on_the_sky_every_segment_holds_a_source(scene) -> None:
    detection = scene["detection"]
    labels = detection.labels
    every = np.unique(labels[labels > 0])

    holds, footprint = segments_above_level(
        detection.smoothed,
        labels,
        every,
        level=np.zeros(labels.shape),
        excess=detection.threshold,
        n_pixels=20,
    )

    assert np.all(holds)
    assert np.array_equal(footprint, labels > 0)


# --- The release -----------------------------------------------------------------------


def test_the_release_frees_the_galaxy_and_keeps_its_neighbours(scene) -> None:
    detection = scene["detection"]
    release = release_segments(
        detection, scene["label"], scene["image"], x0=CENTRE[0], y0=CENTRE[1]
    )
    neighbour = _label_at(detection, NEIGHBOUR)

    # The neighbour joins the detection of the galaxy through its envelope;
    # every other piece of that detection, and every peak of the noise lifted
    # on the wings, is released.
    assert neighbour in release.siblings
    assert release.released_siblings == tuple(
        number for number in release.siblings if number != neighbour
    )
    assert len(release.released) >= 5
    assert neighbour in release.kept
    assert all(_label_at(detection, position) in release.kept for position in FAR)
    assert release.footprint[int(round(NEIGHBOUR[1])), int(round(NEIGHBOUR[0]))]
    assert np.all(release.kept_segments(detection.labels)[release.footprint])
    assert set(release.kept).isdisjoint(release.released)
    assert sorted(release.kept + release.released) == sorted(
        int(number)
        for number in np.unique(detection.labels[detection.labels > 0])
        if number != scene["label"]
    )


def test_the_footprint_of_a_neighbour_on_the_envelope_is_smaller_than_its_segment(
    scene,
) -> None:
    detection = scene["detection"]
    release = release_segments(
        detection, scene["label"], scene["image"], x0=CENTRE[0], y0=CENTRE[1]
    )
    segment = detection.labels == _label_at(detection, NEIGHBOUR)

    assert np.count_nonzero(release.footprint & segment) < np.count_nonzero(segment)


def test_the_second_pass_measures_what_it_states(scene) -> None:
    detection = scene["detection"]
    labels = detection.labels
    image = scene["image"]
    target = labels == scene["label"]
    others = np.unique(labels[(labels > 0) & ~target])
    first = release_segments(
        detection, scene["label"], image, x0=CENTRE[0], y0=CENTRE[1], passes=1
    )
    second = release_segments(
        detection, scene["label"], image, x0=CENTRE[0], y0=CENTRE[1], passes=2
    )
    body = target | np.isin(labels, first.released_siblings)
    ellipse = ellipse_of_segment(image, body, x0=CENTRE[0], y0=CENTRE[1])
    level = light_level(
        detection.smoothed,
        ellipse,
        np.isin(labels, first.kept) | detection.excluded,
    )
    holds, footprint = segments_above_level(
        detection.smoothed,
        labels,
        others,
        level=level,
        excess=detection.threshold,
        n_pixels=20,
    )

    assert first.ellipse == ellipse_of_segment(
        image, target, x0=CENTRE[0], y0=CENTRE[1]
    )
    assert second.first_released == first.released
    assert second.ellipse == ellipse
    assert second.kept == tuple(int(number) for number in others[holds])
    assert np.array_equal(second.footprint, footprint)
    assert second.changed == (second.released != first.released)
    assert not first.changed


def test_a_lone_target_releases_nothing() -> None:
    shape = (121, 121)
    image = _blob(shape, 60.2, 59.8, 50.0, 4.0)
    image = image + np.random.default_rng(2).normal(0.0, 1.0, shape)
    detection = detect(image)
    label = int(detection.labels[60, 60])

    release = release_segments(detection, label, image, x0=60.2, y0=59.8)

    assert release.kept == () and release.released == ()
    assert not np.any(release.footprint)


def test_the_release_refuses_a_wrong_label_or_number_of_passes(scene) -> None:
    detection = scene["detection"]

    with pytest.raises(ValueError, match="1 or 2 passes"):
        release_segments(
            detection, scene["label"], scene["image"], x0=1.0, y0=1.0, passes=3
        )

    with pytest.raises(ValueError, match="not a segment"):
        release_segments(detection, 10_000, scene["image"], x0=1.0, y0=1.0)

    with pytest.raises(ValueError, match="shape"):
        release_segments(detection, scene["label"], np.zeros((5, 5)), x0=1.0, y0=1.0)


def test_the_record_of_a_release_is_json_serializable(scene) -> None:
    release = release_segments(
        scene["detection"], scene["label"], scene["image"], x0=CENTRE[0], y0=CENTRE[1]
    )
    record = json.loads(json.dumps(release.as_dict()))

    assert isinstance(release, Release)
    assert record["kept"] == len(release.kept)
    assert record["released_labels"] == list(release.released)
    assert record["passes"] == 2
    assert not release.footprint.flags.writeable


# --- The options of the release ------------------------------------------------------


@pytest.fixture(scope="module")
def disc():
    """A flat disc of 1.5 noise scales and 105 px, which the deblending splits.

    The target segment is a piece of 296 px of the noise of the disc, with an
    axis ratio of about 0.5; the other 108 pieces of its detection are its
    siblings.
    """
    shape = (601, 601)
    centre = (300.27, 299.87)
    rows, columns = np.indices(shape)
    image = np.where(np.hypot(columns - centre[0], rows - centre[1]) <= 105.0, 1.5, 0.0)
    image = image + np.random.default_rng(3).normal(0.0, 1.0, shape)
    detection = detect(image)
    selection = select_target_segment(
        detection.labels, centre, parents=detection.parents, flux=detection.smoothed
    )

    return {
        "image": image,
        "detection": detection,
        "label": selection.label,
        "centre": centre,
    }


def test_the_level_of_the_other_detections_holds_the_light_of_a_split_galaxy(
    disc,
) -> None:
    # Masking the other pieces of the disc leaves the annuli they cover empty,
    # and the level interpolated across them falls towards the sky; with the
    # pieces in its medians, the level is the light of the disc.
    detection = disc["detection"]
    labels = detection.labels
    target = labels == disc["label"]
    parent = detection.parents[target][0]
    x0, y0 = disc["centre"]
    ellipse = ellipse_of_segment(disc["image"], target, x0=x0, y0=y0)
    every = light_level(
        detection.smoothed, ellipse, ((labels > 0) & ~target) | detection.excluded
    )
    detections = light_level(
        detection.smoothed,
        ellipse,
        ((labels > 0) & (detection.parents != parent)) | detection.excluded,
    )

    for distance in (50.0, 80.0, 100.0):
        column = int(round(x0 + distance * math.cos(ellipse.theta)))
        row = int(round(y0 + distance * math.sin(ellipse.theta)))

        assert detections[row, column] == pytest.approx(1.5, abs=0.06)
        assert every[row, column] < 1.1

    first = release_segments(detection, disc["label"], disc["image"], x0=x0, y0=y0)
    other = release_segments(
        detection,
        disc["label"],
        disc["image"],
        x0=x0,
        y0=y0,
        passes=1,
        level_masks=OTHER_DETECTIONS,
    )

    assert len(first.siblings) >= 100
    assert len(first.released_siblings) <= 0.1 * len(first.siblings)
    assert len(other.released_siblings) >= 0.5 * len(other.siblings)
    assert other.level_masks == OTHER_DETECTIONS
    assert first.level_masks == EVERY_OTHER_SEGMENT
    assert other.as_dict()["level"]["first_pass_leaves_out"] == OTHER_DETECTIONS


def test_the_level_of_the_other_detections_keeps_a_neighbour_on_the_envelope(
    scene,
) -> None:
    # The neighbour on the envelope is a sibling of the target, so it enters
    # the medians of the level, which stay those of the galaxy: a median reads
    # through a compact source.
    detection = scene["detection"]
    release = release_segments(
        detection,
        scene["label"],
        scene["image"],
        x0=CENTRE[0],
        y0=CENTRE[1],
        passes=1,
        level_masks=OTHER_DETECTIONS,
    )
    neighbour = _label_at(detection, NEIGHBOUR)

    assert neighbour in release.siblings
    assert neighbour in release.kept
    assert all(_label_at(detection, position) in release.kept for position in FAR)
    assert len(release.released_siblings) == len(release.siblings) - 1


def _detection_of(smoothed, labels, threshold, plane=(0.0, 0.0, 0.0)) -> Detection:
    """Return a detection built from its parts, each segment its own detection."""
    segments = np.asarray(labels, dtype=np.int32)

    return Detection(
        labels=segments,
        parents=segments.copy(),
        smoothed=np.asarray(smoothed, dtype=float),
        plane=plane,
        noise=threshold / 3.0,
        threshold=threshold,
        excluded=np.zeros(segments.shape, dtype=bool),
        settings={"n_sigma": 3.0, "n_pixels": 20, "n_levels": 32, "contrast": 0.001},
    )


def _far_sky(radius_of, plane=(0.0, 0.0, 0.0)):
    """Return a function giving a flat far sky beyond the radius of an ellipse."""

    def far_sky(ellipse):
        return ReferenceSky(
            radius=radius_of(ellipse),
            plane=plane,
            error=0.0,
            scale=1.0,
            pixels=10_000,
            inner_minus_outer=0.0,
            inner_outer_error=1.0,
        )

    return far_sky


def test_beyond_the_sky_radius_the_level_is_the_plane_of_the_sky() -> None:
    shape = (61, 81)
    ellipse = Ellipse(x0=30.2, y0=29.9, q=0.8, theta=0.3)
    radius, _ = ellipse.coordinates(shape)
    data = np.where(radius < 12.0, 5.0, 1.0) + 0.01 * radius
    masked = np.zeros(shape, dtype=bool)
    plane = (0.5, 0.01, -0.02)

    level = light_level(data, ellipse, masked, sky=plane, sky_radius=25.0)

    rows, columns = np.indices(shape)
    beyond = radius >= 25.0
    expected = 0.5 + 0.01 * (columns - 30.2) - 0.02 * (rows - 29.9)

    np.testing.assert_allclose(level[beyond], expected[beyond])
    # Inside the radius the medians read only the pixels inside it.
    np.testing.assert_array_equal(
        level[~beyond], light_level(data, ellipse, masked)[~beyond]
    )

    with pytest.raises(ValueError, match="needs its plane and its radius"):
        light_level(data, ellipse, masked, sky=plane)

    with pytest.raises(ValueError, match="finite plane"):
        light_level(data, ellipse, masked, sky=(0.0, math.nan, 0.0), sky_radius=5.0)

    with pytest.raises(ValueError, match="finite plane"):
        light_level(data, ellipse, masked, sky=plane, sky_radius=-1.0)


def test_beyond_the_last_complete_annulus_a_segment_meets_the_sky() -> None:
    # A source in the far corner of the image, on a bump of the large-scale
    # field of the sky. The annuli there are short arcs on the bump, whose
    # median lifts the level, and the source is released; against the plane
    # of the far sky it holds a source, as the detection found.
    shape = (201, 201)
    x0, y0 = 70.3, 69.8
    rows, columns = np.indices(shape)
    labels = np.zeros(shape, dtype=np.int32)
    labels[np.hypot(columns - x0, rows - y0) <= 15.0] = 1
    labels[180:186, 180:186] = 2
    smoothed = np.where((columns >= 150) & (rows >= 150), 1.0, 0.0)
    smoothed[labels == 1] = 10.0
    smoothed[labels == 2] = 1.9
    detection = _detection_of(smoothed, labels, threshold=1.2)
    radius_of = lambda ellipse: complete_radius(shape, ellipse)  # noqa: E731

    median = release_segments(detection, 1, smoothed, x0=x0, y0=y0, passes=1)
    sky = release_segments(
        detection,
        1,
        smoothed,
        x0=x0,
        y0=y0,
        passes=1,
        far_sky=_far_sky(radius_of),
    )

    assert median.released == (2,) and median.sky_radius is None
    assert sky.kept == (2,)
    assert sky.sky_radius == pytest.approx(radius_of(sky.ellipse))
    assert sky.sky_radius < 100.0
    assert sky.as_dict()["level"]["beyond_the_last_complete_annulus"] == (
        "the plane of the far sky"
    )


def test_the_plane_of_the_far_sky_enters_the_level_less_that_of_the_detection():
    shape = (41, 61)
    labels = np.zeros(shape, dtype=np.int32)
    labels[15:25, 25:35] = 1
    detection_plane = (2.0, 0.01, -0.03)
    detection = _detection_of(np.zeros(shape), labels, 1.0, plane=detection_plane)
    ellipse = Ellipse(x0=30.4, y0=19.6, q=0.8, theta=0.3)
    far_plane = (2.5, 0.02, 0.01)

    sky, radius = contaminants_module._sky_of(
        _far_sky(lambda _: 12.0, far_plane), ellipse, detection
    )

    rows, columns = np.indices(shape)
    expected = (
        far_plane[0]
        + far_plane[1] * (columns - ellipse.x0)
        + far_plane[2] * (rows - ellipse.y0)
        - plane_image(detection_plane, shape)
    )
    got = sky[0] + sky[1] * (columns - ellipse.x0) + sky[2] * (rows - ellipse.y0)

    np.testing.assert_allclose(got, expected, atol=1e-12)
    assert radius == 12.0

    with pytest.raises(TypeError, match="ReferenceSky"):
        contaminants_module._sky_of(lambda _: (0.0, 0.0, 0.0), ellipse, detection)

    with pytest.raises(ValueError, match="leaves out one of"):
        release_segments(
            detection, 1, np.zeros(shape), x0=30.4, y0=19.6, level_masks="none"
        )
