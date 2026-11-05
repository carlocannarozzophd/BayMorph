"""The detection of sources for the preparation.

The detection is the one of 0.6 with two changes: the sky of the smoothed
image is a clipped plane, so a gradient no longer raises the threshold, and
the map before the deblending is kept. These tests check both changes and
the rules the detection keeps: excluded pixels take no part, the settings
are checked, and an image without sources gives empty maps.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from photutils.utils.exceptions import NoDetectionsWarning

from baymorph.core.detection import detect_sources_segmentation
from baymorph.preparation.detection import (
    SMOOTHING_SIGMA,
    detect,
    fit_sky_plane,
    plane_image,
)


def _blob(shape, x0, y0, peak, width):
    """Return a circular Gaussian source."""
    rows, columns = np.indices(shape, dtype=float)

    return peak * np.exp(-((columns - x0) ** 2 + (rows - y0) ** 2) / (2 * width**2))


def _noise(shape, seed=3):
    return np.random.default_rng(seed).normal(0.0, 1.0, shape)


def test_the_sky_plane_is_recovered_under_sources() -> None:
    shape = (201, 251)
    truth = (5.0, 0.02, -0.03)
    image = plane_image(truth, shape) + _noise(shape)
    image += _blob(shape, 60.0, 50.0, 40.0, 3.0) + _blob(shape, 180.0, 140.0, 25.0, 5.0)

    plane, scale = fit_sky_plane(image)

    assert plane[0] == pytest.approx(truth[0], abs=0.02)
    assert plane[1] == pytest.approx(truth[1], abs=2e-4)
    assert plane[2] == pytest.approx(truth[2], abs=2e-4)
    # The clipped standard deviation of a unit Gaussian at 3 sigma.
    assert scale == pytest.approx(0.9866, abs=0.01)


def test_a_gradient_does_not_raise_the_threshold() -> None:
    shape = (301, 301)
    flat = _noise(shape) + _blob(shape, 150.0, 150.0, 2.0, 4.0)
    tilted = flat + plane_image((0.0, 0.05, 0.03), shape)

    on_flat = detect(flat)
    on_tilted = detect(tilted)

    with pytest.warns(NoDetectionsWarning):
        old = detect_sources_segmentation(tilted, n_sigma=3.0, n_pixels=20)

    old_noise = (old.threshold_level - old.background_level) / 3.0

    assert on_tilted.noise == pytest.approx(on_flat.noise, rel=0.01)
    assert on_tilted.plane[1] == pytest.approx(0.05, abs=1e-3)
    # The constant sky of 0.6 counts the gradient as noise and misses the
    # source.
    assert old_noise > 5.0 * on_tilted.noise
    assert old.segmentation is None
    assert on_tilted.labels[150, 150] > 0
    assert on_flat.labels[150, 150] > 0


def test_the_map_before_the_deblending_is_kept() -> None:
    shape = (161, 161)
    image = _noise(shape) * 0.2
    image += _blob(shape, 70.0, 80.0, 30.0, 6.0) + _blob(shape, 92.0, 80.0, 20.0, 6.0)

    detection = detect(image)
    labels, parents = detection.labels, detection.parents
    left, right = int(labels[80, 70]), int(labels[80, 92])

    assert left > 0 and right > 0 and left != right
    assert parents[80, 70] == parents[80, 92] > 0
    assert detection.count == 2

    for label in np.unique(labels[labels > 0]):
        owners = np.unique(parents[labels == label])

        assert owners.size == 1 and owners[0] > 0


def test_excluded_pixels_take_no_part() -> None:
    shape = (121, 121)
    image = _noise(shape) + _blob(shape, 60.0, 60.0, 30.0, 4.0)
    image[:, :30] = np.nan
    excluded = ~np.isfinite(image)

    detection = detect(image, excluded=excluded)

    assert not np.any(detection.labels[excluded])
    assert not np.any(detection.parents[excluded])
    assert np.all(detection.smoothed[excluded] == 0.0)
    assert detection.labels[60, 60] > 0

    with pytest.raises(ValueError, match="non-finite pixels that are not excluded"):
        detect(image)

    with pytest.raises(ValueError, match="excludes every pixel"):
        detect(image, excluded=np.ones(shape, dtype=bool))

    with pytest.raises(ValueError, match="the shape"):
        detect(image, excluded=excluded[:-1])


def test_an_image_without_sources_gives_empty_maps() -> None:
    detection = detect(_noise((101, 101)), n_sigma=50.0)

    assert detection.count == 0
    assert not np.any(detection.labels)
    assert not np.any(detection.parents)
    assert detection.threshold == pytest.approx(50.0 * detection.noise)

    constant = detect(np.full((40, 40), 7.0))

    assert constant.noise == 0.0
    assert constant.count == 0


def test_the_maps_are_read_only_and_record_their_settings() -> None:
    detection = detect(_noise((81, 81)) + _blob((81, 81), 40.0, 40.0, 30.0, 3.0))

    for array in (detection.labels, detection.parents, detection.smoothed):
        assert not array.flags.writeable

    assert detection.settings["n_pixels"] == 20
    assert detection.settings["smoothing_sigma"] == SMOOTHING_SIGMA


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"n_sigma": 0.0}, "n_sigma"),
        ({"n_sigma": math.nan}, "n_sigma"),
        ({"n_pixels": 0}, "n_pixels"),
        ({"n_levels": 2.5}, "n_levels"),
        ({"contrast": 1.5}, "contrast"),
        ({"contrast": True}, "contrast"),
    ],
)
def test_invalid_settings_are_refused(settings, message) -> None:
    with pytest.raises(ValueError, match=message):
        detect(_noise((41, 41)), **settings)


def test_a_plane_needs_three_pixels() -> None:
    image = np.ones((5, 5))
    excluded = np.ones((5, 5), dtype=bool)
    excluded[0, :2] = False

    with pytest.raises(ValueError, match="at least three pixels"):
        fit_sky_plane(image, excluded)
