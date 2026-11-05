"""The cutout stage of the preparation: where the sky begins.

The stage measures once, on the working region, where the light of the galaxy
falls below the sky, and sets from it the cutout, out to 1 sky scale plus the
margin by default, and the noise region, beyond 0.3 sky scales. These tests
check the radii against Sérsic galaxies whose radii they know, the sides and
the bounds of automatic and fixed cutouts, every stop with its remedy, the
warnings, the identity and the record, and the acceptance of the stage: on a
galaxy of index 4 and axis ratio 0.67 on a sky gradient, the light at each
radius lies near its threshold and the radii do not follow the side of the
image.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.special import gammaincinv

import baymorph.preparation.cutout as cutout_module
import baymorph.preparation.opening as opening
from baymorph.preparation.cutout import (
    AUTOMATIC_CUTOUT_LIMITATION,
    CUTOUT_LIGHT_THRESHOLD,
    CUTOUT_STAGE,
    FAR_SKY_LIMITATION,
    MINIMUM_NOISE_PIXELS,
    NOISE_LIGHT_THRESHOLD,
    cutout_from_settings,
    noise_region,
    resolve_cutout,
    rule,
    source_mask,
)
from baymorph.preparation.opening import open_image
from baymorph.preparation.profile import (
    MINIMUM_REFERENCE_PIXELS,
    Ellipse,
    ReferenceSky,
    elliptical_profile,
)
from baymorph.preparation.schema import read_preparation_settings
from baymorph.preparation.store import ProductStore
from baymorph.preparation.target import resolve_target

CENTRE = (300.27, 299.87)
"""The target position of the 601-pixel scenes, near the pixel (300, 300)."""


@pytest.fixture(autouse=True)
def _fresh_headers():
    """Start and end every test without headers kept for reuse."""
    opening.forget_kept_headers()
    yield
    opening.forget_kept_headers()


def _sersic(shape, x0, y0, *, level=15.0, re=8.0, n=1.0, q=0.8, theta=0.4):
    """Return a Sérsic galaxy of ``level`` sky scales at ``re``, noise-free."""
    radius, _ = Ellipse(x0=x0, y0=y0, q=q, theta=theta).coordinates(shape)
    b = gammaincinv(2.0 * n, 0.5)

    return level * np.exp(-b * ((radius / re) ** (1.0 / n) - 1.0))


def _radius_at(level, *, at_re=15.0, re=8.0, n=1.0) -> float:
    """Return the radius at which that Sérsic galaxy falls to ``level``."""
    b = gammaincinv(2.0 * n, 0.5)

    return re * (1.0 + math.log(at_re / level) / b) ** n


def _noise(shape, seed, *, correlated=False):
    """Return noise of unit scale, white or correlated between neighbours."""
    noise = np.random.default_rng(seed).normal(0.0, 1.0, shape)

    if correlated:
        noise = gaussian_filter(noise, 0.5)
        noise /= noise.std()

    return noise


def _galaxy(seed=1):
    """Return the 601-pixel scene of an exponential galaxy at CENTRE."""
    shape = (601, 601)

    return _sersic(shape, *CENTRE) + _noise(shape, seed)


def _inputs(tmp_path: Path, image, position=CENTRE, name="image.fits", **settings):
    """Write ``image``, open it and check the target position."""
    path = tmp_path / name
    fits.PrimaryHDU(np.asarray(image, np.float32)).writeto(path)
    opened = open_image(path, project_root=tmp_path, **settings)

    return opened, resolve_target(opened, {"x": position[0], "y": position[1]})


# --- The radii and the cutout ---------------------------------------------------


def test_the_automatic_cutout_reaches_the_light_threshold_plus_the_margin(
    tmp_path,
) -> None:
    opened, target = _inputs(tmp_path, _galaxy())

    cutout = resolve_cutout(opened, target, mode="auto")
    side = 2 * math.ceil(cutout.cutout_radius + 20.0) + 1

    assert cutout.mode == "auto"
    assert cutout.cutout_radius == pytest.approx(
        _radius_at(CUTOUT_LIGHT_THRESHOLD), rel=0.1
    )
    assert cutout.noise_radius == pytest.approx(
        _radius_at(NOISE_LIGHT_THRESHOLD), rel=0.1
    )
    # Centred on the pixel of the target position, (300, 300).
    assert cutout.shape == (side, side)
    assert cutout.bounds == {
        "xmin": 300 - side // 2,
        "xmax": 301 + side // 2,
        "ymin": 300 - side // 2,
        "ymax": 301 + side // 2,
    }
    assert cutout.noise_pixels >= MINIMUM_NOISE_PIXELS
    assert cutout.warnings == ()
    # The annuli follow the galaxy; the position angle is defined modulo pi.
    assert cutout.ellipse.q == pytest.approx(0.8, abs=0.02)
    assert math.cos(2.0 * (cutout.ellipse.theta - 0.4)) == pytest.approx(1.0, abs=2e-3)
    assert cutout.reference.scale == pytest.approx(1.0, rel=0.02)


def test_the_radii_follow_the_declared_thresholds(tmp_path) -> None:
    opened, target = _inputs(tmp_path, _galaxy())

    cutout = resolve_cutout(
        opened, target, mode="auto", cutout_threshold=3.0, noise_threshold=1.0
    )

    assert cutout.cutout_radius == pytest.approx(_radius_at(3.0), rel=0.1)
    assert cutout.noise_radius == pytest.approx(_radius_at(1.0), rel=0.1)
    assert cutout.record["radii"]["cutout"]["threshold"] == 3.0
    assert cutout.record["radii"]["noise"]["threshold"] == 1.0


def test_min_size_enlarges_and_max_size_stops_an_automatic_cutout(tmp_path) -> None:
    opened, target = _inputs(tmp_path, _galaxy())
    needed = resolve_cutout(opened, target, mode="auto").shape[0]

    enlarged = resolve_cutout(opened, target, mode="auto", min_size=needed + 70)
    even = resolve_cutout(opened, target, mode="auto", min_size=needed + 19)
    exact = resolve_cutout(opened, target, mode="auto", max_size=needed)

    assert enlarged.shape == (needed + 70, needed + 70)
    assert enlarged.record["cutout"]["needed_side"] == needed
    # An even side keeps one pixel more before the target pixel than after.
    half = (needed + 19) // 2
    assert even.bounds["xmin"] == 300 - half
    assert even.bounds["xmax"] == 300 + half
    assert exact.shape == (needed, needed)

    with pytest.raises(
        ValueError,
        match=f"needs a side of {needed} pixels.*max_size allows {needed - 2}",
    ):
        resolve_cutout(opened, target, mode="auto", max_size=needed - 2)


def test_a_fixed_cutout_keeps_its_side_and_is_checked_against_the_radius(
    tmp_path,
) -> None:
    opened, target = _inputs(tmp_path, _galaxy())
    automatic = resolve_cutout(opened, target, mode="auto")

    fixed = resolve_cutout(opened, target, mode="fixed", size=201)
    small = resolve_cutout(opened, target, mode="fixed", size=41)

    assert fixed.mode == "fixed"
    assert fixed.shape == (201, 201)
    assert fixed.bounds["xmin"] == 200
    # The radii and the noise region do not depend on the mode.
    assert fixed.cutout_radius == automatic.cutout_radius
    assert fixed.noise_radius == automatic.noise_radius
    assert fixed.noise_pixels == automatic.noise_pixels
    assert fixed.warnings == ()
    assert fixed.record["limitations"] == [FAR_SKY_LIMITATION]
    assert fixed.identity != automatic.identity

    assert small.shape == (41, 41)
    assert len(small.warnings) == 1
    assert "The fixed cutout of 41 pixels is smaller than the" in small.warnings[0]
    assert small.record["warnings"] == list(small.warnings)


def test_a_cutout_that_leaves_the_image_stops_with_the_side_that_fits(
    tmp_path,
) -> None:
    near = (60.3, 300.2)
    shape = (601, 601)
    opened, target = _inputs(tmp_path, _sersic(shape, *near) + _noise(shape, 2), near)

    # Sixty pixels of room on the left of the pixel (60, 300).
    with pytest.raises(
        ValueError,
        match=(
            "fixed cutout of 201 pixels.*leaves the image of 601 x 601 pixels: at "
            "most 121 pixels fit.*smaller data.cutout.size"
        ),
    ):
        resolve_cutout(opened, target, mode="fixed", size=201)

    with pytest.raises(
        ValueError, match="min_size sets the side: lower it to at most 121"
    ):
        resolve_cutout(opened, target, mode="auto", min_size=201)

    assert resolve_cutout(opened, target, mode="fixed", size=121).bounds["xmin"] == 0

    closer = (30.3, 300.2)
    opened, target = _inputs(
        tmp_path, _sersic(shape, *closer) + _noise(shape, 2), closer, name="edge.fits"
    )

    with pytest.raises(ValueError, match=r"at most 61 pixels fit.*The galaxy needs"):
        resolve_cutout(opened, target, mode="auto")


def test_a_working_region_keeps_positions_in_pixels_of_the_whole_image(
    tmp_path,
) -> None:
    shape = (1001, 1001)
    position = (650.3, 420.8)
    image = _sersic(shape, *position) + _noise(shape, 7)
    opened, target = _inputs(
        tmp_path,
        image,
        position,
        region={"size": 601.0},
        target_position={"x": position[0], "y": position[1]},
        pixel_scale=1.0,
    )

    cutout = resolve_cutout(opened, target, mode="auto")
    half = cutout.shape[0] // 2

    assert cutout.origin == (350, 121)
    assert cutout.bounds["xmin"] == 650 - half
    assert cutout.bounds["ymin"] == 421 - half
    assert cutout.ellipse.x0 == pytest.approx(300.3)
    assert cutout.record["ellipse"]["x0"] == pytest.approx(650.3)
    assert cutout.record["origin"] == {"x": 350, "y": 121}
    assert cutout.record["working_region"] == {
        "xmin": 350,
        "xmax": 951,
        "ymin": 121,
        "ymax": 722,
    }

    with pytest.raises(
        ValueError, match=r"leaves the working region x 350-951, y 121-722.*data.region"
    ):
        resolve_cutout(opened, target, mode="fixed", size=701)


def test_pixels_without_data_end_the_annuli_and_stay_out_of_the_noise_region(
    tmp_path,
) -> None:
    image = _galaxy()
    # A band without data, as a mosaic leaves, 150 pixels right of the galaxy.
    image[:, 450:470] = np.nan
    opened, target = _inputs(tmp_path, image)
    whole, whole_target = _inputs(tmp_path, _galaxy(), name="whole.fits")

    cutout = resolve_cutout(opened, target, mode="auto")
    reference = resolve_cutout(whole, whole_target, mode="auto")
    region = noise_region(
        source_mask(cutout.detection.labels, opened.no_data),
        cutout.ellipse,
        cutout.noise_radius,
    )

    assert cutout.reference.radius < 0.6 * reference.reference.radius
    assert cutout.noise_radius == pytest.approx(reference.noise_radius, rel=0.1)
    assert not np.any(region & opened.no_data)
    assert np.count_nonzero(region) == cutout.noise_pixels
    assert cutout.record["cutout"]["no_data_pixels"] == 0


def test_the_noise_region_lies_beyond_its_radius_outside_every_source() -> None:
    labels = np.zeros((101, 101), np.int32)
    labels[10:13, 10:13] = 1
    no_data = np.zeros((101, 101), bool)
    no_data[90:, :] = True
    ellipse = Ellipse(x0=50.0, y0=50.0, q=1.0, theta=0.0)

    masked = source_mask(labels, no_data)
    region = noise_region(masked, ellipse, 30.0)
    radius, _ = ellipse.coordinates(labels.shape)

    # Every source is grown by 20 pixels; the pixels without data are not.
    assert masked[11, 31] and not masked[11, 33]
    assert masked[90, 0] and not masked[89, 0]
    assert region[89, 0]
    assert not np.any(region & (radius < 30.0))
    assert not np.any(region & masked)

    with pytest.raises(ValueError, match="without data"):
        source_mask(labels, np.zeros((5, 5), bool))


# --- The stops ----------------------------------------------------------------------


def test_a_galaxy_whose_light_never_falls_below_the_threshold_stops(tmp_path) -> None:
    # A peaked profile cut sharply 104 pixels from its centre, in an image 201
    # pixels high: the last complete annulus lies at 100 pixels, where the
    # galaxy still holds about half a sky scale, and the far sky is clean.
    shape = (201, 2401)
    rows, columns = np.indices(shape)
    radius = np.hypot(columns - 1200.0, rows - 100.0)
    galaxy = np.where(radius <= 104.0, 148.0 * np.exp(-(radius**2) / 1756.0), 0.0)
    opened, target = _inputs(tmp_path, galaxy + _noise(shape, 3), (1200.0, 100.0))

    with pytest.raises(
        ValueError,
        match=(
            "The sky is not reached.*below 0.3 sky scales within the last complete "
            "annulus.*The noise region needs.*preparation.noise.light_threshold.*Use "
            "an image that extends further"
        ),
    ):
        resolve_cutout(opened, target, mode="auto")

    # With the thresholds the other way round, the noise region is reached and
    # the cutout is not.
    with pytest.raises(
        ValueError, match="The automatic cutout needs.*data.cutout.light_threshold"
    ):
        resolve_cutout(
            opened, target, mode="auto", cutout_threshold=0.3, noise_threshold=2.0
        )

    fixed = resolve_cutout(
        opened,
        target,
        mode="fixed",
        size=151,
        cutout_threshold=0.3,
        noise_threshold=2.0,
    )

    assert fixed.cutout_radius is None
    assert fixed.record["cutout"]["needed_side"] is None
    assert "is not checked against it" in fixed.warnings[0]


def test_galaxy_light_on_the_far_sky_stops(tmp_path) -> None:
    shape = (601, 601)
    rows, columns = np.indices(shape)
    envelope = 3.0 * np.exp(-np.hypot(columns - CENTRE[0], rows - CENTRE[1]) / 150.0)
    opened, target = _inputs(tmp_path, _galaxy(seed=4) + envelope)

    with pytest.raises(
        ValueError,
        match="inner and outer halves of the far sky.*times the error, more than 3",
    ):
        resolve_cutout(opened, target, mode="auto")


def test_a_far_sky_without_enough_pixels_stops(tmp_path) -> None:
    image = _galaxy()
    radius, _ = Ellipse(x0=CENTRE[0], y0=CENTRE[1], q=0.8, theta=0.4).coordinates(
        image.shape
    )
    # No data beyond the ellipse of 250 pixels, which leaves the far sky only
    # slivers.
    image[radius > 250.0] = np.nan
    opened, target = _inputs(tmp_path, image)

    with pytest.raises(
        ValueError,
        match=f"The far sky.*fewer than the {MINIMUM_REFERENCE_PIXELS} that its plane",
    ):
        resolve_cutout(opened, target, mode="auto")


def test_a_far_sky_without_scatter_or_directions_stops() -> None:
    flat = ReferenceSky(
        radius=100.0,
        plane=(0.0, 0.0, 0.0),
        error=0.0,
        scale=0.0,
        pixels=20_000,
        inner_minus_outer=0.0,
        inner_outer_error=0.0,
    )

    with pytest.raises(ValueError, match="no scatter about its plane"):
        cutout_module._check_far_sky(flat, 100.0, "")

    with pytest.raises(ValueError, match="enough directions"):
        cutout_module._check_far_sky(
            replace(flat, scale=1.0, inner_outer_error=math.nan), 100.0, ""
        )


def test_a_noise_region_with_too_few_pixels_stops(tmp_path) -> None:
    shape = (301, 301)
    centre = (150.27, 149.87)
    opened, target = _inputs(
        tmp_path, _sersic(shape, *centre) + _noise(shape, 5), centre
    )

    with pytest.raises(
        ValueError,
        match=f"The noise region.*fewer than the {MINIMUM_NOISE_PIXELS} that the sky",
    ):
        resolve_cutout(opened, target, mode="auto")


def test_a_target_on_the_edge_of_the_image_stops(tmp_path) -> None:
    edge = (-0.4, 300.2)
    shape = (601, 601)
    opened, target = _inputs(tmp_path, _sersic(shape, *edge) + _noise(shape, 2), edge)

    with pytest.raises(
        ValueError, match=r"No annulus about the target position \(-0.4, 300.2\)"
    ):
        resolve_cutout(opened, target, mode="auto")


def test_a_galaxy_split_into_pieces_stops_when_its_profile_is_empty(
    tmp_path,
) -> None:
    # A flat disc of 1.5 sky scales: the deblending splits its noise into
    # many segments, and every piece but the target segment is masked.
    shape = (201, 2401)
    rows, columns = np.indices(shape)
    disc = np.where(np.hypot(columns - 1200.0, rows - 100.0) <= 105.0, 1.5, 0.0)
    opened, target = _inputs(tmp_path, disc + _noise(shape, 3), (1200.0, 100.0))

    with pytest.raises(
        ValueError, match="holds no pixel on its complete annuli.*n_levels"
    ):
        resolve_cutout(opened, target, mode="auto")


# --- Warnings, settings, identity and record ---------------------------------------


def test_the_warnings_of_the_target_segment_are_kept(tmp_path) -> None:
    shape = (601, 601)
    rows, columns = np.indices(shape)
    blob = 8.0 * np.exp(-0.5 * ((columns - 300.0) ** 2 + (rows - 300.0) ** 2) / 4.0)
    opened, target = _inputs(tmp_path, blob + _noise(shape, 6), (309.0, 300.0))

    cutout = resolve_cutout(opened, target, mode="auto")

    assert not cutout.selection.on_source
    assert len(cutout.warnings) == 1
    assert "No source was detected at the target position" in cutout.warnings[0]
    assert cutout.record["target_segment"]["on_source"] is False


def _config(**cutout) -> dict:
    return {
        "object": {"name": "galaxy"},
        "data": {"image_path": "image.fits", "cutout": {"mode": "auto", **cutout}},
        "outputs": {"root": "runs"},
    }


def test_a_declared_initial_factor_has_no_effect_and_is_warned(tmp_path) -> None:
    opened, target = _inputs(tmp_path, _galaxy())

    plain = cutout_from_settings(read_preparation_settings(_config()), opened, target)
    declared = cutout_from_settings(
        read_preparation_settings(_config(initial_factor=4.0)), opened, target
    )

    assert plain.warnings == ()
    assert declared.bounds == plain.bounds
    assert declared.noise_radius == plain.noise_radius
    assert declared.warnings == (
        "data.cutout.initial_factor has no effect: the automatic cutout follows the "
        "light of the galaxy, measured once on the working region, with no "
        "preliminary cutout. Remove it from the configuration; a later version will "
        "refuse it.",
    )
    assert declared.record["without_effect"] == ["data.cutout.initial_factor"]
    assert declared.identity != plain.identity


def test_the_settings_of_a_configuration_reach_the_stage(tmp_path) -> None:
    opened, target = _inputs(tmp_path, _galaxy())
    config = _config(light_threshold=3.0, sky_margin=10.0)
    config["preparation"] = {
        "noise": {"light_threshold": 1.0},
        "detection": {"n_sigma": 4.0},
    }

    cutout = cutout_from_settings(read_preparation_settings(config), opened, target)
    direct = resolve_cutout(
        opened,
        target,
        mode="auto",
        cutout_threshold=3.0,
        noise_threshold=1.0,
        sky_margin=10.0,
        n_sigma=4.0,
    )

    assert cutout.identity == direct.identity
    assert cutout.record["settings"]["cutout_threshold"] == 3.0
    assert cutout.record["settings"]["noise_threshold"] == 1.0
    assert cutout.record["settings"]["sky_margin"] == 10.0
    assert cutout.record["detection"]["settings"]["n_sigma"] == 4.0


def test_the_identity_follows_the_settings_the_target_and_the_pixels(
    tmp_path,
) -> None:
    image = _galaxy()
    opened, target = _inputs(tmp_path, image)
    first = resolve_cutout(opened, target, mode="auto")
    other, other_target = _inputs(tmp_path, image + 0.01, name="other.fits")
    moved = resolve_target(opened, {"x": 300.0, "y": 300.0})

    assert resolve_cutout(opened, target, mode="auto").identity == first.identity
    assert (
        resolve_cutout(opened, target, mode="auto", noise_threshold=0.2).identity
        != first.identity
    )
    assert (
        resolve_cutout(opened, target, mode="auto", sky_margin=10.0).identity
        != first.identity
    )
    assert resolve_cutout(opened, moved, mode="auto").identity != first.identity
    assert resolve_cutout(other, other_target, mode="auto").identity != first.identity


def test_the_record_is_json_and_the_product_round_trips(tmp_path) -> None:
    opened, target = _inputs(tmp_path, _galaxy())
    cutout = resolve_cutout(opened, target, mode="auto")

    record = json.loads(json.dumps(dict(cutout.record)))

    assert record["stage"] == CUTOUT_STAGE
    assert record["mode"] == "auto"
    assert record["cutout"]["bounds"] == cutout.bounds
    assert record["radii"]["cutout"]["radius"] == cutout.cutout_radius
    assert record["radii"]["noise"]["radius"] == cutout.noise_radius
    assert record["noise_region"]["pixels"] == cutout.noise_pixels
    assert record["noise_region"]["minimum"] == MINIMUM_NOISE_PIXELS
    assert record["far_sky"]["consistent"] is True
    assert record["rule"] == rule()
    assert record["limitations"] == [FAR_SKY_LIMITATION, AUTOMATIC_CUTOUT_LIMITATION]
    assert record["image_identity"] == opened.image_identity
    assert record["target_identity"] == target.identity

    store = ProductStore(tmp_path / "products")
    store.save(cutout.product())
    loaded = store.load(CUTOUT_STAGE, identity=cutout.identity)

    assert json.loads(json.dumps(dict(loaded.record))) == record
    assert np.array_equal(loaded.images["segments"], cutout.detection.labels)
    assert np.array_equal(loaded.arrays["profile_fit"], cutout.fitted, equal_nan=True)
    assert np.array_equal(loaded.arrays["profile_complete"], cutout.profile.complete)


def test_the_stage_takes_the_products_of_the_opening_and_the_target(
    tmp_path,
) -> None:
    opened, target = _inputs(tmp_path, _galaxy())
    other, _ = _inputs(tmp_path, _galaxy(seed=9), name="other.fits")

    with pytest.raises(TypeError, match="OpenedObservation"):
        resolve_cutout(object(), target, mode="auto")

    with pytest.raises(TypeError, match="Target"):
        resolve_cutout(opened, {"x": 300.0, "y": 300.0}, mode="auto")

    with pytest.raises(ValueError, match="other pixels"):
        resolve_cutout(other, target, mode="auto")


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"mode": "half"}, "'auto' or 'fixed'"),
        ({"mode": "fixed"}, "needs data.cutout.size"),
        ({"mode": "fixed", "size": 0}, "size must be null or a positive integer"),
        ({"mode": "fixed", "size": 201.0}, "size must be null or a positive integer"),
        ({"mode": "auto", "min_size": True}, "min_size must be null"),
        ({"mode": "auto", "min_size": 301, "max_size": 201}, "must not exceed"),
        ({"mode": "auto", "sky_margin": -1.0}, "at least 0 pixels"),
        ({"mode": "auto", "sky_margin": math.nan}, "finite number"),
        ({"mode": "auto", "cutout_threshold": 0.0}, "greater than 0"),
        ({"mode": "auto", "noise_threshold": math.inf}, "finite number"),
        ({"mode": "auto", "noise_threshold": True}, "finite number"),
        ({"mode": "auto", "n_sigma": 0.0}, "n_sigma"),
        ({"mode": "auto", "n_pixels": 2.5}, "n_pixels"),
        ({"mode": "auto", "without_effect": ("data.cutout.size",)}, "must be among"),
        ({"mode": "auto", "without_effect": "data.cutout.initial_factor"}, "string"),
    ],
)
def test_invalid_settings_are_refused(tmp_path, settings, message) -> None:
    image = np.random.default_rng(0).normal(0.0, 1.0, (101, 101))
    opened, target = _inputs(tmp_path, image, (50.0, 50.0))

    with pytest.raises(ValueError, match=message):
        resolve_cutout(opened, target, **settings)


# --- The acceptance -----------------------------------------------------------------


def test_the_radii_meet_their_thresholds_and_do_not_follow_the_side(tmp_path) -> None:
    """A Sérsic galaxy of index 4 and axis ratio 0.67 on a sky gradient.

    The light of the galaxy at each measured radius lies within 0.67 and 1.5
    times its threshold, and the radii measured on stamps of 1013, 1201 and
    1451 pixels about the galaxy differ by at most 5 per cent.
    """
    big = 1451
    middle = (big - 1) / 2.0
    shape = (big, big)
    rows, columns = np.indices(shape, dtype=float)
    galaxy = _sersic(
        shape, middle + 0.27, middle - 0.13, re=50.0, n=4.0, q=0.67, theta=0.5
    )
    sky = 3.6e-4 * (columns - middle) - 1.6e-4 * (rows - middle)
    image = galaxy + sky + _noise(shape, 7, correlated=True)
    radii: dict[str, list[float]] = {"cutout": [], "noise": []}

    for side in (1451, 1201, 1013):
        start = (big - side) // 2
        window = (slice(start, start + side), slice(start, start + side))
        position = (middle + 0.27 - start, middle - 0.13 - start)
        opened, target = _inputs(
            tmp_path, image[window], position, name=f"stamp_{side}.fits"
        )

        cutout = resolve_cutout(opened, target, mode="auto")
        truth = elliptical_profile(
            galaxy[window], np.zeros((side, side), bool), cutout.ellipse
        )

        for name, radius, threshold in (
            ("cutout", cutout.cutout_radius, CUTOUT_LIGHT_THRESHOLD),
            ("noise", cutout.noise_radius, NOISE_LIGHT_THRESHOLD),
        ):
            index = int(np.searchsorted(truth.inner, radius, side="right")) - 1

            assert 0.67 * threshold <= truth.value[index] <= 1.5 * threshold
            radii[name].append(radius)

    for values in radii.values():
        assert (max(values) - min(values)) / float(np.median(values)) <= 0.05
