"""Where the sky begins: the profile of the galaxy against a far sky (D1).

These tests check the pieces of the rule against answers they build: the
ellipse of the annuli, the completeness of an annulus against an unbounded
grid and against areas without data, the profile of a known radial function,
the reference plane and its checks, the non-increasing fit and the radius it
reads.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import gammaincinv

from baymorph.preparation.profile import (
    BIN_GROWTH,
    COMPLETE_FRACTION,
    LINEAR_BINS_UNTIL,
    MINIMUM_REFERENCE_PIXELS,
    Ellipse,
    Profile,
    annulus_edges,
    complete_radius,
    ellipse_of_segment,
    elliptical_profile,
    grown,
    monotone_fit,
    monotone_radius,
    nonincreasing_fit,
    reference_sky,
)


def _sersic(shape, ellipse: Ellipse, level: float, re: float, n: float) -> np.ndarray:
    """Return a noise-free Sérsic profile of ``level`` at ``re``."""
    radius, _ = ellipse.coordinates(shape)
    b = gammaincinv(2.0 * n, 0.5)

    return level * np.exp(-b * ((radius / re) ** (1.0 / n) - 1.0))


# --- The ellipse ----------------------------------------------------------------


def test_the_radius_is_the_semi_major_axis_through_the_pixel() -> None:
    ellipse = Ellipse(x0=10.0, y0=20.0, q=0.5, theta=math.pi / 2)
    radius, angle = ellipse.coordinates((41, 31))

    # The major axis runs along y, the minor along x.
    assert radius[30, 10] == pytest.approx(10.0)
    assert radius[20, 15] == pytest.approx(10.0)
    assert angle[30, 10] == pytest.approx(0.0, abs=1e-12)
    assert ellipse.half_extent(10.0) == pytest.approx((5.0, 10.0))


def test_the_inscribed_radius_touches_the_nearest_edge() -> None:
    ellipse = Ellipse(x0=50.0, y0=30.0, q=0.5, theta=0.0)

    # Along x the ellipse of radius r spans r; along y, r / 2.
    assert ellipse.inscribed_radius((61, 201)) == pytest.approx(50.5)
    assert ellipse.inscribed_radius((41, 201)) == pytest.approx(2 * 10.5)


@pytest.mark.parametrize("q", [0.0, 1.2, math.nan])
def test_an_invalid_ellipse_is_refused(q) -> None:
    with pytest.raises(ValueError):
        Ellipse(x0=0.0, y0=0.0, q=q, theta=0.0)


def test_the_ellipse_of_a_segment_follows_its_light() -> None:
    truth = Ellipse(x0=100.3, y0=99.6, q=0.6, theta=0.4)
    image = _sersic((201, 201), truth, 10.0, 15.0, 1.0)
    segment = image > 0.5

    ellipse = ellipse_of_segment(image, segment, x0=100.3, y0=99.6)

    assert ellipse.q == pytest.approx(0.6, abs=0.02)
    assert math.cos(2 * (ellipse.theta - 0.4)) == pytest.approx(1.0, abs=2e-3)
    assert (ellipse.x0, ellipse.y0) == (100.3, 99.6)

    with pytest.raises(ValueError, match="three pixels"):
        ellipse_of_segment(image, np.zeros_like(segment), x0=0.0, y0=0.0)


# --- The annuli and the profile ------------------------------------------------


def test_the_annuli_are_one_pixel_wide_then_grow() -> None:
    edges = annulus_edges(200.0)

    assert np.array_equal(edges[: LINEAR_BINS_UNTIL + 1], np.arange(51))
    assert edges[LINEAR_BINS_UNTIL + 1] == pytest.approx(50.0 * BIN_GROWTH)
    assert edges[-1] >= 200.0 > edges[-2]


def test_only_complete_annuli_count() -> None:
    ellipse = Ellipse(x0=60.2, y0=60.4, q=0.7, theta=0.3)
    shape = (121, 121)
    profile = elliptical_profile(np.zeros(shape), np.zeros(shape, bool), ellipse)
    counted = np.flatnonzero(profile.complete)
    inscribed = ellipse.inscribed_radius(shape)

    assert counted[0] == 0 and np.all(np.diff(counted) == 1)
    assert np.all(
        profile.in_image[counted] >= COMPLETE_FRACTION * profile.unbounded[counted]
    )
    # Every annulus inside the inscribed ellipse lies wholly in the image.
    inside = profile.outer <= inscribed
    assert np.array_equal(profile.in_image[inside], profile.unbounded[inside])
    assert profile.complete_radius >= inscribed
    assert profile.complete_radius < 1.1 * inscribed + 3.0
    beyond = counted[-1] + 1
    assert profile.in_image[beyond] < COMPLETE_FRACTION * profile.unbounded[beyond]
    assert complete_radius(shape, ellipse) == profile.complete_radius


def test_pixels_without_data_count_as_outside_the_image() -> None:
    ellipse = Ellipse(x0=100.0, y0=100.0, q=1.0, theta=0.0)
    shape = (201, 201)
    image = np.ones(shape)
    no_data = np.zeros(shape, bool)
    # A strip without data, as a mosaic leaves, 60 pixels right of the centre.
    no_data[:, 160:175] = True
    image[no_data] = np.nan

    profile = elliptical_profile(image, np.zeros(shape, bool), ellipse, no_data=no_data)
    whole = elliptical_profile(np.ones(shape), np.zeros(shape, bool), ellipse)

    # The first annulus that crosses the strip stops the profile, as an edge
    # of the image would.
    edges = annulus_edges(100.0)
    crossing = int(np.searchsorted(edges, 60.0, side="right")) - 1

    assert profile.complete_radius == pytest.approx(edges[crossing])
    assert whole.complete_radius > 90.0
    assert complete_radius(shape, ellipse, no_data=no_data) == profile.complete_radius
    assert np.allclose(profile.value[profile.complete], 1.0)

    # A few pixels without data leave an annulus complete.
    sparse = np.zeros(shape, bool)
    sparse[100, 140] = sparse[60, 100] = True

    assert complete_radius(shape, ellipse, no_data=sparse) == whole.complete_radius

    with pytest.raises(ValueError, match="without data"):
        complete_radius(shape, ellipse, no_data=np.zeros((5, 5), bool))


def test_the_profile_of_a_radial_function_is_that_function() -> None:
    ellipse = Ellipse(x0=150.3, y0=149.8, q=0.8, theta=-0.6)
    shape = (301, 301)
    radius, _ = ellipse.coordinates(shape)
    plane = (2.0, 0.01, -0.02)
    image = (
        np.exp(-radius / 40.0)
        + plane[0]
        + plane[1] * (np.indices(shape)[1] - ellipse.x0)
        + plane[2] * (np.indices(shape)[0] - ellipse.y0)
    )

    profile = elliptical_profile(image, np.zeros(shape, bool), ellipse, plane=plane)
    middle = 0.5 * (profile.inner + profile.outer)
    counted = profile.complete & (profile.inner >= 5.0)

    assert np.allclose(
        profile.value[counted], np.exp(-middle[counted] / 40.0), rtol=2e-3
    )


def test_masked_pixels_never_enter_the_profile() -> None:
    ellipse = Ellipse(x0=50.0, y0=50.0, q=1.0, theta=0.0)
    shape = (101, 101)
    image = np.ones(shape)
    masked = np.zeros(shape, bool)
    image[:, 60:] = 1.0e6
    masked[:, 60:] = True
    image[0, 0] = np.nan
    masked[0, 0] = True

    profile = elliptical_profile(image, masked, ellipse)

    assert np.allclose(profile.value[profile.complete & (profile.used > 0)], 1.0)

    masked[0, 0] = False

    with pytest.raises(ValueError, match="finite"):
        elliptical_profile(image, masked, ellipse)


def test_the_error_of_an_annulus_follows_its_scatter() -> None:
    ellipse = Ellipse(x0=200.0, y0=200.0, q=1.0, theta=0.0)
    shape = (401, 401)
    errors = []

    for seed in range(20):
        image = np.random.default_rng(seed).normal(0.0, 1.0, shape)
        profile = elliptical_profile(image, np.zeros(shape, bool), ellipse)
        wide = profile.complete & (profile.inner >= 100.0)
        errors.append(profile.error[wide] * np.sqrt(profile.used[wide]))

    # For white noise the error of a mean is 1 / sqrt(n).
    assert np.median(np.concatenate(errors)) == pytest.approx(1.0, rel=0.1)


# --- The reference sky -----------------------------------------------------------


def _sky(shape, plane, ellipse, seed=0):
    rows, columns = np.indices(shape, dtype=float)
    noise = np.random.default_rng(seed).normal(0.0, 1.0, shape)

    return (
        plane[0]
        + plane[1] * (columns - ellipse.x0)
        + plane[2] * (rows - ellipse.y0)
        + noise
    )


def test_the_reference_plane_is_recovered_beyond_its_radius() -> None:
    ellipse = Ellipse(x0=250.4, y0=249.7, q=0.85, theta=0.5)
    shape = (501, 501)
    plane = (3.0, 2.0e-3, -1.0e-3)
    image = _sky(shape, plane, ellipse) + _sersic(shape, ellipse, 50.0, 10.0, 1.0)
    masked = np.zeros(shape, bool)
    masked[400:420, 30:60] = True
    image[400:420, 30:60] = 1.0e4

    reference = reference_sky(image, masked, ellipse, radius=200.0)

    assert reference.plane[0] == pytest.approx(3.0, abs=0.02)
    assert reference.plane[1] == pytest.approx(2.0e-3, abs=2e-4)
    assert reference.plane[2] == pytest.approx(-1.0e-3, abs=2e-4)
    assert reference.scale == pytest.approx(1.0, rel=0.02)
    assert 0.0 < reference.error < 0.02
    assert reference.consistent
    assert reference.as_dict()["consistent"] is True


def test_galaxy_light_on_the_reference_sky_breaks_the_agreement() -> None:
    ellipse = Ellipse(x0=250.0, y0=250.0, q=1.0, theta=0.0)
    shape = (501, 501)
    image = _sky(shape, (0.0, 0.0, 0.0), ellipse, seed=4)
    image += _sersic(shape, ellipse, 30.0, 60.0, 4.0)

    reference = reference_sky(image, np.zeros(shape, bool), ellipse, radius=200.0)

    assert reference.inner_minus_outer > 0.0
    assert not reference.consistent


def test_a_reference_sky_needs_enough_pixels() -> None:
    ellipse = Ellipse(x0=50.0, y0=50.0, q=1.0, theta=0.0)

    with pytest.raises(ValueError, match=str(MINIMUM_REFERENCE_PIXELS)):
        reference_sky(
            np.zeros((101, 101)), np.zeros((101, 101), bool), ellipse, radius=20
        )


# --- The radius ------------------------------------------------------------------


def _profile(values, errors=None) -> Profile:
    count = len(values)
    edges = np.arange(count + 1, dtype=float) * 10.0

    return Profile(
        inner=edges[:-1],
        outer=edges[1:],
        value=np.asarray(values, dtype=float),
        error=np.full(count, 0.05) if errors is None else np.asarray(errors, float),
        used=np.full(count, 100),
        in_image=np.full(count, 100),
        unbounded=np.full(count, 100),
        complete=np.ones(count, dtype=bool),
    )


def test_the_monotone_rule_reads_the_non_increasing_fit() -> None:
    values = [5.0, 2.0, 0.8, 0.25, 0.35, 0.15, 0.1, 0.05]
    profile = _profile(values)

    # The bump 0.25, 0.35 pools to 0.30, which is not below 0.3.
    assert monotone_radius(profile, values, 0.3) == 50.0
    assert monotone_radius(profile, values, 0.31) == 30.0
    assert monotone_radius(profile, values, 0.04) is None
    assert np.allclose(
        monotone_fit(profile, values), [5.0, 2.0, 0.8, 0.3, 0.3, 0.15, 0.1, 0.05]
    )


def test_the_fit_weights_the_annuli_by_their_errors() -> None:
    values = [1.0, 2.0]

    # The precise second annulus pulls the pooled value towards it.
    fitted = monotone_fit(_profile(values, errors=[1.0, 0.5]), values)

    assert np.allclose(fitted, [1.8, 1.8])

    # An annulus without an error takes the median weight.
    fitted = monotone_fit(_profile(values, errors=[np.nan, 0.5]), values)

    assert np.allclose(fitted, [1.5, 1.5])

    with pytest.raises(ValueError, match="annuli"):
        monotone_fit(_profile(values), [1.0, 2.0, 3.0])


def test_the_non_increasing_fit_pools_violators() -> None:
    fitted = nonincreasing_fit([3.0, 1.0, 2.0, 0.5, 0.6, 0.4], np.ones(6))

    assert np.allclose(fitted, [3.0, 1.5, 1.5, 0.55, 0.55, 0.4])
    assert np.allclose(nonincreasing_fit([1.0, 2.0], [3.0, 1.0]), [1.25, 1.25])
    assert np.all(
        np.diff(nonincreasing_fit(np.random.default_rng(1).random(50), np.ones(50)))
        <= 0
    )


def test_incomplete_annuli_never_enter_the_rules() -> None:
    values = [5.0, 2.0, 0.2, 0.1, 9.0]
    profile = _profile(values)
    complete = profile.complete.copy()
    complete[-1] = False
    profile = Profile(**{**profile.__dict__, "complete": complete})

    assert monotone_radius(profile, values, 0.3) == 20.0
    assert np.isnan(monotone_fit(profile, values)[-1])


# --- Masks ---------------------------------------------------------------------


def test_a_mask_grows_by_its_distance() -> None:
    mask = np.zeros((41, 41), bool)
    mask[20, 20] = True

    grown_mask = grown(mask, 5)

    assert grown_mask[20, 25] and grown_mask[23, 24]
    assert not grown_mask[20, 26] and not grown_mask[24, 24]
    assert not np.any(grown(np.zeros((5, 5), bool), 5))
