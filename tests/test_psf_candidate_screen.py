"""Tests for the point-source screen of the empirical-PSF candidates."""

from __future__ import annotations

import json
import math
import pickle

import numpy as np
import pytest
from scipy.special import erf

from baymorph.pipeline.checkpoint_signing import canonical_json_sha256
from baymorph.pipeline import psf_candidate_screen
from baymorph.pipeline.psf_candidate_screen import (
    PSF_CANDIDATE_SCREEN_SCHEMA,
    PSF_CANDIDATE_SCREEN_SCHEMA_VERSION,
    SIZE_SEQUENCE_V1,
    SIZE_SEQUENCE_V1_SHA256,
    CandidateSize,
    PointSourceScreenError,
    find_stellar_sequence,
    measure_candidate_size,
    point_source_screen_marker,
    point_source_screen_rule,
    recorded_screen_rule,
    required_core_count,
    screen_change_message,
    screen_psf_candidates,
    screen_record_lines,
    screen_stop_message,
    screen_summary,
)
from baymorph.pipeline.psf_candidates import PSFCandidate
from baymorph.pipeline.psf_construction import estimate_psf_core_fwhm

SHAPE = (61, 61)


def _gaussian(x0, y0, sigma, *, flux=1000.0, shape=SHAPE, integrated=True):
    """Return a Gaussian source, integrated over the pixels or sampled."""
    ny, nx = shape

    if not integrated:
        yy, xx = np.indices(shape, dtype=float)
        image = np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / sigma**2)

        return flux * image / image.sum()

    edges_x = np.arange(nx + 1) - 0.5
    edges_y = np.arange(ny + 1) - 0.5
    fx = np.diff(0.5 * (1.0 + erf((edges_x - x0) / (np.sqrt(2.0) * sigma))))
    fy = np.diff(0.5 * (1.0 + erf((edges_y - y0) / (np.sqrt(2.0) * sigma))))

    return flux * np.outer(fy, fx)


def _moffat(x0, y0, fwhm, *, beta=2.5, flux=1000.0, shape=SHAPE, sub=11):
    """Return a Moffat source integrated over the pixels."""
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    offsets = (np.arange(sub) + 0.5) / sub - 0.5
    yy, xx = np.indices(shape, dtype=float)
    image = np.zeros(shape)

    for dy in offsets:
        for dx in offsets:
            image += (
                1.0 + ((xx + dx - x0) ** 2 + (yy + dy - y0) ** 2) / alpha**2
            ) ** (-beta)

    return flux * image / image.sum()


def _candidate(x, y, *, half_size=12, sigma=1.2, label=1):
    """Return a pool candidate at a position."""
    return PSFCandidate(
        x=float(x),
        y=float(y),
        label=label,
        half_size=half_size,
        semimajor_sigma=sigma,
        semiminor_sigma=sigma,
    )


# --------------------------------------------------------------- measurement


@pytest.mark.parametrize("phase", [(0.0, 0.0), (0.3, -0.2), (0.5, 0.5)])
def test_the_size_of_a_gaussian_is_its_width_at_every_phase(phase):
    """The adaptive size of a Gaussian source is its own width."""
    x0, y0 = 30.0 + phase[0], 30.0 + phase[1]
    sampled = measure_candidate_size(
        _gaussian(x0, y0, 1.2, integrated=False), _candidate(x0, y0)
    )
    # Integration over a pixel adds the variance of a unit box, 1/12.
    integrated = measure_candidate_size(_gaussian(x0, y0, 1.2), _candidate(x0, y0))

    assert sampled.status == integrated.status == "measured"
    assert sampled.size == pytest.approx(1.2, rel=0.005)
    assert integrated.size == pytest.approx(math.sqrt(1.2**2 + 1 / 12), rel=0.005)
    assert abs(integrated.e1) < 1.0e-4
    assert abs(integrated.e2) < 1.0e-4


@pytest.mark.parametrize("fwhm", [2.3, 2.7, 3.5])
def test_the_size_of_a_moffat_star_is_about_half_its_core_fwhm(fwhm):
    """For a Moffat profile of index 2.5, s / FWHM lies in [0.47, 0.49]."""
    core = estimate_psf_core_fwhm(_moffat(30.0, 30.0, fwhm))

    for phase in ((0.0, 0.0), (0.3, -0.2), (0.5, 0.5)):
        x0, y0 = 30.0 + phase[0], 30.0 + phase[1]
        size = measure_candidate_size(_moffat(x0, y0, fwhm), _candidate(x0, y0))

        assert size.status == "measured"
        assert 0.47 <= size.size / core <= 0.49


def test_one_and_two_pixel_artifacts_collapse():
    """A hot pixel and a pair of pixels are sharper than any point source."""
    hot = np.zeros(SHAPE)
    hot[30, 30] = 100.0
    pair = np.zeros(SHAPE)
    pair[30, 30] = 100.0
    pair[30, 31] = 80.0

    assert measure_candidate_size(hot, _candidate(30, 30)).status == "collapsed"
    assert measure_candidate_size(pair, _candidate(30.4, 30)).status == "collapsed"


def test_a_two_by_two_block_measures_above_the_floor():
    """A raw 2 x 2 block has s = 1/sqrt(2) px: above the floor of 0.60 px.

    It is not rejected as sharper than a point source, but it lies below a
    sequence of stars, as the decision test below shows.
    """
    block = np.zeros(SHAPE)
    block[30:32, 30:32] = 100.0

    size = measure_candidate_size(block, _candidate(30.5, 30.5))

    assert size.status == "measured"
    assert size.size == pytest.approx(1.0 / math.sqrt(2.0), rel=1.0e-9)
    assert size.size > SIZE_SEQUENCE_V1["size_floor_pixels"]


def test_a_large_galaxy_exceeds_the_window_of_its_cutout():
    """An exponential of effective radius 8 px outgrows a half-size of 12."""
    yy, xx = np.indices(SHAPE, dtype=float)
    radius = np.hypot(xx - 30.0, yy - 30.0)
    image = 100.0 * np.exp(-1.678 * (radius / 8.0 - 1.0))

    size = measure_candidate_size(image, _candidate(30, 30, sigma=5.0))

    assert size.status == "exceeds_window"


def test_an_empty_stamp_has_no_flux_and_an_edge_cutout_no_size():
    """Neither an empty stamp nor a cutout leaving the image has a size."""
    empty = measure_candidate_size(np.zeros(SHAPE), _candidate(30, 30))
    outside = measure_candidate_size(np.zeros(SHAPE), _candidate(5, 30))

    assert empty.status == "no_flux"
    assert outside.status == "outside_image"

    sequence = find_stellar_sequence([empty, outside], min_peak_snr=30.0)

    assert sequence.reasons == ("size_not_measurable", "size_not_measurable")


def test_the_size_does_not_grow_with_the_flux():
    """The same star at peak S/N 50 and 2000 has one size, within the noise."""
    rng = np.random.default_rng(3)
    star = _moffat(30.3, 29.8, 2.7)
    sizes = []

    for peak_snr in (50.0, 2000.0):
        image = star * peak_snr / star.max() + rng.normal(0.0, 1.0, SHAPE)
        sizes.append(measure_candidate_size(image, _candidate(30.3, 29.8)))

    faint, bright = sizes
    sigma = math.hypot(1.5 / faint.snr_w, 1.5 / bright.snr_w)

    assert faint.status == bright.status == "measured"
    assert abs(math.log(faint.size / bright.size)) < 3.0 * sigma


# ------------------------------------------------------------------ decision


def _star(u, *, snr_w=500.0, rho4=2.0, peak_snr=100.0):
    """Return a measured candidate of size exp(u)."""
    return CandidateSize(
        status="measured",
        size=float(np.exp(u)),
        rho4=rho4,
        e1=0.0,
        e2=0.0,
        snr_w=snr_w,
        peak_snr=peak_snr,
    )


U0 = math.log(1.3)


def _stars(n, *, spread=0.004, rho4=None):
    """Return n stars about U0 within +-spread, with optional rho4 values."""
    offsets = np.linspace(-spread, spread, n)
    values = [2.0] * n if rho4 is None else rho4

    return [_star(U0 + d, rho4=r) for d, r in zip(offsets, values, strict=True)]


def _galaxies(n=30):
    """Return n larger sources spread above the stars, never six together."""
    return [_star(U0 + 0.2 + 0.045 * k, rho4=1.6) for k in range(n)]


def test_the_members_are_exactly_the_stars():
    """Twenty stars are found among thirty larger sources."""
    sizes = _stars(20) + _galaxies()
    sequence = find_stellar_sequence(sizes, min_peak_snr=30.0)

    assert sequence.outcome == "found"
    assert sequence.member == (True,) * 20 + (False,) * 30
    assert set(sequence.reasons[20:]) == {"larger_than_stellar_sequence"}
    assert sequence.sequence["n_core"] == 20
    assert sequence.sequence["size_pix"] == pytest.approx(1.3, rel=0.005)


def test_sources_below_the_floor_or_the_signal_to_noise_are_not_searched():
    """Sharp sources and faint sources never enter the search."""
    sizes = [
        *_stars(10),
        CandidateSize(status="collapsed"),
        _star(math.log(0.55)),
        _star(U0, peak_snr=10.0),
        CandidateSize(status="exceeds_window", peak_snr=200.0),
    ]
    sequence = find_stellar_sequence(sizes, min_peak_snr=30.0)

    assert sequence.outcome == "found"
    assert sequence.reasons[10:] == (
        "sharper_than_point_source",
        "sharper_than_point_source",
        "peak_snr_below_threshold",
        "larger_than_window",
    )
    assert sequence.n_search == 10


def test_a_two_by_two_block_is_smaller_than_a_stellar_sequence():
    """A block of 1/sqrt(2) px lies below a sequence of stars at 1.3 px."""
    sizes = _stars(12) + [_star(math.log(1.0 / math.sqrt(2.0)), rho4=1.0)]
    sequence = find_stellar_sequence(sizes, min_peak_snr=30.0)

    assert sequence.outcome == "found"
    assert sequence.reasons[-1] == "smaller_than_stellar_sequence"


def test_too_few_measurable_candidates_stop_the_screen():
    sequence = find_stellar_sequence(_stars(5), min_peak_snr=30.0)

    assert sequence.outcome == "too_few"
    assert "only 5 of the 5 candidates" in sequence.finding


def test_no_dense_group_stops_the_screen():
    sizes = [_star(U0 + 0.06 * k) for k in range(10)]
    sequence = find_stellar_sequence(sizes, min_peak_snr=30.0)

    assert sequence.outcome == "no_sequence"
    assert sequence.finding.startswith("no 6 candidates share one size")


def test_a_group_that_does_not_stand_out_from_its_flanks_stops_the_screen():
    """Eight values against six in the adjacent interval arise by chance."""
    flank = [_star(U0 + 0.07 + 0.014 * k, rho4=1.9) for k in range(6)]
    stars = _stars(8, spread=0.01)
    sequence = find_stellar_sequence(stars + flank, min_peak_snr=30.0)

    assert sequence.outcome == "no_sequence"
    assert sequence.detail == "not_standing_out"
    assert "does not stand out from its neighbours" in sequence.finding
    assert "6 candidates are 5 to 16 per cent larger" in sequence.finding
    assert "against 6, the group needs 16." in sequence.finding
    assert sequence.sequence["p_tail"] > 1.0e-3
    assert sequence.sequence["n_required"] == required_core_count(6) == 16


@pytest.mark.parametrize(
    ("n_flank", "required"), [(0, 6), (1, 6), (2, 9), (3, 11), (4, 12), (6, 16)]
)
def test_the_count_a_group_needs_to_stand_out(n_flank, required):
    """The smallest count of tail probability at most 1e-3, at least six."""
    assert required_core_count(n_flank) == required


@pytest.mark.parametrize(
    ("step", "tau_hat", "stops"), [(0.007, 0.0311, True), (0.0065, 0.0289, False)]
)
def test_a_group_wider_than_the_stars_stops_the_screen(step, tau_hat, stops):
    """A robust scatter of ln s above 0.03 is too wide for one PSF."""
    sizes = [_star(U0 + k * step) for k in range(-5, 6)]
    sequence = find_stellar_sequence(sizes, min_peak_snr=30.0)

    assert sequence.sequence["n_core"] == 11
    assert sequence.sequence["tau_hat"] == pytest.approx(tau_hat, abs=1.0e-4)
    assert (sequence.outcome == "no_sequence") is stops

    if stops:
        assert sequence.detail == "too_wide"
        assert "is too wide to be stars of one PSF" in sequence.finding


@pytest.mark.parametrize(("deviation", "stops"), [(3.6, True), (3.4, False)])
def test_a_group_without_one_profile_stops_the_screen(deviation, stops):
    """The fourth moment of the stars scatters at most 3.5 times its noise."""
    # With snr_w = 1e6, the noise of rho4 is its floor, 0.005.
    delta = deviation * 0.005 / 1.4826
    rho4 = [2.0 - delta] * 10 + [2.0 + delta] * 10
    sizes = [
        _star(U0 + d, snr_w=1.0e6, rho4=r)
        for d, r in zip(np.linspace(-0.004, 0.004, 20), rho4, strict=True)
    ]
    sequence = find_stellar_sequence(sizes, min_peak_snr=30.0)

    assert (sequence.outcome == "no_sequence") is stops
    assert sequence.sequence["rho4_z"] == pytest.approx(deviation, rel=1.0e-3)

    if stops:
        assert sequence.detail == "profile"
        assert "does not share one profile" in sequence.finding
        # A coarsely sampled PSF does this too, not only compact galaxies.
        assert "position within a pixel" in sequence.finding


@pytest.mark.parametrize(("n_rival", "stops"), [(10, True), (9, False)])
def test_a_rival_group_half_as_large_stops_the_screen(n_rival, stops):
    """Two narrow groups of sizes could each be the stars."""
    rival = [_star(U0 + 0.5 + d) for d in np.linspace(-0.004, 0.004, n_rival)]
    sequence = find_stellar_sequence(_stars(20) + rival, min_peak_snr=30.0)

    if stops:
        assert sequence.outcome == "ambiguous"
        assert sequence.sequence["rival"]["n_core"] == n_rival
    else:
        assert sequence.outcome == "found"
        assert sum(sequence.member) == 20


@pytest.mark.parametrize(("n_smaller", "stops"), [(3, True), (2, False)])
def test_sources_smaller_than_the_group_stop_the_screen(n_smaller, stops):
    """More than max(2, 0.2 members) smaller sources: not the smallest."""
    smaller = [_star(U0 - 0.3 - 0.06 * k, rho4=1.8) for k in range(n_smaller)]
    sequence = find_stellar_sequence(_stars(10) + smaller, min_peak_snr=30.0)

    assert sequence.outcome == ("not_smallest" if stops else "found")
    assert sequence.reasons.count("smaller_than_stellar_sequence") == n_smaller

    if stops:
        assert sequence.detail == "smaller_sources"
        assert "more than the 2 allowed" in sequence.finding


def test_too_few_members_stop_the_screen():
    """Two of seven core values have another profile, leaving five members."""
    rho4 = [2.0] * 5 + [2.5, 2.5]
    sequence = find_stellar_sequence(_stars(7, rho4=rho4), min_peak_snr=30.0)

    assert sequence.outcome == "too_few_members"
    assert sequence.reasons.count("profile_unlike_stellar_sequence") == 2
    assert sum(sequence.member) == 5


def test_the_record_of_a_late_stop_counts_its_members(monkeypatch):
    """A stop after membership records its members, none of them kept."""
    sizes = iter(_stars(7, rho4=[2.0] * 5 + [2.5, 2.5]))
    monkeypatch.setattr(
        psf_candidate_screen, "measure_candidate_size", lambda image, c: next(sizes)
    )
    pool = [_candidate(30, 30, label=k) for k in range(7)]

    with pytest.raises(PointSourceScreenError) as error:
        screen_psf_candidates(
            np.zeros(SHAPE), pool, min_peak_snr=30.0, target_n_candidates=30
        )

    record = error.value.record

    assert error.value.code == "too_few_members"
    assert record["detail"] == "too_few_members"
    assert record["n_members"] == record["sequence"]["n_members"] == 5
    assert record["n_kept"] == 0
    assert not any(c["kept"] for c in record["candidates"])


# ---------------------------------------------------------- frozen constants


def test_the_constants_of_size_sequence_v1_are_frozen():
    """A changed constant needs a new rule name, not a silent edit."""
    assert dict(SIZE_SEQUENCE_V1) == {
        "border_width_pixels": 3,
        "start_size_floor_pixels": 0.5,
        "max_iterations": 200,
        "moment_tolerance": 1.0e-7,
        "centre_tolerance_pixels": 1.0e-5,
        "collapse_eigenvalue_pixels2": 0.04,
        "window_size_divisor": 3.0,
        "window_shift_divisor": 2.0,
        "size_floor_pixels": 0.60,
        "window_ln_size": 0.05,
        "flank_multiple": 3.0,
        "flank_mean_floor": 1.0,
        "min_members": 6,
        "tail_probability_max": 1.0e-3,
        "core_scatter_max": 0.03,
        "mode_steps_max": 100,
        "mode_tolerance": 1.0e-6,
        "mode_merge_fraction": 0.5,
        "mad_to_sigma": 1.4826,
        "ambiguity_ratio": 0.5,
        "ambiguity_separation_windows": 2.0,
        "member_widths": 3.0,
        "width_floor": 0.01,
        "size_error_scale": 1.5,
        "rho4_error_scale": 2.5,
        "rho4_noise_floor": 0.005,
        "rho4_z_max": 3.5,
        "rho4_member_floor": 0.05,
        "smaller_max_count": 2,
        "smaller_max_fraction": 0.2,
    }
    assert SIZE_SEQUENCE_V1_SHA256 == (
        "4c0b6561102bfdeb453bb2ca479f2a359ab562bf6ab9cc621464ab4414cabcbc"
    )
    assert canonical_json_sha256(dict(SIZE_SEQUENCE_V1)) == SIZE_SEQUENCE_V1_SHA256

    with pytest.raises(TypeError):
        SIZE_SEQUENCE_V1["window_ln_size"] = 0.06


def test_the_rule_names_and_markers():
    assert point_source_screen_rule({}) == "size_sequence_v1"
    assert point_source_screen_rule({"point_source_screen": "none"}) == "none"
    assert point_source_screen_marker("none") is None
    assert point_source_screen_marker("size_sequence_v1") == {
        "rule": "size_sequence_v1",
        "constants_sha256": SIZE_SEQUENCE_V1_SHA256,
    }

    for value in ("v1", "size_sequence", True, None):
        with pytest.raises(ValueError, match="point_source_screen must be one of"):
            point_source_screen_rule({"point_source_screen": value})

    marker = point_source_screen_marker("size_sequence_v1")

    assert recorded_screen_rule({}) == "none"
    assert recorded_screen_rule({"point_source_screen": marker}) == "size_sequence_v1"
    assert (
        recorded_screen_rule(
            {"point_source_screen": {**marker, "constants_sha256": "0" * 64}}
        )
        is None
    )


def test_products_of_another_screen_are_refused_without_an_overwrite():
    """The refusal names the declaration that reuses the products."""
    earlier = screen_change_message(
        "PSF candidates", recorded="none", current="size_sequence_v1"
    )
    later = screen_change_message(
        "PSF candidates", recorded="size_sequence_v1", current="none"
    )

    assert "as BayMorph does by default" in earlier
    assert "point_source_screen: none to reuse them unchanged" in earlier
    assert "point_source_screen: size_sequence_v1 to reuse them" in later

    for message in (earlier, later):
        assert "new output namespace" in message
        assert "overwrite" not in message


# -------------------------------------------------------------- whole screen


def _field(rng, *, stars=12, galaxies=4, hot_pixels=3):
    """Return an image of isolated sources on a grid, and the pool."""
    image = rng.normal(0.0, 1.0, (260, 260))
    positions = [
        (
            30.0 + 40.0 * (k % 6) + rng.uniform(-0.4, 0.4),
            30.0 + 40.0 * (k // 6) + rng.uniform(-0.4, 0.4),
        )
        for k in range(stars + galaxies + hot_pixels)
    ]
    pool = []

    for index, (x, y) in enumerate(positions):
        if index < stars:
            image += _gaussian(x, y, 1.15, flux=2.0e3 * (1 + index), shape=image.shape)
        elif index < stars + galaxies:
            image += _gaussian(x, y, 2.4, flux=1.0e4, shape=image.shape)
        else:
            x, y = round(x), round(y)
            image[int(y), int(x)] += 300.0

        pool.append(_candidate(x, y, label=index + 1))

    return image, pool


def test_the_screen_keeps_the_stars_and_records_every_candidate():
    """Members keep the pool order; the record gives whole-image positions."""
    image, pool = _field(np.random.default_rng(5))
    result = screen_psf_candidates(
        image, pool, min_peak_snr=30.0, target_n_candidates=10, origin=(100, 200)
    )
    record = json.loads(json.dumps(result.record))

    assert [c.label for c in result.members] == list(range(1, 13))
    assert [c.label for c in result.kept] == list(range(1, 11))
    assert record["schema"] == PSF_CANDIDATE_SCREEN_SCHEMA
    assert record["schema_version"] == PSF_CANDIDATE_SCREEN_SCHEMA_VERSION
    assert record["rule"] == "size_sequence_v1"
    assert record["constants_sha256"] == SIZE_SEQUENCE_V1_SHA256
    assert record["outcome"] == "found"
    assert (record["n_pool"], record["n_members"], record["n_kept"]) == (19, 12, 10)
    assert [c["kept"] for c in record["candidates"]] == [True] * 10 + [False] * 9
    assert [c["reason"] for c in record["candidates"][12:]] == (
        ["larger_than_stellar_sequence"] * 4 + ["sharper_than_point_source"] * 3
    )
    assert [(c["x"], c["y"]) for c in record["candidates"]] == [
        (c.x + 100, c.y + 200) for c in pool
    ]
    assert sum(record["counts"].values()) == 19

    summary = screen_summary(record, pixel_scale=0.04)

    assert summary["n_kept"] == 10
    assert summary["sequence_size_arcsec"] == pytest.approx(
        0.04 * summary["sequence_size_pix"]
    )
    assert summary["rejected"] == {
        "larger_than_stellar_sequence": 4,
        "sharper_than_point_source": 3,
    }


def test_a_field_without_stars_stops_with_the_record_of_every_candidate():
    """No stellar sequence: the error carries the verdict of each candidate."""
    image, pool = _field(np.random.default_rng(6), stars=0, galaxies=4)

    with pytest.raises(PointSourceScreenError) as error:
        screen_psf_candidates(image, pool, min_peak_snr=30.0, target_n_candidates=30)

    message = str(error.value)

    assert error.value.code == "too_few"
    assert message.startswith("PSF candidate screen size_sequence_v1: only 4 of")
    assert "No PSF candidate was saved" in message
    assert "each of the 7 candidates are in the record of this error" in message
    assert "A larger part of the image, when there is one, offers more" in message
    assert "point_source_screen: none" in message
    assert len(error.value.record["candidates"]) == 7
    assert all(c["reason"] for c in error.value.record["candidates"])
    assert not any(c["kept"] for c in error.value.record["candidates"])

    copy = pickle.loads(pickle.dumps(error.value))

    assert (copy.code, copy.record, copy.finding) == (
        error.value.code,
        error.value.record,
        error.value.finding,
    )


# ----------------------------------------------------------- stop messages


def _stop_record(sizes):
    """Return the record of a stop of the decision on measured sizes."""
    sequence = find_stellar_sequence(sizes, min_peak_snr=30.0)

    assert sequence.outcome != "found"

    return {
        "rule": "size_sequence_v1",
        "outcome": sequence.outcome,
        "finding": sequence.finding,
        "detail": sequence.detail,
        "n_pool": len(sizes),
        "n_search": sequence.n_search,
        "modes": list(sequence.modes),
        "sequence": sequence.sequence,
        "counts": {"searched": len(sizes)},
    }


def test_an_ambiguous_stop_offers_a_smaller_region_only():
    """Two narrow groups: never a larger region, wherever the search ran."""
    rival = [_star(U0 + 0.5 + d) for d in np.linspace(-0.004, 0.004, 10)]
    record = _stop_record(_stars(20) + rival)

    for whole_image, searched in ((False, "Within a region."), (True, "")):
        message = screen_stop_message(
            record, searched=searched, whole_image=whole_image
        )

        assert "A smaller data.region about the target holds stars" in message
        assert "larger" not in message.split("No PSF candidate was saved")[1]


def test_a_stop_after_a_search_of_the_whole_image_offers_no_larger_region():
    """When the whole image was searched, no larger region is suggested."""
    record = _stop_record(_stars(5))
    whole = screen_stop_message(
        record, searched="The whole image was searched.", whole_image=True
    )
    region = screen_stop_message(
        record, searched="Within the working region.", whole_image=False
    )

    assert "The whole image was searched. No larger region offers more stars." in (
        whole
    )
    assert "a larger data.region" not in whole.lower()
    assert "A larger data.region, or data.region: whole, offers more stars." in region


@pytest.mark.parametrize(
    ("n_flank", "expected"),
    [
        (2, "a region about 2 times larger in area would let the group stand out"),
        (8, "as many as its members or more, so a larger region"),
    ],
)
def test_a_group_next_to_many_sources_says_whether_a_larger_region_helps(
    n_flank, expected
):
    """The remedy is phrased in the counts that the tail test compares."""
    flank = [_star(U0 + 0.07 + 0.07 * k / n_flank) for k in range(n_flank)]
    record = _stop_record(_stars(7 if n_flank == 2 else 8, spread=0.01) + flank)

    assert record["detail"] == "not_standing_out"

    message = screen_stop_message(record, searched="Here.", whole_image=False)

    assert expected in message


def test_a_profile_stop_suggests_no_change_of_region():
    """A profile scatter is not lifted by a larger or a smaller region."""
    delta = 3.6 * 0.005 / 1.4826
    rho4 = [2.0 - delta] * 10 + [2.0 + delta] * 10
    sizes = [
        _star(U0 + d, snr_w=1.0e6, rho4=r)
        for d, r in zip(np.linspace(-0.004, 0.004, 20), rho4, strict=True)
    ]
    message = screen_stop_message(_stop_record(sizes), whole_image=False)

    assert "region" not in message.split("(error.record in Python).")[1].split(
        "Otherwise"
    )[0]
    assert "Otherwise declare psf.type: gaussian" in message


def test_the_account_of_a_stop_for_a_terminal():
    """Counts by reason, the smallest groups and the group taken for stars."""
    flank = [_star(U0 + 0.07 + 0.014 * k, rho4=1.9) for k in range(6)]
    record = _stop_record(_stars(8, spread=0.01) + flank)
    lines = screen_record_lines(record, max_groups=1)

    assert lines[0] == (
        "candidates: 14 after the shape cuts, 14 searched; verdicts: 14 searched"
    )
    assert lines[1].startswith("group at 1.3")
    assert "8 within 5 per cent, 0 smaller and 6 larger next to it" in lines[1]
    assert lines[2].startswith("... and ")
    assert lines[-1].startswith("taken for the stars: the group at 1.3")
