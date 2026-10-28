"""Point-source screen of the candidate stars of an empirical PSF.

Candidate selection keeps the compact, round and isolated sources, which
include galaxies, cosmic rays and hot pixels as well as stars, and the
centroid check of the PSF does not detect them. The screen measures every
candidate that passes the shape cuts on its own cutout and keeps those that
form the stellar sequence of their sizes; when the candidates hold no such
sequence, or two, the selection stops and says why.

The rule ``size_sequence_v1`` is frozen with its constants, as the builder of
the PSF is: any change of a constant is a new rule with a new name. The rule
and the digest of its constants enter the provenance of the candidates and of
the PSF; the neutral rule ``none`` selects by shape only and leaves every
identity as it was before the screen existed.

Measurement
    The cutout is the cutout of the signal-to-noise stage. The median and
    the robust scale of its outer frame of three pixels, each pixel counted
    once, are the background and the noise. The light is weighted by an
    elliptical Gaussian whose covariance ``M`` is matched to it by iterating
    ``M <- 2 C``, where ``C`` is the weighted covariance of the light about
    its weighted centroid; for a Gaussian source the fixed point is its own
    covariance (adaptive moments). The size is
    ``s = det(M) ** (1 / 4)``, which does not grow with the brightness of the
    source, unlike the isophotal segment; ``rho4`` is the fourth moment of
    the light within the weight, a measure of the shape of the profile.

Decision
    The stars are the smallest resolved sources and share one PSF, so their
    values of ``ln s`` form a narrow dense group at the small end. Sources
    sharper than any point source, such as cosmic rays and hot pixels,
    larger sources, such as galaxies, and sources whose profile differs
    from that of the group are rejected; see :func:`find_stellar_sequence`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson

from baymorph.pipeline.checkpoint_signing import canonical_json_sha256
from baymorph.pipeline.psf_cutouts import (
    estimate_cutout_peak_snr,
    extract_centered_cutout,
)
from baymorph.stats.robust import robust_sigma

if TYPE_CHECKING:
    from baymorph.pipeline.psf_candidates import PSFCandidate

POINT_SOURCE_SCREEN_RULES = ("none", "size_sequence_v1")
"""Rules of ``psf.empirical.candidate_selection.point_source_screen``."""

DEFAULT_POINT_SOURCE_SCREEN = "size_sequence_v1"
"""Rule of a configuration that declares none."""

PSF_CANDIDATE_SCREEN_SCHEMA = "baymorph.psf_candidate_screen"
PSF_CANDIDATE_SCREEN_SCHEMA_VERSION = 1

SIZE_SEQUENCE_V1: Mapping[str, Any] = MappingProxyType(
    {
        # Measurement on the cutout of the signal-to-noise stage.
        "border_width_pixels": 3,
        "start_size_floor_pixels": 0.5,
        "max_iterations": 200,
        "moment_tolerance": 1.0e-7,
        "centre_tolerance_pixels": 1.0e-5,
        # A weight whose smaller eigenvalue falls below this, in square
        # pixels, has collapsed onto one or two pixels.
        "collapse_eigenvalue_pixels2": 0.04,
        # The weight must stay within a third of the half-size of the cutout,
        # and its centre within half of it.
        "window_size_divisor": 3.0,
        "window_shift_divisor": 2.0,
        # Decision. The floor is the adaptive size of an Airy core of FWHM
        # 1.5 pixels; one- to three-pixel artifacts collapse below it.
        "size_floor_pixels": 0.60,
        "window_ln_size": 0.05,
        "flank_multiple": 3.0,
        # The Poisson mean of the flank count is at least this, so that a
        # group with empty flanks still needs six members to stand out.
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
)
"""Frozen constants of the rule ``size_sequence_v1``.

They were set on synthetic benches whose acceptance criteria were written
before the runs; a change of any of them is a new rule.
"""

SIZE_SEQUENCE_V1_SHA256 = canonical_json_sha256(dict(SIZE_SEQUENCE_V1))
"""Digest of the constants, recorded in the provenance of the products."""

_RULE_CONSTANTS = {"size_sequence_v1": (SIZE_SEQUENCE_V1, SIZE_SEQUENCE_V1_SHA256)}

STOP_CODES = ("too_few", "no_sequence", "ambiguous", "not_smallest", "too_few_members")
"""Outcomes of a screen that stops."""

STOP_DETAILS: Mapping[str, str | None] = MappingProxyType(
    {
        "too_few": "larger_region",
        "no_dense_group": "larger_region",
        "not_standing_out": "larger_region_if_stars_dominate",
        "too_wide": "smaller_region",
        "profile": None,
        "rival": "smaller_region",
        "smaller_sources": None,
        "too_few_members": "larger_region",
    }
)
"""Why a screen stopped, and which change of the searched region can help.

``None`` means that no change of the region is expected to help.
"""

_UNMEASURED_STATUSES = ("outside_image", "cutout_too_small", "no_flux", "not_converged")

PIPELINE_REMEDY_TERMS: Mapping[str, str] = MappingProxyType(
    {
        "larger": "a larger data.region, or data.region: whole,",
        "smaller": "a smaller data.region about the target",
        "fallback": (
            "declare psf.type: gaussian, or "
            "psf.empirical.candidate_selection.point_source_screen: none to "
            "select the stars by shape only, at the risk of galaxies, cosmic "
            "rays and hot pixels in the PSF"
        ),
    }
)
"""Remedies of a stop in the terms of the configuration."""

_LARGEST_AREA_FACTOR = 10
"""Largest enlargement of the searched area that a stop message quotes."""


class PointSourceScreenChangeError(ValueError):
    """Saved PSF products differ from the configuration only in the screen.

    Products selected with one point-source screen are not reused under
    another, and accepted products are never rewritten in place, so the
    message names the declaration that reuses them and the new namespace
    that selects the stars again.
    """


def point_source_screen_rule(candidate_selection: Mapping[str, Any]) -> str:
    """Return the point-source screen of a candidate-selection section.

    Parameters
    ----------
    candidate_selection : mapping
        The section ``psf.empirical.candidate_selection``; a section without
        the key takes the default rule, as configuration resolution gives it.

    Returns
    -------
    str
        One of :data:`POINT_SOURCE_SCREEN_RULES`.

    Raises
    ------
    ValueError
        If the value is not one of the rules.
    """
    rule = candidate_selection.get("point_source_screen", DEFAULT_POINT_SOURCE_SCREEN)

    if not isinstance(rule, str) or rule not in POINT_SOURCE_SCREEN_RULES:
        raise ValueError(
            "psf.empirical.candidate_selection.point_source_screen must be one "
            f"of {list(POINT_SOURCE_SCREEN_RULES)}, not {rule!r}."
        )

    return rule


def point_source_screen_marker(rule: str) -> dict[str, str] | None:
    """Return the provenance marker of a rule, or None for ``"none"``.

    Parameters
    ----------
    rule : str
        One of :data:`POINT_SOURCE_SCREEN_RULES`.

    Returns
    -------
    dict or None
        ``{"rule": ..., "constants_sha256": ...}``; the neutral rule has no
        marker, so the products it selects keep their earlier identity.
    """
    if rule == "none":
        return None

    if rule not in _RULE_CONSTANTS:
        raise ValueError(f"Unknown point-source screen {rule!r}.")

    return {"rule": rule, "constants_sha256": _RULE_CONSTANTS[rule][1]}


def recorded_screen_rule(provenance: Mapping[str, Any]) -> str | None:
    """Return the screen under which a recorded provenance was made.

    Parameters
    ----------
    provenance : mapping
        A recorded provenance of PSF candidates or of an empirical PSF.

    Returns
    -------
    str or None
        ``"none"`` when the provenance carries no marker, as those of earlier
        releases, the rule of a marker that this version of BayMorph writes,
        or None for a marker it does not know.
    """
    marker = provenance.get("point_source_screen")

    if marker is None:
        return "none"

    if isinstance(marker, Mapping):
        rule = marker.get("rule")

        if rule in _RULE_CONSTANTS and dict(marker) == point_source_screen_marker(rule):
            return rule

    return None


def screen_change_message(products: str, *, recorded: str, current: str) -> str:
    """Refuse products that differ from the configuration only in the screen.

    Parameters
    ----------
    products : str
        What was saved, such as ``"PSF candidates"``.
    recorded : str
        The rule under which the products were selected.
    current : str
        The rule of the configuration.

    Returns
    -------
    str
        The refusal, with the declaration that reuses the products and the
        new output namespace that selects the stars again; accepted products
        are never rewritten in place, so it does not suggest an overwrite.
    """
    key = "psf.empirical.candidate_selection.point_source_screen"

    if recorded == "none":
        how = (
            f"as BayMorph does by default ({key}: {current})"
            if current == DEFAULT_POINT_SOURCE_SCREEN
            else f"as it declares ({key}: {current})"
        )

        return (
            f"The saved {products} were selected without the point-source "
            "screen of the stars, as BayMorph selected them before the screen "
            f"existed, while the configuration screens the stars, {how}; nothing "
            f"else differs. Declare {key}: none to reuse them unchanged, or use a "
            "new output namespace, such as another object.name, to select "
            "screened stars; accepted products are not rewritten in place."
        )

    return (
        f"The saved {products} were selected with the point-source screen "
        f"{recorded}, and the configuration declares {key}: {current}; nothing "
        f"else differs. Declare {key}: {recorded} to reuse them, or use a new "
        "output namespace, such as another object.name, to select the stars "
        "again; accepted products are not rewritten in place."
    )


def screen_difference_note(recorded: str | None, current: str) -> str:
    """Name a difference of the screen among other differences of provenance.

    Parameters
    ----------
    recorded : str or None
        The rule of the recorded provenance, None when it is unknown.
    current : str
        The rule of the configuration.

    Returns
    -------
    str
        A sentence, or an empty string when the screens agree.
    """
    if recorded == current:
        return ""

    def described(rule: str | None) -> str:
        if rule is None:
            return "a screen this version of BayMorph does not know"

        return "no point-source screen" if rule == "none" else f"the screen {rule}"

    return (
        f" The stars were also selected with {described(recorded)}, and the "
        f"configuration selects them with {described(current)}."
    )


@dataclass(frozen=True, slots=True)
class CandidateSize:
    """Measurement of one PSF candidate on its own cutout.

    Parameters
    ----------
    status : str
        ``"measured"``, or why the size could not be measured:
        ``"collapsed"`` (the weight shrank onto one or two pixels),
        ``"exceeds_window"`` (the weight outgrew the cutout),
        ``"outside_image"``, ``"cutout_too_small"``, ``"no_flux"`` or
        ``"not_converged"``.
    size : float, optional
        Adaptive size ``det(M) ** (1 / 4)``, in pixels.
    rho4 : float, optional
        Fourth moment of the light within the matched weight.
    e1, e2 : float, optional
        Ellipticity components of the matched weight; recorded only.
    snr_w : float, optional
        Signal-to-noise ratio of the weighted flux.
    peak_snr : float, optional
        Peak signal-to-noise ratio of the signal-to-noise stage.
    background, noise : float, optional
        Median and robust scale of the border of the cutout.
    """

    status: str
    size: float | None = None
    rho4: float | None = None
    e1: float | None = None
    e2: float | None = None
    snr_w: float | None = None
    peak_snr: float | None = None
    background: float | None = None
    noise: float | None = None


@dataclass(frozen=True, slots=True)
class StellarSequence:
    """Outcome of the search for the stellar sequence among measured sizes.

    Parameters
    ----------
    outcome : str
        ``"found"``, or the code of the stop: one of :data:`STOP_CODES`.
    finding : str
        What the search found, in one sentence; empty when it was found.
    detail : str
        Why the search stopped, a key of :data:`STOP_DETAILS`; empty when
        it was found.
    reasons : tuple of str
        Verdict of every candidate, in the order of the input.
    member : tuple of bool
        Whether each candidate belongs to the sequence.
    n_search : int
        Candidates with a measured size of at least the floor and a peak
        signal-to-noise of at least the threshold.
    modes : tuple of dict
        Every group of sizes found, in increasing size.
    sequence : dict, optional
        The group taken for the stars, when the search reached one.
    """

    outcome: str
    finding: str
    detail: str
    reasons: tuple[str, ...]
    member: tuple[bool, ...]
    n_search: int
    modes: tuple[dict[str, Any], ...]
    sequence: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class ScreenResult:
    """Candidates that passed the screen, and its record.

    Parameters
    ----------
    members : tuple of PSFCandidate
        Every member of the stellar sequence, in the order of the pool.
    kept : tuple of PSFCandidate
        The first ``target_n_candidates`` members.
    record : dict
        The record of the screen, with positions in pixels of the whole
        image.
    """

    members: tuple[PSFCandidate, ...]
    kept: tuple[PSFCandidate, ...]
    record: dict[str, Any]


class PointSourceScreenError(ValueError):
    """The screen found no stellar sequence among the PSF candidates.

    Attributes
    ----------
    code : str
        One of :data:`STOP_CODES`.
    record : dict
        The record of the screen: the measurement and verdict of every
        candidate, with positions in pixels of the whole image.
    finding : str
        What the screen found, in one sentence, without remedies.
    """

    def __init__(
        self, message: str, *, code: str, record: dict[str, Any], finding: str
    ) -> None:
        super().__init__(message)
        self.code = code
        self.record = record
        self.finding = finding

    def __reduce__(self):
        """Pickle the code, the record and the finding with the message."""
        return (
            _rebuild_screen_error,
            (str(self), self.code, self.record, self.finding),
        )


def _rebuild_screen_error(
    message: str, code: str, record: dict[str, Any], finding: str
) -> PointSourceScreenError:
    """Rebuild a pickled :class:`PointSourceScreenError`."""
    return PointSourceScreenError(message, code=code, record=record, finding=finding)


def _finite_or_none(value: Any) -> float | None:
    """Return a finite float, or None."""
    if value is None:
        return None

    value = float(value)

    return value if np.isfinite(value) else None


def _border_once(data: NDArray[np.floating], width: int) -> NDArray[np.floating]:
    """Return the outer frame of a cutout, each pixel once."""
    return np.concatenate(
        [
            data[:width, :].ravel(),
            data[-width:, :].ravel(),
            data[width:-width, :width].ravel(),
            data[width:-width, -width:].ravel(),
        ]
    )


def _adaptive_moments(
    stamp: NDArray[np.floating], x0: float, y0: float, start_size: float, noise: float
) -> CandidateSize:
    """Return the adaptive moments of a background-subtracted cutout.

    The weight ``W = exp(-d^T M^-1 d / 2)`` is updated as ``M <- 2 C``, where
    ``C`` is the W-weighted covariance of the light about the W-weighted
    centroid; for a Gaussian source the fixed point is its covariance.
    """
    c = SIZE_SEQUENCE_V1
    data = np.asarray(stamp, dtype=float)
    yy, xx = np.indices(data.shape, dtype=float)
    half = 0.5 * (min(data.shape) - 1)
    start = start_size if np.isfinite(start_size) else 0.0
    mxx = myy = max(start, c["start_size_floor_pixels"]) ** 2
    mxy = 0.0
    x, y = x0, y0

    for _ in range(c["max_iterations"]):
        det = mxx * myy - mxy * mxy

        if not np.isfinite(det) or det <= 0:
            return CandidateSize(status="collapsed")

        ixx, iyy, ixy = myy / det, mxx / det, -mxy / det
        dx = xx - x
        dy = yy - y
        rho2 = ixx * dx * dx + 2 * ixy * dx * dy + iyy * dy * dy
        weighted = np.exp(-0.5 * rho2) * data
        flux = weighted.sum()

        if not np.isfinite(flux) or flux <= 0:
            return CandidateSize(status="no_flux")

        x_new = (weighted * xx).sum() / flux
        y_new = (weighted * yy).sum() / flux
        dx = xx - x_new
        dy = yy - y_new
        nxx = 2 * (weighted * dx * dx).sum() / flux
        nyy = 2 * (weighted * dy * dy).sum() / flux
        nxy = 2 * (weighted * dx * dy).sum() / flux
        trace = nxx + nyy
        spread = np.sqrt(max(0.25 * (nxx - nyy) ** 2 + nxy**2, 0.0))
        smaller = 0.5 * trace - spread
        larger = 0.5 * trace + spread

        if not np.isfinite(smaller) or smaller < c["collapse_eigenvalue_pixels2"]:
            return CandidateSize(status="collapsed")

        if (
            np.sqrt(larger) > half / c["window_size_divisor"]
            or abs(x_new - x0) > half / c["window_shift_divisor"]
            or abs(y_new - y0) > half / c["window_shift_divisor"]
        ):
            return CandidateSize(status="exceeds_window")

        change = (abs(nxx - mxx) + abs(nyy - myy) + 2 * abs(nxy - mxy)) / (mxx + myy)
        moved = np.hypot(x_new - x, y_new - y)
        mxx, myy, mxy, x, y = nxx, nyy, nxy, x_new, y_new

        if change < c["moment_tolerance"] and moved < c["centre_tolerance_pixels"]:
            break
    else:
        return CandidateSize(status="not_converged")

    # The fourth moment and the weighted flux use the converged weight.
    det = mxx * myy - mxy * mxy
    ixx, iyy, ixy = myy / det, mxx / det, -mxy / det
    dx = xx - x
    dy = yy - y
    rho2 = ixx * dx * dx + 2 * ixy * dx * dy + iyy * dy * dy
    weight = np.exp(-0.5 * rho2)
    flux = float((weight * data).sum())

    if not np.isfinite(flux) or flux <= 0:
        return CandidateSize(status="no_flux")

    trace = mxx + myy

    return CandidateSize(
        status="measured",
        size=float(det**0.25),
        rho4=float((weight * data * rho2 * rho2).sum() / flux),
        e1=float((mxx - myy) / trace),
        e2=float(2 * mxy / trace),
        snr_w=(
            _finite_or_none(flux / (noise * np.sqrt((weight * weight).sum())))
            if noise > 0
            else None
        ),
    )


def measure_candidate_size(
    image: NDArray[np.floating], candidate: PSFCandidate
) -> CandidateSize:
    """Measure the adaptive size and profile of one PSF candidate.

    Parameters
    ----------
    image : ndarray
        The array in which the candidate was found, that of the working
        region when there is one.
    candidate : PSFCandidate
        A candidate of the pool, with its position in the pixels of
        ``image``, its cutout half-size and its catalogue sigmas, which start
        the weight.

    Returns
    -------
    CandidateSize
        The measurement, or the status that explains why there is none.
    """
    c = SIZE_SEQUENCE_V1
    half_size = int(candidate.half_size)
    width = c["border_width_pixels"]

    try:
        cutout = extract_centered_cutout(
            image, x=candidate.x, y=candidate.y, half_size=half_size
        )
    except ValueError:
        return CandidateSize(status="outside_image")

    data = np.asarray(cutout, dtype=float)

    if 2 * width >= min(data.shape):
        return CandidateSize(status="cutout_too_small")

    peak_snr = _finite_or_none(estimate_cutout_peak_snr(data, border_width=width))
    border = _border_once(data, width)
    border = border[np.isfinite(border)]

    if border.size == 0:
        return CandidateSize(status="no_flux", peak_snr=peak_snr)

    background = float(np.median(border))
    noise = float(robust_sigma(border, center=background))
    sigmas = (candidate.semimajor_sigma, candidate.semiminor_sigma)
    start = (
        float(np.sqrt(sigmas[0] * sigmas[1]))
        if all(value is not None and value > 0 for value in sigmas)
        else 0.0
    )
    measured = _adaptive_moments(
        data - background,
        half_size + (candidate.x - int(round(candidate.x))),
        half_size + (candidate.y - int(round(candidate.y))),
        start,
        noise,
    )

    return replace(
        measured,
        peak_snr=peak_snr,
        background=_finite_or_none(background),
        noise=_finite_or_none(noise),
    )


def _local_modes(u: NDArray[np.floating]) -> list[float]:
    """Return the distinct fixed points of a flat-kernel median shift."""
    c = SIZE_SEQUENCE_V1
    window = c["window_ln_size"]
    modes: list[float] = []

    for start in np.sort(u):
        m = float(start)

        for _ in range(c["mode_steps_max"]):
            new = float(np.median(u[np.abs(u - m) <= window]))

            if abs(new - m) < c["mode_tolerance"]:
                break

            m = new

        if not any(abs(m - q) < c["mode_merge_fraction"] * window for q in modes):
            modes.append(m)

    return sorted(modes)


def _tail_probability(n_core: int, n_flank: int) -> float:
    """Return the chance of ``n_core`` values or more against a flank count."""
    mean = max(SIZE_SEQUENCE_V1["flank_mean_floor"], float(n_flank))

    return float(poisson.sf(n_core - 1, mean))


def required_core_count(n_flank: int) -> int:
    """Return the smallest group that stands out from an adjacent interval.

    Parameters
    ----------
    n_flank : int
        Number of candidates in the larger of the two adjacent intervals.

    Returns
    -------
    int
        The smallest count, at least six, whose Poisson tail probability
        against ``n_flank`` is at most the frozen threshold of 1e-3.
    """
    c = SIZE_SEQUENCE_V1
    n = int(c["min_members"])

    while _tail_probability(n, int(n_flank)) > c["tail_probability_max"]:
        n += 1

    return n


def _area_factor(n_core: int, n_flank: int) -> int | None:
    """Return the enlargement at which a group of these proportions stands out.

    Both counts grow with the area searched when the stars and the sources
    next to them in size are spread alike; the factor is the smallest whole
    multiple of the area, up to ten, at which the group would stand out.
    """
    if n_core <= n_flank:
        return None

    for factor in range(2, _LARGEST_AREA_FACTOR + 1):
        if factor * n_core >= required_core_count(factor * n_flank):
            return factor

    return None


def _mode_row(u: NDArray[np.floating], m: float) -> dict[str, Any]:
    """Return the counts, scatter and significance of one group of sizes."""
    c = SIZE_SEQUENCE_V1
    window = c["window_ln_size"]
    reach = c["flank_multiple"] * window
    core = np.abs(u - m) <= window
    n_core = int(core.sum())
    n_lo = int(((u < m - window) & (u >= m - reach)).sum())
    n_hi = int(((u > m + window) & (u <= m + reach)).sum())
    scatter = (
        float(c["mad_to_sigma"] * np.median(np.abs(u[core] - m))) if n_core else np.nan
    )
    p_tail = _tail_probability(n_core, max(n_lo, n_hi))
    significant = (
        n_core >= c["min_members"]
        and p_tail <= c["tail_probability_max"]
        and scatter <= c["core_scatter_max"]
    )

    return {
        "ln_size": m,
        "size_pix": float(np.exp(m)),
        "n_core": n_core,
        "n_lo": n_lo,
        "n_hi": n_hi,
        "p_tail": p_tail,
        "tau_hat": _finite_or_none(scatter),
        "significant": bool(significant),
    }


def _as_array(sizes: Sequence[CandidateSize], name: str) -> NDArray[np.floating]:
    """Return one field of the measurements as floats, NaN where absent."""
    return np.array(
        [
            np.nan if getattr(m, name) is None else float(getattr(m, name))
            for m in sizes
        ],
        dtype=float,
    )


def find_stellar_sequence(
    sizes: Sequence[CandidateSize], *, min_peak_snr: float
) -> StellarSequence:
    """Find the stellar sequence among the measured sizes of the candidates.

    The search set holds the candidates with a measured size of at least
    0.60 pixels and a peak signal-to-noise of at least ``min_peak_snr``;
    ``u = ln s``. Groups are the fixed points of a median shift of width
    ``W = 0.05`` started from every ``u``. For each group, ``n_core`` counts
    the values within ``W``, the flanks count those between ``W`` and ``3 W``
    on either side, ``tau_hat`` is the robust scatter of the core, and the
    group is significant when it holds six values, its count has a Poisson
    tail probability of at most 1e-3 under the larger flank count, and
    ``tau_hat`` is at most 0.03.

    The lowest group of six values must be significant, no significant
    group more than ``2 W`` above it may hold half as many values, and its
    core must share one profile: the robust scatter of ``rho4`` divided by
    its noise ``hypot(2.5 / snr_w, 0.005)`` is at most 3.5. Members lie
    within ``3 hypot(tau, 1.5 / snr_w)`` of the group in ``u``, where ``tau``
    is the scatter of the core less its noise, at least 0.01, and within
    ``3 hypot(0.05, sigma_rho4)`` of its median ``rho4``. The search stops
    when more than ``max(2, 0.2 n_members)`` candidates are smaller than the
    group, or when fewer than six are members.

    Parameters
    ----------
    sizes : sequence of CandidateSize
        Measurements of the candidates, in the order of the pool.
    min_peak_snr : float
        Peak signal-to-noise that the construction of the PSF demands of a
        star.

    Returns
    -------
    StellarSequence
        The outcome, with the verdict of every candidate.
    """
    c = SIZE_SEQUENCE_V1
    window = c["window_ln_size"]
    n_min = c["min_members"]
    n = len(sizes)
    status = np.array([m.status for m in sizes], dtype=object)
    size = _as_array(sizes, "size")
    peak_snr = _as_array(sizes, "peak_snr")
    snr_w = _as_array(sizes, "snr_w")
    rho4 = _as_array(sizes, "rho4")
    measured = status == "measured"
    sharp = (status == "collapsed") | (measured & (size < c["size_floor_pixels"]))
    too_large = status == "exceeds_window"
    unmeasured = np.isin(status, _UNMEASURED_STATUSES)
    bright = np.isfinite(peak_snr) & (peak_snr >= min_peak_snr)
    weighed = np.isfinite(snr_w) & (snr_w > 0)
    search = measured & ~sharp & bright & weighed
    u_all = np.full(n, np.nan)
    u_all[measured] = np.log(size[measured])
    u = u_all[search]

    reasons = []

    for i in range(n):
        if sharp[i]:
            reasons.append("sharper_than_point_source")
        elif too_large[i]:
            reasons.append("larger_than_window")
        elif unmeasured[i]:
            reasons.append("size_not_measurable")
        elif not bright[i]:
            reasons.append("peak_snr_below_threshold")
        elif not weighed[i]:
            # A border without a positive noise scale gives no weighted S/N.
            reasons.append("size_not_measurable")
        else:
            reasons.append("searched")

    def stop(code, detail, finding, modes=(), sequence=None, member=None):
        return StellarSequence(
            outcome=code,
            finding=finding,
            detail=detail,
            reasons=tuple(reasons),
            member=tuple(bool(v) for v in (member if member is not None else [0] * n)),
            n_search=int(search.sum()),
            modes=tuple(modes),
            sequence=sequence,
        )

    if search.sum() < n_min:
        return stop(
            "too_few",
            "too_few",
            f"only {int(search.sum())} of the {n} candidates that pass the shape "
            "cuts have a measured size of at least "
            f"{c['size_floor_pixels']:.2f} px and a peak signal-to-noise of at "
            f"least {min_peak_snr:g}; the stellar sequence needs {n_min}.",
        )

    modes = [_mode_row(u, m) for m in _local_modes(u)]
    dense = [row for row in modes if row["n_core"] >= n_min]

    if not dense:
        best = max(modes, key=lambda row: (row["n_core"], -row["ln_size"]))

        return stop(
            "no_sequence",
            "no_dense_group",
            f"no {n_min} candidates share one size within "
            f"{100 * window:.0f} per cent; the largest group holds "
            f"{best['n_core']} at {best['size_pix']:.2f} px.",
            modes,
        )

    # The stars are the smallest resolved sources: the lowest dense group
    # must itself be the sequence.
    lowest = min(dense, key=lambda row: row["ln_size"])
    sequence = dict(lowest)
    k, s = lowest["n_core"], lowest["size_pix"]

    if not lowest["significant"]:
        if lowest["p_tail"] > c["tail_probability_max"]:
            detail = "not_standing_out"
            n_flank = max(lowest["n_lo"], lowest["n_hi"])
            reach = c["flank_multiple"] * window
            side = (
                f"{100 * np.expm1(window):.0f} to {100 * np.expm1(reach):.0f} "
                "per cent larger"
                if lowest["n_hi"] >= lowest["n_lo"]
                else f"{-100 * np.expm1(-window):.0f} to "
                f"{-100 * np.expm1(-reach):.0f} per cent smaller"
            )
            sequence["n_required"] = required_core_count(n_flank)
            finding = (
                f"the smallest group of candidates with one size, {k} within "
                f"{100 * window:.0f} per cent of {s:.2f} px, does not stand out "
                f"from its neighbours: {n_flank} candidates are {side}, so a "
                f"group of {k} arises by chance with probability "
                f"{lowest['p_tail']:.2g} (at most {c['tail_probability_max']:g}); "
                f"against {n_flank}, the group needs {sequence['n_required']}."
            )
        else:
            detail = "too_wide"
            finding = (
                f"the smallest group of candidates with one size, {k} within "
                f"{100 * window:.0f} per cent of {s:.2f} px, is too wide to be "
                f"stars of one PSF: its sizes scatter by {lowest['tau_hat']:.3f} "
                f"in ln s (at most {c['core_scatter_max']:g})."
            )

        return stop("no_sequence", detail, finding, modes, sequence)

    separation = c["ambiguity_separation_windows"] * window
    rivals = [
        row
        for row in modes
        if row["significant"]
        and row["ln_size"] > lowest["ln_size"] + separation
        and row["n_core"] >= max(n_min, c["ambiguity_ratio"] * lowest["n_core"])
    ]

    if rivals:
        rival = rivals[0]
        sequence["rival"] = dict(rival)

        return stop(
            "ambiguous",
            "rival",
            f"two narrow groups of sizes, {k} candidates at {s:.2f} px and "
            f"{rival['n_core']} at {rival['size_pix']:.2f} px, could each be the "
            "stars, or the PSF changes across the searched region.",
            modes,
            sequence,
        )

    centre = lowest["ln_size"]
    core = search & (np.abs(u_all - centre) <= window)
    rho4_core = rho4[core]
    rho4_centre = float(np.median(rho4_core))
    rho4_sigma_core = np.hypot(
        c["rho4_error_scale"] / snr_w[core], c["rho4_noise_floor"]
    )
    rho4_z = float(
        c["mad_to_sigma"]
        * np.median(np.abs((rho4_core - rho4_centre) / rho4_sigma_core))
    )
    sequence["rho4_centre"] = rho4_centre
    sequence["rho4_z"] = rho4_z

    if rho4_z > c["rho4_z_max"]:
        # In synthetic benches both compact galaxies and bright stars of a
        # coarsely sampled PSF, whose profile changes with their position
        # within a pixel, scattered this much.
        return stop(
            "no_sequence",
            "profile",
            f"the smallest group of candidates with one size ({k} at {s:.2f} px) "
            f"does not share one profile: its fourth moment scatters {rho4_z:.1f} "
            f"times its noise (at most {c['rho4_z_max']:g}). Compact galaxies do "
            "this, and so do bright stars of a PSF sampled with little more than "
            "two pixels per FWHM, whose profile changes with their position "
            "within a pixel.",
            modes,
            sequence,
        )

    size_sigma = c["size_error_scale"] / snr_w
    tau = float(
        np.sqrt(max(0.0, lowest["tau_hat"] ** 2 - np.median(size_sigma[core] ** 2)))
    )
    tau = max(tau, c["width_floor"])
    sequence["tau"] = tau
    member = np.zeros(n, dtype=bool)

    for i in np.flatnonzero(search):
        distance = u_all[i] - centre
        tolerance = c["member_widths"] * np.hypot(tau, size_sigma[i])
        rho4_sigma = np.hypot(c["rho4_error_scale"] / snr_w[i], c["rho4_noise_floor"])
        rho4_tolerance = c["member_widths"] * np.hypot(
            c["rho4_member_floor"], rho4_sigma
        )

        if distance > tolerance:
            reasons[i] = "larger_than_stellar_sequence"
        elif distance < -tolerance:
            reasons[i] = "smaller_than_stellar_sequence"
        elif abs(rho4[i] - rho4_centre) > rho4_tolerance:
            reasons[i] = "profile_unlike_stellar_sequence"
        else:
            reasons[i] = "member"
            member[i] = True

    n_members = int(member.sum())
    n_smaller = reasons.count("smaller_than_stellar_sequence")
    sequence["n_members"] = n_members

    allowed = max(c["smaller_max_count"], c["smaller_max_fraction"] * n_members)

    if n_smaller > allowed:
        return stop(
            "not_smallest",
            "smaller_sources",
            f"{n_smaller} candidates are smaller than the group of {n_members} at "
            f"{s:.2f} px, more than the {allowed:g} allowed, so that group "
            "cannot be shown to be the smallest sources, the stars. Sources "
            "smaller than the stars but above the floor are, for example, "
            "cosmic rays or hot pixels smoothed by resampling.",
            modes,
            sequence,
            member,
        )

    if n_members < n_min:
        return stop(
            "too_few_members",
            "too_few_members",
            f"only {n_members} candidates belong to the stellar sequence at "
            f"{s:.2f} px; the PSF needs {n_min}.",
            modes,
            sequence,
            member,
        )

    return StellarSequence(
        outcome="found",
        finding="",
        detail="",
        reasons=tuple(reasons),
        member=tuple(bool(v) for v in member),
        n_search=int(search.sum()),
        modes=tuple(modes),
        sequence=sequence,
    )


def _region_remedy(
    record: Mapping[str, Any], *, whole_image: bool | None, terms: Mapping[str, str]
) -> str:
    """Return the change of the searched region that can lift a stop."""
    kind = STOP_DETAILS.get(record.get("detail", ""))

    if kind is None:
        return ""

    larger = terms["larger"]
    larger = larger[0].upper() + larger[1:]
    smaller = terms["smaller"]
    smaller = smaller[0].upper() + smaller[1:]

    if kind == "smaller_region":
        return f" {smaller} holds stars of one PSF."

    if whole_image:
        return " No larger region offers more stars."

    where = (
        larger
        if whole_image is False
        else "A larger part of the image, when there is one,"
    )

    if kind == "larger_region":
        return f" {where} offers more stars."

    # The group did not stand out from the sources next to it in size: a
    # larger region adds both, so it helps only while the stars dominate.
    sequence = record.get("sequence") or {}
    n_core = int(sequence.get("n_core", 0))
    n_flank = max(int(sequence.get("n_lo", 0)), int(sequence.get("n_hi", 0)))
    factor = _area_factor(n_core, n_flank)

    if n_core <= n_flank:
        return (
            f" The {n_flank} candidates next to the group in size are as many as "
            "its members or more, so a larger region, which adds both, does not "
            "help."
        )

    if factor is None:
        return (
            f" {where} helps only if it adds stars faster than sources next to "
            f"them in size: at the present proportions, {n_core} to {n_flank}, a "
            f"region more than {_LARGEST_AREA_FACTOR} times larger in area would "
            "be needed."
        )

    return (
        f" {where} helps if it adds stars faster than sources next to them in "
        f"size: at the present proportions, {n_core} to {n_flank}, a region "
        f"about {factor} times larger in area would let the group stand out."
    )


def screen_stop_message(
    record: Mapping[str, Any],
    *,
    searched: str = "",
    whole_image: bool | None = None,
    terms: Mapping[str, str] = PIPELINE_REMEDY_TERMS,
) -> str:
    """Return the message of a stop of the screen, with its remedies.

    Parameters
    ----------
    record : mapping
        The record of the stop, as :func:`screen_psf_candidates` writes it.
    searched : str, optional
        A sentence that says where the stars were searched.
    whole_image : bool, optional
        Whether the whole image was searched; None when it is not known.
    terms : mapping, default=PIPELINE_REMEDY_TERMS
        How the remedies are named: ``"larger"`` and ``"smaller"`` name a
        larger and a smaller searched region, and ``"fallback"`` the
        alternatives to the screen, in the terms of the configuration or of
        the caller.

    Returns
    -------
    str
        The finding, the statement that no candidate was saved, where the
        stars were searched, the change of region that can help, if any,
        and the alternatives to the screen.
    """
    where = f" {searched.strip()}" if searched.strip() else ""

    return (
        f"PSF candidate screen {record.get('rule', DEFAULT_POINT_SOURCE_SCREEN)}: "
        f"{record.get('finding', '')} No PSF candidate was saved; the size, "
        "profile and verdict of each of the "
        f"{record.get('n_pool', 0)} candidates are in the record of this error "
        f"(error.record in Python).{where}"
        f"{_region_remedy(record, whole_image=whole_image, terms=terms)}"
        f" Otherwise {terms['fallback']}."
    )


def screen_record_lines(record: Mapping[str, Any], *, max_groups: int = 8) -> list[str]:
    """Return a short account of a screen record, for a terminal.

    Parameters
    ----------
    record : mapping
        A record of :func:`screen_psf_candidates`.
    max_groups : int, default=8
        Number of the smallest groups of sizes listed.

    Returns
    -------
    list of str
        The verdicts counted by reason, the smallest groups of sizes with
        their counts, scatter and tail probability, and the group taken for
        the stars, if any.
    """
    counts = record.get("counts") or {}
    lines = [
        f"candidates: {record.get('n_pool', 0)} after the shape cuts, "
        f"{record.get('n_search', 0)} searched; verdicts: "
        + (
            ", ".join(f"{count} {reason}" for reason, count in sorted(counts.items()))
            or "none"
        )
    ]
    modes = list(record.get("modes") or [])

    for row in modes[:max_groups]:
        tau_hat = row.get("tau_hat")
        lines.append(
            f"group at {row['size_pix']:.3f} px: {row['n_core']} within 5 per "
            f"cent, {row['n_lo']} smaller and {row['n_hi']} larger next to it, "
            "scatter "
            + ("-" if tau_hat is None else f"{tau_hat:.4f}")
            + f" in ln s, chance {row['p_tail']:.2g}"
            + (", significant" if row.get("significant") else "")
        )

    if len(modes) > max_groups:
        lines.append(f"... and {len(modes) - max_groups} larger groups")

    sequence = record.get("sequence")

    if sequence:
        line = f"taken for the stars: the group at {sequence['size_pix']:.3f} px"

        if sequence.get("rho4_z") is not None:
            line += f", profile scatter {sequence['rho4_z']:.2f} times its noise"

        if sequence.get("n_members") is not None:
            line += f", {sequence['n_members']} members"

        lines.append(line)

    return lines


def screen_psf_candidates(
    image: NDArray[np.floating],
    pool: Sequence[PSFCandidate],
    *,
    min_peak_snr: float,
    target_n_candidates: int,
    origin: tuple[int, int] = (0, 0),
) -> ScreenResult:
    """Keep the PSF candidates that form the stellar sequence of their sizes.

    Parameters
    ----------
    image : ndarray
        The array in which the pool was selected, that of the working region
        when there is one.
    pool : sequence of PSFCandidate
        Every candidate that passes the shape, isolation and no-data cuts,
        in the order of the selection, with positions in pixels of
        ``image``.
    min_peak_snr : float
        Peak signal-to-noise that the construction of the PSF demands of a
        star.
    target_n_candidates : int
        Number of members to keep.
    origin : tuple of int, default=(0, 0)
        Pixel of the whole image at the origin of ``image``; the positions
        of the record are moved by it.

    Returns
    -------
    ScreenResult
        The members in the order of the pool, the first
        ``target_n_candidates`` of them, and the record of the screen.

    Raises
    ------
    PointSourceScreenError
        If the candidates hold no stellar sequence, or two; its record holds
        the measurement and verdict of every candidate.
    """
    min_peak_snr = float(min_peak_snr)

    if not np.isfinite(min_peak_snr) or min_peak_snr < 0.0:
        raise ValueError("min_peak_snr must be finite and non-negative.")

    if int(target_n_candidates) <= 0:
        raise ValueError("target_n_candidates must be positive.")

    sizes = [measure_candidate_size(image, candidate) for candidate in pool]
    sequence = find_stellar_sequence(sizes, min_peak_snr=min_peak_snr)
    member_index = [i for i, flag in enumerate(sequence.member) if flag]
    found = sequence.outcome == "found"
    kept_index = set(member_index[: int(target_n_candidates)]) if found else set()
    origin_x, origin_y = (int(value) for value in origin)
    rule = "size_sequence_v1"
    counts = Counter(sequence.reasons)
    record = {
        "schema": PSF_CANDIDATE_SCREEN_SCHEMA,
        "schema_version": PSF_CANDIDATE_SCREEN_SCHEMA_VERSION,
        "rule": rule,
        "constants": dict(SIZE_SEQUENCE_V1),
        "constants_sha256": SIZE_SEQUENCE_V1_SHA256,
        "outcome": sequence.outcome,
        "finding": sequence.finding,
        "detail": sequence.detail,
        "min_peak_snr": min_peak_snr,
        "target_n_candidates": int(target_n_candidates),
        "n_pool": len(pool),
        "n_search": sequence.n_search,
        # A late stop counts the members it found; none of them is kept.
        "n_members": len(member_index),
        "n_kept": len(kept_index),
        "modes": [_json_row(row) for row in sequence.modes],
        "sequence": None if sequence.sequence is None else _json_row(sequence.sequence),
        "counts": {reason: counts[reason] for reason in sorted(counts)},
        "candidates": [
            {
                "label": candidate.label,
                "x": float(candidate.x) + origin_x,
                "y": float(candidate.y) + origin_y,
                "half_size": candidate.half_size,
                "status": size.status,
                "size_pix": size.size,
                "rho4": size.rho4,
                "e1": size.e1,
                "e2": size.e2,
                "snr_w": size.snr_w,
                "peak_snr": size.peak_snr,
                "reason": reason,
                "kept": index in kept_index,
            }
            for index, (candidate, size, reason) in enumerate(
                zip(pool, sizes, sequence.reasons, strict=True)
            )
        ],
    }

    if not found:
        raise PointSourceScreenError(
            screen_stop_message(record),
            code=sequence.outcome,
            record=record,
            finding=sequence.finding,
        )

    members = tuple(pool[i] for i in member_index)

    return ScreenResult(
        members=members, kept=members[: int(target_n_candidates)], record=record
    )


def _json_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a table row with every float finite or None."""
    result = {}

    for key, value in row.items():
        if isinstance(value, Mapping):
            result[key] = _json_row(value)
        elif isinstance(value, (bool, np.bool_)):
            result[key] = bool(value)
        elif isinstance(value, (int, np.integer)):
            result[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            result[key] = _finite_or_none(value)
        else:
            result[key] = value

    return result


def screen_summary(
    record: Mapping[str, Any], *, pixel_scale: float | None = None
) -> dict[str, Any]:
    """Return the summary of a screen record that the advisor reports.

    Parameters
    ----------
    record : mapping
        A record of :func:`screen_psf_candidates` whose outcome is
        ``"found"``.
    pixel_scale : float, optional
        Pixel scale in arcsec per pixel, to give the size of the sequence
        on the sky.

    Returns
    -------
    dict
        The rule and its digest, the counts of the pool, of the members and
        of the kept candidates, the number of stars sought, the size of the
        sequence, its scatter, the count of its core and its tail
        probability, and the rejections by reason.
    """
    sequence = record.get("sequence") or {}
    size_pix = sequence.get("size_pix")
    rejected = {
        reason: count
        for reason, count in record.get("counts", {}).items()
        if reason != "member"
    }

    return {
        "rule": record["rule"],
        "constants_sha256": record["constants_sha256"],
        "n_pool": int(record["n_pool"]),
        "n_members": int(record["n_members"]),
        "n_kept": int(record["n_kept"]),
        "target_n_candidates": record.get("target_n_candidates"),
        "sequence_size_pix": size_pix,
        "sequence_size_arcsec": (
            None
            if size_pix is None or pixel_scale is None
            else float(size_pix) * float(pixel_scale)
        ),
        "tau_hat": sequence.get("tau_hat"),
        "n_core": sequence.get("n_core"),
        "p_tail": sequence.get("p_tail"),
        "rejected": rejected,
    }
