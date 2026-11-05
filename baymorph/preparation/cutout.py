"""The cutout stage of the preparation: where the sky begins.

The stage measures once, on the whole working region, how far the light of
the galaxy reaches, and reads two radii from it:

- the cutout, the pixels of the likelihood, reaches the radius at which the
  galaxy falls below ``data.cutout.light_threshold`` sky scales, 1 by
  default, plus ``data.cutout.sky_margin`` pixels; a fixed cutout keeps its
  declared side, ``data.cutout.size``, and is checked against that radius;
- the noise region, where the sky level, the sky scale and the sky
  correlation are measured, is the working region beyond the radius at which
  the galaxy falls below ``preparation.noise.light_threshold`` sky scales,
  0.3 by default, outside the masks of the sources.

The measurement is the same in both modes:

1. the sources of the working region are detected on the image less a sky
   plane (:func:`baymorph.preparation.detection.detect`), and the target
   segment is selected, with its warnings
   (:func:`baymorph.preparation.target.select_target_segment`);
2. the annuli are ellipses about the target position, with the axis ratio and
   the position angle of the target segment;
3. the far sky is the working region beyond the last complete annulus, with
   every source masked and its mask grown by :data:`MASK_GROWTH_PIXELS`
   pixels; the plane fitted to it is the reference of the profile, its scale
   is the sky scale of the thresholds, and its inner and outer halves must
   agree;
4. the profile is the clipped mean of each complete annulus less that plane,
   with the other sources masked, and each radius is where the
   non-increasing fit of the profile crosses its threshold
   (:mod:`baymorph.preparation.profile`).

The stage stops, with the remedy, when no annulus about the target is
complete, when the far sky is too small, has no scatter or its halves
disagree, when the masks of the other sources leave the profile no pixel,
when the light of the galaxy does not fall below a threshold within the last
complete annulus (the sky is not reached), when the noise region holds fewer
than :data:`MINIMUM_NOISE_PIXELS` pixels, or when the cutout, centred on the
pixel of the target position, exceeds ``data.cutout.max_size`` or leaves the
working region.

Two limitations follow from the rule. Under a large-scale field of the sky
(AD-052) the far sky may hold galaxy light at the level of the field, which no
check on the far sky can see; every product records it. Every segment but the
target segment is masked in the profile, grown by :data:`MASK_GROWTH_PIXELS`
pixels: the pieces into which the deblending splits the detection of the
target, and the peaks of the noise that the light of its wings lifts above the
threshold (AD-050), carry positive noise or light of the galaxy, so the
profile of the wings comes out low and the radii somewhat small. On synthetic
galaxies of index 4 and 6 the light at the radius of 0.3 sky scales is then
1.1 to 1.2 times that threshold; a galaxy that the deblending splits into many
pieces, such as a flat or clumpy disc, loses more, and the products record
when the detection of the target was split.

Positions are zero-based pixels of the whole image, ``x`` along the columns;
the bounds of a cutout are half-open, like those of a working region.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from baymorph.preparation.detection import (
    PLANE_CLIP_SIGMA,
    PLANE_MAX_ITERATIONS,
    SMOOTHING_SIGMA,
    Detection,
    check_detection_settings,
    detect,
)
from baymorph.preparation.identity import stage_identity
from baymorph.preparation.opening import OpenedObservation
from baymorph.preparation.profile import (
    AGREEMENT_SIGMA,
    BIN_GROWTH,
    CLIP_ITERATIONS,
    CLIP_SIGMA,
    COMPLETE_FRACTION,
    LINEAR_BINS_UNTIL,
    MINIMUM_REFERENCE_PIXELS,
    MINIMUM_SECTOR_PIXELS,
    SECTORS,
    Ellipse,
    Profile,
    ReferenceSky,
    complete_radius,
    ellipse_of_segment,
    elliptical_profile,
    grown,
    monotone_fit,
    monotone_radius,
    reference_sky,
)
from baymorph.preparation.store import StageProduct
from baymorph.preparation.target import Target, TargetSegment, select_target_segment

CUTOUT_STAGE = "cutout"
"""Name of the stage in the product store."""

CUTOUT_VERSION = 1
"""Version of the rules of the stage; raising it changes every identity."""

CUTOUT_LIGHT_THRESHOLD = 1.0
"""Default light of the galaxy at the edge of an automatic cutout, in sky
scales: ``data.cutout.light_threshold``."""

NOISE_LIGHT_THRESHOLD = 0.3
"""Default light of the galaxy below which the noise region begins, in sky
scales: ``preparation.noise.light_threshold``."""

MASK_GROWTH_PIXELS = 20.0
"""Growth, in pixels, of the masks of the sources in the profile, the far sky
and the noise region."""

MINIMUM_NOISE_PIXELS = 100_000
"""Fewest pixels of the noise region. On resampled VST-like noise the kernel
sum of the sky correlation scatters by about 5 per cent at this size."""

DEBLEND_CONTRAST = 0.001
"""Smallest fraction of the flux of a detection that a deblended segment
holds."""

WITHOUT_EFFECT = {
    "data.cutout.initial_factor": (
        "the automatic cutout follows the light of the galaxy, measured once on "
        "the working region, with no preliminary cutout"
    ),
}
"""Keys that 0.6 used and that have no effect here, with the reason."""

FAR_SKY_LIMITATION = (
    "The far sky lies beyond the last complete annulus. Galaxy light left there "
    "at the level of the large-scale field of the sky (AD-052) is invisible to "
    "every check on the far sky, and the profile, the radii and the noise region "
    "are measured against it."
)
"""The limitation that every product of the stage records."""

AUTOMATIC_CUTOUT_LIMITATION = (
    "The automatic cutout holds the galaxy out to where its light falls below the "
    "threshold of the cutout, plus the margin: a model fitted to it is "
    "constrained by the light within it, and the light of an envelope beyond it "
    "does not constrain the model."
)
"""The limitation that the products of an automatic cutout record."""


def rule() -> dict[str, Any]:
    """Return the constants of the rule of the stage, as its records give them."""
    return {
        "far_sky": "beyond the last complete annulus",
        "radius": "where the non-increasing fit of the profile crosses a threshold",
        "mask_growth_pixels": MASK_GROWTH_PIXELS,
        "minimum_noise_pixels": MINIMUM_NOISE_PIXELS,
        "minimum_far_sky_pixels": MINIMUM_REFERENCE_PIXELS,
        "agreement_sigma": AGREEMENT_SIGMA,
        "complete_fraction": COMPLETE_FRACTION,
        "linear_bins_until": LINEAR_BINS_UNTIL,
        "bin_growth": BIN_GROWTH,
        "clip_sigma": CLIP_SIGMA,
        "clip_iterations": CLIP_ITERATIONS,
        "sectors": SECTORS,
        "minimum_sector_pixels": MINIMUM_SECTOR_PIXELS,
        "deblend_contrast": DEBLEND_CONTRAST,
        "smoothing_sigma": SMOOTHING_SIGMA,
        "plane_clip_sigma": PLANE_CLIP_SIGMA,
        "plane_max_iterations": PLANE_MAX_ITERATIONS,
    }


# --- The product --------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Cutout:
    """The product of the cutout stage.

    Parameters
    ----------
    mode : {"auto", "fixed"}
        How the side was set: from the light of the galaxy, or declared.
    xmin, xmax, ymin, ymax : int
        The bounds of the cutout, half-open, in pixels of the whole image.
    origin : tuple of int
        The pixel ``(x, y)`` of the whole image at the first element of the
        arrays of the working region.
    ellipse : Ellipse
        The annuli, about the target position, in pixels of the working
        region.
    detection : Detection
        The sources of the working region.
    selection : TargetSegment
        The target segment and how it was selected.
    reference : ReferenceSky
        The far sky, whose plane is given about the target position.
    profile : Profile
        The light of the galaxy on the annuli, less the plane of the far sky.
    excess : ndarray
        The profile in sky scales, the scale of the far sky.
    fitted : ndarray
        The non-increasing fit of ``excess``; NaN on the annuli that do not
        count.
    cutout_radius : float or None
        The radius at which the fit falls below the threshold of the cutout;
        None only for a fixed cutout of a galaxy whose light does not fall
        below it.
    noise_radius : float
        The radius at which the fit falls below the threshold of the noise
        region.
    noise_pixels : int
        The pixels of the noise region.
    warnings : tuple of str
        What the stage records and shows.
    identity : str
        Identity of the product.
    record : Mapping
        What the stage measured and decided, JSON-serializable.
    """

    mode: str
    xmin: int
    xmax: int
    ymin: int
    ymax: int
    origin: tuple[int, int]
    ellipse: Ellipse
    detection: Detection = field(repr=False)
    selection: TargetSegment = field(repr=False)
    reference: ReferenceSky = field(repr=False)
    profile: Profile = field(repr=False)
    excess: NDArray[np.float64] = field(repr=False)
    fitted: NDArray[np.float64] = field(repr=False)
    cutout_radius: float | None
    noise_radius: float
    noise_pixels: int
    warnings: tuple[str, ...]
    identity: str
    record: Mapping[str, Any] = field(repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        """tuple of int : Shape of the cutout as ``(ny, nx)``."""
        return (self.ymax - self.ymin, self.xmax - self.xmin)

    @property
    def bounds(self) -> dict[str, int]:
        """dict : The bounds of the cutout in pixels of the whole image."""
        return {
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
        }

    def product(self) -> StageProduct:
        """Return the product of the stage, for the product store.

        The profile is saved as arrays and the map of the sources of the
        working region as an image, ``segments``; the pixels follow from the
        file, whose content the identity covers.
        """
        profile = self.profile

        return StageProduct(
            stage=CUTOUT_STAGE,
            version=CUTOUT_VERSION,
            identity=self.identity,
            record=dict(self.record),
            arrays={
                "profile_inner": profile.inner,
                "profile_outer": profile.outer,
                "profile_value": profile.value,
                "profile_error": profile.error,
                "profile_used": profile.used,
                "profile_in_image": profile.in_image,
                "profile_unbounded": profile.unbounded,
                "profile_complete": profile.complete,
                "profile_excess": self.excess,
                "profile_fit": self.fitted,
            },
            images={"segments": self.detection.labels},
        )


# --- The masks of the sky -----------------------------------------------------


def source_mask(labels: ArrayLike, no_data: ArrayLike) -> NDArray[np.bool_]:
    """Return every source with its mask grown, and the pixels without data.

    Parameters
    ----------
    labels : array-like of int
        A map of segments, 0 where there is no source.
    no_data : array-like of bool
        The pixels without data, of the same shape.

    Returns
    -------
    ndarray of bool
        The pixels left out of the far sky and of the noise region: every
        segment grown by :data:`MASK_GROWTH_PIXELS`, and the pixels without
        data, which are not grown.
    """
    sources = np.asarray(labels) > 0
    missing = np.asarray(no_data, dtype=bool)

    if missing.shape != sources.shape:
        raise ValueError(
            f"The pixels without data have the shape {missing.shape}, the map of "
            f"segments {sources.shape}."
        )

    return grown(sources, MASK_GROWTH_PIXELS) | missing


def noise_region(
    masked: ArrayLike, ellipse: Ellipse, radius: float
) -> NDArray[np.bool_]:
    """Return the noise region: the pixels at ``radius`` and beyond, unmasked.

    Parameters
    ----------
    masked : array-like of bool
        The pixels left out, as :func:`source_mask` gives them.
    ellipse : Ellipse
        The annuli, in pixels of the array.
    radius : float
        The radius at which the noise region begins, along the major axis.

    Returns
    -------
    ndarray of bool
        The pixels of the noise region.
    """
    out = np.asarray(masked, dtype=bool)
    radii, _ = ellipse.coordinates(out.shape)

    return (radii >= float(radius)) & ~out


# --- The stage ----------------------------------------------------------------


def resolve_cutout(
    opened: OpenedObservation,
    target: Target,
    *,
    mode: str,
    size: int | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    sky_margin: float = 20.0,
    cutout_threshold: float = CUTOUT_LIGHT_THRESHOLD,
    noise_threshold: float = NOISE_LIGHT_THRESHOLD,
    n_sigma: float = 3.0,
    n_pixels: int = 20,
    n_levels: int = 32,
    without_effect: Sequence[str] = (),
) -> Cutout:
    """Measure where the sky begins and set the cutout, as the cutout stage.

    Parameters
    ----------
    opened : OpenedObservation
        The product of the opening.
    target : Target
        The product of the target stage, checked against the same pixels.
    mode : {"auto", "fixed"}
        ``data.cutout.mode``: the side from the light of the galaxy, or the
        declared side.
    size : int, optional
        ``data.cutout.size``, the side of a fixed cutout, in pixels.
    min_size, max_size : int, optional
        ``data.cutout.min_size`` and ``data.cutout.max_size``, the smallest
        and the largest side of an automatic cutout, in pixels.
    sky_margin : float, default=20.0
        ``data.cutout.sky_margin``, the pixels of sky beyond the radius of the
        cutout.
    cutout_threshold : float, default=CUTOUT_LIGHT_THRESHOLD
        ``data.cutout.light_threshold``, in sky scales.
    noise_threshold : float, default=NOISE_LIGHT_THRESHOLD
        ``preparation.noise.light_threshold``, in sky scales.
    n_sigma, n_pixels, n_levels : float, int, int
        ``preparation.detection``, the settings of the detection.
    without_effect : sequence of str, default=()
        Keys of :data:`WITHOUT_EFFECT` that the configuration declares; each
        is warned and recorded.

    Returns
    -------
    Cutout
        The cutout, the two radii, the measurement behind them, the identity
        and the record.

    Raises
    ------
    TypeError
        If ``opened`` or ``target`` is not the product of its stage.
    ValueError
        If a setting is not valid, the target was checked against other
        pixels, no source lies at the target position, or the stage stops
        (see the module docstring).
    """
    if not isinstance(opened, OpenedObservation):
        raise TypeError(
            "resolve_cutout takes the OpenedObservation of the opening stage, not "
            f"{type(opened).__name__}."
        )

    if not isinstance(target, Target):
        raise TypeError(
            "resolve_cutout takes the Target of the target stage, not "
            f"{type(target).__name__}."
        )

    if target.image_identity != opened.image_identity:
        raise ValueError(
            "The target position was checked against other pixels than those of "
            "this opening: run the target stage on this opening."
        )

    settings = _checked_settings(
        mode=mode,
        size=size,
        min_size=min_size,
        max_size=max_size,
        sky_margin=sky_margin,
        cutout_threshold=cutout_threshold,
        noise_threshold=noise_threshold,
        n_sigma=n_sigma,
        n_pixels=n_pixels,
        n_levels=n_levels,
    )
    declared = _checked_without_effect(without_effect)
    image = opened.pixels
    no_data = opened.no_data
    region = opened.region
    origin = (0, 0) if region is None else (region.xmin, region.ymin)
    area = _area(opened)
    remedy = _more_image(opened)

    if settings["mode"] == "fixed":
        # A declared side that cannot fit stops before the detection.
        _check_inside(settings, settings["size"], None, target, opened)

    detection = detect(
        image,
        excluded=no_data,
        n_sigma=settings["detection"]["n_sigma"],
        n_pixels=settings["detection"]["n_pixels"],
        n_levels=settings["detection"]["n_levels"],
        contrast=DEBLEND_CONTRAST,
    )
    selection = select_target_segment(
        detection.labels,
        target.position,
        parents=detection.parents,
        flux=detection.smoothed,
        origin=origin,
    )
    labels = detection.labels
    segment = labels == selection.label
    ellipse = ellipse_of_segment(
        image, segment, x0=target.x - origin[0], y0=target.y - origin[1]
    )
    others = grown((labels > 0) & ~segment, MASK_GROWTH_PIXELS)
    masked = source_mask(labels, no_data)
    far_radius = complete_radius(image.shape, ellipse, no_data=no_data)

    if far_radius <= 0.0:
        raise ValueError(
            f"No annulus about the target position ({target.x:g}, {target.y:g}) is "
            f"complete in {area}: the target lies too close to its edge, or to "
            f"pixels without data, to measure the light of the galaxy. {remedy}"
        )

    far_pixels = int(np.count_nonzero(noise_region(masked, ellipse, far_radius)))

    if far_pixels < MINIMUM_REFERENCE_PIXELS:
        raise ValueError(
            f"The far sky, the pixels of {area} beyond the last complete annulus, "
            f"{far_radius:.0f} pixels from the target position along the major "
            f"axis, holds {far_pixels} pixels outside the masks of the sources, "
            f"fewer than the {MINIMUM_REFERENCE_PIXELS} that its plane needs. "
            f"{remedy}"
        )

    reference = reference_sky(image, masked, ellipse, radius=far_radius)
    _check_far_sky(reference, far_radius, remedy)
    profile = elliptical_profile(
        image, others, ellipse, plane=reference.plane, no_data=no_data
    )
    excess = profile.value / reference.scale
    fitted = monotone_fit(profile, excess)

    if not np.any(np.isfinite(fitted)):
        raise ValueError(
            "The profile of the galaxy holds no pixel on its complete annuli: the "
            f"masks of the other sources, grown by {MASK_GROWTH_PIXELS:g} pixels, "
            "cover them. The deblending split the detection of the target into "
            f"{selection.siblings} segments, and every one but the target segment "
            "is masked as another source. If they are parts of the galaxy, let the "
            "deblending split it less, with a lower preparation.detection.n_levels "
            "or a larger n_pixels."
        )

    cutout_radius = monotone_radius(profile, excess, settings["cutout_threshold"])
    noise_radius = monotone_radius(profile, excess, settings["noise_threshold"])

    if noise_radius is None:
        raise ValueError(
            _sky_not_reached(
                "noise region",
                "preparation.noise.light_threshold",
                settings["noise_threshold"],
                far_radius,
                fitted,
                area,
                remedy,
            )
        )

    if settings["mode"] == "auto" and cutout_radius is None:
        raise ValueError(
            _sky_not_reached(
                "automatic cutout",
                "data.cutout.light_threshold",
                settings["cutout_threshold"],
                far_radius,
                fitted,
                area,
                remedy,
            )
        )

    noise_pixels = int(np.count_nonzero(noise_region(masked, ellipse, noise_radius)))

    if noise_pixels < MINIMUM_NOISE_PIXELS:
        raise ValueError(
            f"The noise region, the pixels of {area} beyond {noise_radius:.0f} "
            "pixels from the target position along the major axis, where the "
            f"galaxy falls below {settings['noise_threshold']:g} sky scales "
            f"(preparation.noise.light_threshold), holds {noise_pixels} pixels "
            f"outside the masks of the sources, fewer than the "
            f"{MINIMUM_NOISE_PIXELS} that the sky scale and the sky correlation "
            f"need. {remedy}"
        )

    warnings = list(selection.warnings)
    margin = settings["sky_margin"]
    needed = (
        None if cutout_radius is None else 2 * math.ceil(cutout_radius + margin) + 1
    )

    if settings["mode"] == "auto":
        side = (
            needed
            if settings["min_size"] is None
            else max(needed, settings["min_size"])
        )

        if settings["max_size"] is not None and needed > settings["max_size"]:
            raise ValueError(
                f"The automatic cutout needs a side of {needed} pixels: the galaxy "
                f"falls below {settings['cutout_threshold']:g} sky scales "
                f"{cutout_radius:.1f} pixels from the target position along the "
                f"major axis, and the margin adds {margin:g} pixels. "
                f"data.cutout.max_size allows {settings['max_size']}: raise it, "
                "or declare a fixed cutout, data.cutout.size."
            )
    else:
        side = settings["size"]

        if needed is None:
            warnings.append(
                "The light of the galaxy does not fall below "
                f"{settings['cutout_threshold']:g} sky scales "
                "(data.cutout.light_threshold) within the last complete annulus, so "
                f"the fixed cutout of {side} pixels is not checked against it."
            )
        elif side < needed:
            warnings.append(
                f"The fixed cutout of {side} pixels is smaller than the {needed} "
                "pixels that hold the galaxy out to where it falls below "
                f"{settings['cutout_threshold']:g} sky scales, {cutout_radius:.1f} "
                "pixels from the target position along the major axis, plus the "
                f"margin of {margin:g} pixels: it cuts light of the galaxy above "
                "that level."
            )

    xmin, xmax, ymin, ymax = _check_inside(settings, side, needed, target, opened)

    for key in declared:
        warnings.append(
            f"{key} has no effect: {WITHOUT_EFFECT[key]}. Remove it from the "
            "configuration; a later version will refuse it."
        )

    rows = slice(ymin - origin[1], ymax - origin[1])
    columns = slice(xmin - origin[0], xmax - origin[0])
    identity = stage_identity(
        stage=CUTOUT_STAGE,
        version=CUTOUT_VERSION,
        settings={**settings, "rule": rule(), "without_effect": list(declared)},
        upstream={"image": opened.image_identity, "target": target.identity},
    )
    record = _record(
        settings=settings,
        declared=declared,
        opened=opened,
        target=target,
        origin=origin,
        bounds={"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
        side=side,
        needed=needed,
        no_data_in_cutout=int(np.count_nonzero(no_data[rows, columns])),
        detection=detection,
        selection=selection,
        ellipse=ellipse,
        reference=reference,
        profile=profile,
        fitted=fitted,
        cutout_radius=cutout_radius,
        noise_radius=noise_radius,
        noise_pixels=noise_pixels,
        far_pixels=far_pixels,
        warnings=warnings,
    )

    for array in (excess, fitted):
        array.flags.writeable = False

    return Cutout(
        mode=settings["mode"],
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        origin=origin,
        ellipse=ellipse,
        detection=detection,
        selection=selection,
        reference=reference,
        profile=profile,
        excess=excess,
        fitted=fitted,
        cutout_radius=cutout_radius,
        noise_radius=noise_radius,
        noise_pixels=noise_pixels,
        warnings=tuple(warnings),
        identity=identity,
        record=record,
    )


def cutout_from_settings(
    settings: Any, opened: OpenedObservation, target: Target
) -> Cutout:
    """Run the cutout stage with the settings of a configuration.

    Parameters
    ----------
    settings : PreparationSettings
        The configuration as :func:`baymorph.preparation.schema.
        read_preparation_settings` reads it.
    opened : OpenedObservation
        The product of the opening.
    target : Target
        The product of the target stage.

    Returns
    -------
    Cutout
        As :func:`resolve_cutout`; a key of :data:`WITHOUT_EFFECT` that the
        configuration declares is warned and recorded.
    """
    declared = tuple(
        key for key in WITHOUT_EFFECT if settings.sources.get(key) == "user"
    )

    return resolve_cutout(
        opened,
        target,
        mode=settings.get("data.cutout.mode"),
        size=settings.get("data.cutout.size"),
        min_size=settings.get("data.cutout.min_size"),
        max_size=settings.get("data.cutout.max_size"),
        sky_margin=settings.get("data.cutout.sky_margin"),
        cutout_threshold=settings.get("data.cutout.light_threshold"),
        noise_threshold=settings.get("preparation.noise.light_threshold"),
        n_sigma=settings.get("preparation.detection.n_sigma"),
        n_pixels=settings.get("preparation.detection.n_pixels"),
        n_levels=settings.get("preparation.detection.n_levels"),
        without_effect=declared,
    )


# --- Checks and messages ------------------------------------------------------


def _checked_settings(
    *,
    mode: Any,
    size: Any,
    min_size: Any,
    max_size: Any,
    sky_margin: Any,
    cutout_threshold: Any,
    noise_threshold: Any,
    n_sigma: Any,
    n_pixels: Any,
    n_levels: Any,
) -> dict[str, Any]:
    """Return the settings of the stage, checked, in canonical form."""
    if mode not in ("auto", "fixed"):
        raise ValueError(f"data.cutout.mode is 'auto' or 'fixed'; got {mode!r}.")

    sides = {"size": size, "min_size": min_size, "max_size": max_size}

    for name, value in sides.items():
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        ):
            raise ValueError(
                f"data.cutout.{name} must be null or a positive integer; got "
                f"{value!r}."
            )

    if mode == "fixed" and size is None:
        raise ValueError(
            "A fixed cutout needs data.cutout.size in pixels; or set "
            "data.cutout.mode: auto."
        )

    if min_size is not None and max_size is not None and min_size > max_size:
        raise ValueError("data.cutout.min_size must not exceed data.cutout.max_size.")

    if _number(sky_margin, "data.cutout.sky_margin") < 0.0:
        raise ValueError(
            f"data.cutout.sky_margin must be at least 0 pixels; got {sky_margin!r}."
        )

    for name, value in (
        ("data.cutout.light_threshold", cutout_threshold),
        ("preparation.noise.light_threshold", noise_threshold),
    ):
        if not _number(value, name) > 0.0:
            raise ValueError(f"{name} must be greater than 0; got {value!r}.")

    check_detection_settings(n_sigma, n_pixels, n_levels, DEBLEND_CONTRAST)

    return {
        "mode": mode,
        **{
            name: None if value is None else int(value) for name, value in sides.items()
        },
        "sky_margin": float(sky_margin),
        "cutout_threshold": float(cutout_threshold),
        "noise_threshold": float(noise_threshold),
        "detection": {
            "n_sigma": float(n_sigma),
            "n_pixels": int(n_pixels),
            "n_levels": int(n_levels),
        },
    }


def _number(value: Any, name: str) -> float:
    """Return a finite number, or refuse it."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number; got {value!r}.")

    return float(value)


def _checked_without_effect(keys: Sequence[str]) -> tuple[str, ...]:
    """Return the declared keys without effect, sorted, or refuse an unknown one."""
    if isinstance(keys, str):
        raise ValueError("without_effect is a sequence of keys, not one string.")

    unknown = sorted(str(key) for key in keys if key not in WITHOUT_EFFECT)

    if unknown:
        raise ValueError(
            f"Keys without effect must be among {sorted(WITHOUT_EFFECT)}; got "
            f"{unknown}."
        )

    return tuple(sorted(set(keys)))


def _area(opened: OpenedObservation) -> str:
    """Return how a message names the pixels the stage measures."""
    region = opened.region

    if region is None:
        height, width = opened.pixels.shape

        return f"the image of {width} x {height} pixels"

    return (
        f"the working region x {region.xmin}-{region.xmax}, y "
        f"{region.ymin}-{region.ymax}"
    )


def _more_image(opened: OpenedObservation) -> str:
    """Return the remedy of a stop for want of sky."""
    if opened.region is None:
        return "Use an image that extends further from the galaxy."

    return (
        "Open more of the image about the target, with a larger data.region or "
        "data.region: whole, or use an image that extends further from the galaxy."
    )


def _check_far_sky(reference: ReferenceSky, far_radius: float, remedy: str) -> None:
    """Stop when the far sky cannot serve as the reference of the profile."""
    if not (math.isfinite(reference.scale) and reference.scale > 0.0):
        raise ValueError(
            "The far sky has no scatter about its plane, so the light of the "
            "galaxy cannot be measured in sky scales: its pixels are constant, "
            "which a sky with noise never is. Check the image."
        )

    if not math.isfinite(reference.inner_outer_error):
        raise ValueError(
            f"The far sky beyond {far_radius:.0f} pixels from the target position "
            "does not spread over enough directions about it to compare its inner "
            f"and outer halves. {remedy}"
        )

    if not reference.consistent:
        difference = reference.inner_minus_outer
        error = reference.inner_outer_error

        raise ValueError(
            "The inner and outer halves of the far sky, beyond "
            f"{far_radius:.0f} pixels from the target position, differ by "
            f"{difference:.3g} ± {error:.2g} in the units of the image, "
            f"{abs(difference) / error:.1f} times the error, more than "
            f"{AGREEMENT_SIGMA:g}: light of the galaxy or of a large neighbour, or "
            f"a sky that is not a plane, reaches the far sky. {remedy} Check also "
            "the image for a background that the reduction left."
        )


def _sky_not_reached(
    what: str,
    key: str,
    threshold: float,
    far_radius: float,
    fitted: NDArray[np.float64],
    area: str,
    remedy: str,
) -> str:
    """Return the message of a threshold that the light of the galaxy never meets."""
    counted = np.flatnonzero(np.isfinite(fitted))
    last = (
        ""
        if counted.size == 0
        else f" The fitted profile ends there at {fitted[counted[-1]]:.2f} sky scales."
    )

    return (
        "The sky is not reached: the light of the galaxy does not fall below "
        f"{threshold:g} sky scales within the last complete annulus, "
        f"{far_radius:.0f} pixels from the target position along the major axis, "
        f"in {area}.{last} The {what} needs the radius at which it does ({key}). "
        f"{remedy}"
    )


def _check_inside(
    settings: Mapping[str, Any],
    side: int,
    needed: int | None,
    target: Target,
    opened: OpenedObservation,
) -> tuple[int, int, int, int]:
    """Return the bounds of the cutout about the target pixel, or stop.

    The cutout of ``side`` pixels holds ``side // 2`` pixels before the
    target pixel on each axis, so an odd side is centred on it.
    """
    xmin = target.column - side // 2
    ymin = target.row - side // 2
    xmax, ymax = xmin + side, ymin + side
    height, width = opened.pixels.shape
    region = opened.region
    left, bottom = (0, 0) if region is None else (region.xmin, region.ymin)

    if (
        left <= xmin
        and xmax <= left + width
        and bottom <= ymin
        and ymax <= bottom + height
    ):
        return xmin, xmax, ymin, ymax

    raise ValueError(
        _leaves_the_region(
            settings, side, needed, (xmin, xmax, ymin, ymax), target, opened
        )
    )


def _leaves_the_region(
    settings: Mapping[str, Any],
    side: int,
    needed: int | None,
    box: tuple[int, int, int, int],
    target: Target,
    opened: OpenedObservation,
) -> str:
    """Return the message of a cutout that leaves the working region."""
    xmin, xmax, ymin, ymax = box
    height, width = opened.pixels.shape
    region = opened.region
    left, bottom = (0, 0) if region is None else (region.xmin, region.ymin)
    room = min(
        target.column - left,
        left + width - 1 - target.column,
        target.row - bottom,
        bottom + height - 1 - target.row,
    )
    fits = 2 * max(room, 0) + 1
    automatic = settings["mode"] == "auto"
    message = (
        f"The {'automatic' if automatic else 'fixed'} cutout of {side} pixels about "
        f"the pixel ({target.column}, {target.row}) of the target position, x "
        f"{xmin}-{xmax} and y {ymin}-{ymax} of the whole image, leaves "
        f"{_area(opened)}: at most {fits} pixels fit about the target."
    )

    if not automatic:
        return f"{message} Declare a smaller data.cutout.size. {_more_image(opened)}"

    if needed is not None and needed <= fits:
        return (
            f"{message} data.cutout.min_size sets the side: lower it to at most "
            f"{fits}."
        )

    return (
        f"{message} The galaxy needs {needed} pixels, out to where it falls below "
        f"{settings['cutout_threshold']:g} sky scales plus the margin. "
        f"{_more_image(opened)}"
    )


def _finite(value: float) -> float | None:
    """Return a finite number for a record, or None."""
    return float(value) if math.isfinite(value) else None


def _record(
    *,
    settings: Mapping[str, Any],
    declared: tuple[str, ...],
    opened: OpenedObservation,
    target: Target,
    origin: tuple[int, int],
    bounds: Mapping[str, int],
    side: int,
    needed: int | None,
    no_data_in_cutout: int,
    detection: Detection,
    selection: TargetSegment,
    ellipse: Ellipse,
    reference: ReferenceSky,
    profile: Profile,
    fitted: NDArray[np.float64],
    cutout_radius: float | None,
    noise_radius: float,
    noise_pixels: int,
    far_pixels: int,
    warnings: Sequence[str],
) -> dict[str, Any]:
    """Return the record of the stage."""
    counted = np.flatnonzero(np.isfinite(fitted))
    far_sky = {
        key: (_finite(value) if isinstance(value, float) else value)
        for key, value in reference.as_dict().items()
    }
    far_sky["plane_about"] = "the target position"
    far_sky["pixels_before_clipping"] = far_pixels
    far_sky["error_in_sky_scales"] = _finite(reference.error / reference.scale)
    limitations = [FAR_SKY_LIMITATION]

    if settings["mode"] == "auto":
        limitations.append(AUTOMATIC_CUTOUT_LIMITATION)

    if selection.siblings is not None and selection.siblings > 1:
        limitations.append(
            f"The deblending split the detection of the target into "
            f"{selection.siblings} segments, and the other "
            f"{selection.siblings - 1} are masked in the profile as other sources: "
            "if they are parts of the galaxy, its radii come out small."
        )

    return {
        "stage": CUTOUT_STAGE,
        "version": CUTOUT_VERSION,
        "mode": settings["mode"],
        "settings": dict(settings),
        "without_effect": list(declared),
        "cutout": {
            "bounds": dict(bounds),
            "side": side,
            "needed_side": needed,
            "centre_pixel": {"x": target.column, "y": target.row},
            "no_data_pixels": no_data_in_cutout,
        },
        "radii": {
            "cutout": {
                "key": "data.cutout.light_threshold",
                "threshold": settings["cutout_threshold"],
                "radius": cutout_radius,
            },
            "noise": {
                "key": "preparation.noise.light_threshold",
                "threshold": settings["noise_threshold"],
                "radius": noise_radius,
            },
            "along": "the major axis of the ellipse",
        },
        "noise_region": {
            "radius": noise_radius,
            "pixels": noise_pixels,
            "minimum": MINIMUM_NOISE_PIXELS,
            "masks": (
                f"every source grown by {MASK_GROWTH_PIXELS:g} pixels, and the "
                "pixels without data"
            ),
        },
        "ellipse": {
            "x0": ellipse.x0 + origin[0],
            "y0": ellipse.y0 + origin[1],
            "q": ellipse.q,
            "theta": ellipse.theta,
            "centre": "the target position, in pixels of the whole image",
        },
        "far_sky": far_sky,
        "profile": {
            "annuli": int(np.count_nonzero(profile.complete)),
            "complete_radius": profile.complete_radius,
            "last_fit": None if counted.size == 0 else float(fitted[counted[-1]]),
            "arrays": "profile_* arrays of the product",
        },
        "detection": {
            "segments": detection.count,
            "plane": [_finite(value) for value in detection.plane],
            "noise": _finite(detection.noise),
            "threshold": _finite(detection.threshold),
            "settings": dict(detection.settings),
            "map": "segments image of the product, on the working region",
        },
        "target_segment": selection.as_dict(),
        "working_region": None if opened.region is None else opened.region.as_dict(),
        "origin": {"x": origin[0], "y": origin[1]},
        "target": {
            "position": {"x": target.x, "y": target.y},
            "pixel": {"x": target.column, "y": target.row},
        },
        "rule": rule(),
        "warnings": list(warnings),
        "limitations": limitations,
        "image_identity": opened.image_identity,
        "target_identity": target.identity,
    }


__all__ = [
    "AUTOMATIC_CUTOUT_LIMITATION",
    "CUTOUT_LIGHT_THRESHOLD",
    "CUTOUT_STAGE",
    "CUTOUT_VERSION",
    "DEBLEND_CONTRAST",
    "FAR_SKY_LIMITATION",
    "MASK_GROWTH_PIXELS",
    "MINIMUM_NOISE_PIXELS",
    "NOISE_LIGHT_THRESHOLD",
    "WITHOUT_EFFECT",
    "Cutout",
    "cutout_from_settings",
    "noise_region",
    "resolve_cutout",
    "rule",
    "source_mask",
]
