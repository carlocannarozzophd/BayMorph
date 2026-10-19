"""Priors and observation settings proposed by the observation advisor.

AD-049 lets the advisor propose a configuration whose scientific priors come
only from the header, from physics and from the fixed inputs of the
observation model, never from the intensities of the image being fitted.
This module holds its rules, as functions of those inputs alone:

- a flux takes a range of magnitudes, 8 to 30 by default, converted through
  the zeropoint, read from the header or declared in ``data.calibration``,
  into a log-uniform prior in image units;
- without a usable zeropoint, a flux is log-uniform in image units from the
  flux of a point source at signal-to-noise 1, the prepared sky scale times
  the square root of the noise-equivalent area of the PSF, to as many
  magnitudes above it as the range of magnitudes spans, 22 by default; the
  bounds come from the fixed inputs, never from the flux of the galaxy;
- a length of a model with one extended component takes an angular range,
  0.1 to 120 arcsec by default, converted through the pixel scale into a
  log-uniform prior in pixels and intersected with the validated rendering
  domain;
- a length of a model with several components keeps its reference prior,
  whose ranges tell the components apart, clipped to the same angular range:
  one range for every length would make the components interchangeable;
- the centre lies within three times the FWHM of the PSF of the target
  position;
- dimensionless parameters and angles keep the reference priors of BayMorph;
- the residual background level lies within five times the prepared sky
  scale, and a plane changes by at most as much over half the cutout.

The advisor also proposes the source of the effective gain of AD-047 and the
ceiling on the peak of the stars of the empirical PSF. Every proposal records
where it comes from.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from baymorph.core.effective_gain import SKY_LEVEL_GUARD
from baymorph.core.header_metadata import IMAGE_UNIT, HeaderReading
from baymorph.models.labels import PARAMETER_KINDS, parameter_names
from baymorph.models.validated_domains import validated_domain
from baymorph.observation.background import background_parameter_names
from baymorph.pipeline.config import DEFAULT_CONFIG
from baymorph.pipeline.prior_config import prior_config_for_parameter

MAGNITUDE_RANGE = (8.0, 30.0)
"""Default range of magnitudes of a flux, brightest first."""

POINT_SOURCE_SNR = 1.0
"""Signal-to-noise ratio of the point source whose flux is the lower bound of
a flux prior proposed without a zeropoint."""

ANGULAR_RANGE_ARCSEC = (0.1, 120.0)
"""Default angular range of a length, in arcseconds."""

CENTRE_FWHM_MULTIPLE = 3.0
"""Half-width of the centre prior, in units of the FWHM of the PSF."""

BACKGROUND_SKY_SCALE_MULTIPLE = 5.0
"""Half-width of the residual background prior, in units of the prepared sky
scale: the bound of the level, and the change of a plane over half the
cutout."""

PSF_PEAK_SATURATION_FRACTION = 0.5
"""Fraction of the saturation level proposed as the ceiling on the peak of a
star of the empirical PSF."""

GAIN_RATIO_TOLERANCE = 0.15
"""Relative departure of the ratio of two sources of the effective gain from
its prediction above which the advisor warns."""

PHYSICAL_LENGTH_MODELS = frozenset(
    {
        "single_sersic",
        "boxy_sersic",
        "core_sersic",
        "sersic_point_source",
    }
)
"""Models with one extended component, whose lengths take the angular range.
The lengths of the other models keep the reference ranges that tell their
components apart."""

SERSIC_RADIUS_PARAMETERS = frozenset({"re", "re_bulge", "re_disk"})
"""Effective radii of Sérsic components."""

SERSIC_RADIUS_DOMAIN_PIXELS = 1.0
"""Smallest effective radius, in pixels, for which AD-046 validated the
rendering of a Sérsic profile."""

PRIOR_SOURCES = (
    "header",
    "declaration",
    "fixed_inputs",
    "reference",
    "calibration",
    "user",
)
"""Sources of a proposed prior: the header together with a physical range, a
quantity declared in ``data.calibration`` together with a physical range,
the fixed inputs of the observation model, the reference priors of BayMorph,
the calibration of detectability and saturation, or the user."""

_ADU_UNITS = frozenset({"adu", "count", "counts", "dn"})
_ADU_RATE_UNITS = frozenset(
    {
        "adu/s",
        "adu/sec",
        "adu/second",
        "count/s",
        "counts/s",
        "counts/sec",
        "dn/s",
    }
)


@dataclass(frozen=True)
class PriorProposal:
    """The prior proposed for one parameter, with its provenance.

    Parameters
    ----------
    parameter : str
        Name of the parameter.
    kind : str
        Kind of the parameter, one of
        :data:`baymorph.models.labels.PARAMETER_KIND_NAMES`, or
        ``"background"`` for the residual background.
    prior : dict
        The prior, as a configuration declares it.
    source : str
        One of ``PRIOR_SOURCES``.
    detail : str
        How the prior was obtained.
    """

    parameter: str
    kind: str
    prior: dict[str, Any]
    source: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description of the proposal."""
        return {
            "kind": self.kind,
            "prior": deepcopy(self.prior),
            "source": self.source,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class GainProposal:
    """The proposed source of the effective gain of AD-047.

    Parameters
    ----------
    declaration : dict or None
        The ``fit.likelihood.effective_gain`` block, or None when no source
        qualifies and the user must declare the gain.
    value : float or None
        Gain, in the units of the image, that the declaration gives.
    candidates : dict
        Gain given by each source found, in the order of preference:
        ``sky_level``, ``measured_sky`` and ``gain_keyword``; the gain keyword
        is reported converted to the units of the image but not corrected for
        any resampling.
    ratio : float or None
        Ratio of the gain of the sky to that of the gain keyword, or of the
        two sky sources, when two sources exist.
    expected_ratio : float or None
        The ratio that the declared resampling predicts, when known.
    notes : tuple of str
        Why each source was or was not proposed.
    warnings : tuple of str
        Departures that the user must review.
    """

    declaration: dict[str, Any] | None
    value: float | None
    candidates: dict[str, float]
    ratio: float | None
    expected_ratio: float | None
    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def source(self) -> str | None:
        """Return the proposed source, or None."""
        if self.declaration is None:
            return None

        return str(self.declaration["source"])

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description of the proposal."""
        return {
            "declaration": deepcopy(self.declaration),
            "value": self.value,
            "candidates": dict(self.candidates),
            "ratio": self.ratio,
            "expected_ratio": self.expected_ratio,
            "notes": list(self.notes),
            "warnings": list(self.warnings),
        }


def _positive_finite(value: Any, name: str) -> float:
    """Return ``value`` as a float, or raise unless positive and finite."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number.")

    result = float(value)

    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")

    return result


def _ordered_range(values: Sequence[float], name: str) -> tuple[float, float]:
    """Return a finite range ``(lower, upper)`` with ``lower < upper``."""
    lower, upper = (float(value) for value in values)

    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError(f"{name} must be two finite values in increasing order.")

    return lower, upper


def magnitude_flux_prior(
    zeropoint: float,
    *,
    magnitudes: Sequence[float] = MAGNITUDE_RANGE,
) -> dict[str, Any]:
    """Return the log-uniform flux prior of a range of magnitudes.

    A prior uniform in magnitude is log-uniform in flux. A magnitude ``m``
    corresponds to the flux ``10**(0.4 * (zeropoint - m))`` in the units of
    the image whose zeropoint is given.

    Parameters
    ----------
    zeropoint : float
        Photometric zeropoint of the image units.
    magnitudes : pair of float, default=(8, 30)
        Brightest and faintest magnitude.

    Returns
    -------
    dict
        ``{"type": "log_uniform", "lower": ..., "upper": ...}`` in image
        units.
    """
    point = float(zeropoint)

    if not math.isfinite(point):
        raise ValueError("The zeropoint must be finite.")

    bright, faint = _ordered_range(magnitudes, "magnitudes")

    return {
        "type": "log_uniform",
        "lower": 10.0 ** (0.4 * (point - faint)),
        "upper": 10.0 ** (0.4 * (point - bright)),
    }


def noise_equivalent_area(kernel: Any) -> float:
    """Return the noise-equivalent area of a PSF, in square pixels.

    The area is ``1 / sum(p**2)``, with ``p`` the kernel normalized to unit
    sum. A point source of flux ``F`` measured with the weights of the PSF on
    a sky of scale ``sigma`` per pixel has the signal-to-noise ratio
    ``F / (sigma * sqrt(area))``.

    Parameters
    ----------
    kernel : array_like
        Two-dimensional PSF kernel, finite and non-negative, with a positive
        sum; it need not be normalized.

    Returns
    -------
    float
        The noise-equivalent area, in square pixels.
    """
    array = np.asarray(kernel, dtype=float)

    if array.ndim != 2 or array.size == 0:
        raise ValueError("The PSF kernel must be a non-empty two-dimensional array.")

    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("The PSF kernel must be finite and non-negative.")

    total = float(array.sum())

    if total <= 0.0:
        raise ValueError("The PSF kernel must have a positive sum.")

    return float(1.0 / np.sum((array / total) ** 2))


def point_source_flux_prior(
    sky_scale: float,
    psf_noise_equivalent_area: float,
    *,
    magnitude_span: float = MAGNITUDE_RANGE[1] - MAGNITUDE_RANGE[0],
    snr: float = POINT_SOURCE_SNR,
) -> dict[str, Any]:
    """Return the log-uniform flux prior proposed without a zeropoint.

    The lower bound is the flux of a point source at the signal-to-noise
    ratio ``snr`` on the prepared sky, ``snr * sky_scale *
    sqrt(psf_noise_equivalent_area)``; the upper bound is ``magnitude_span``
    magnitudes brighter, the lower bound times ``10**(0.4 * magnitude_span)``.
    Both come from fixed inputs of the observation model, never from the flux
    of the target.

    Parameters
    ----------
    sky_scale : float
        Prepared sky scale, in image units.
    psf_noise_equivalent_area : float
        Noise-equivalent area of the PSF, in square pixels; see
        :func:`noise_equivalent_area`.
    magnitude_span : float, default=22
        Width of the prior in magnitudes, that of ``MAGNITUDE_RANGE``.
    snr : float, default=1
        Signal-to-noise ratio of the faintest point source.

    Returns
    -------
    dict
        ``{"type": "log_uniform", "lower": ..., "upper": ...}`` in image
        units.
    """
    lower = (
        _positive_finite(snr, "snr")
        * _positive_finite(sky_scale, "sky_scale")
        * math.sqrt(
            _positive_finite(psf_noise_equivalent_area, "psf_noise_equivalent_area")
        )
    )
    span = _positive_finite(magnitude_span, "magnitude_span")

    return {
        "type": "log_uniform",
        "lower": lower,
        "upper": lower * 10.0 ** (0.4 * span),
    }


def length_domain(
    model_name: str,
    parameter: str,
) -> tuple[float | None, float | None]:
    """Return the validated rendering domain of a length, in pixels.

    The domain joins the table of AD-048 for the morphology families and the
    domain of AD-046 for every Sérsic component, whose rendering was
    validated for effective radii of at least 1 px.

    Returns
    -------
    tuple of float or None
        ``(lower, upper)``, with None for an open side.
    """
    lower, upper = validated_domain(model_name).get(parameter, (None, None))

    if parameter in SERSIC_RADIUS_PARAMETERS:
        lower = (
            SERSIC_RADIUS_DOMAIN_PIXELS
            if lower is None
            else max(float(lower), SERSIC_RADIUS_DOMAIN_PIXELS)
        )

    return lower, upper


def _clip_to_domain(
    lower: float,
    upper: float,
    domain: tuple[float | None, float | None],
    *,
    name: str,
) -> tuple[float, float]:
    """Intersect ``[lower, upper]`` with a domain, or raise if empty."""
    domain_lower, domain_upper = domain

    if domain_lower is not None:
        lower = max(lower, float(domain_lower))

    if domain_upper is not None:
        upper = min(upper, float(domain_upper))

    if not lower < upper:
        raise ValueError(
            f"The proposed range of {name} is empty once intersected with its "
            "validated domain; declare its prior."
        )

    return lower, upper


def angular_length_prior(
    pixel_scale: float,
    *,
    angular_range: Sequence[float] = ANGULAR_RANGE_ARCSEC,
    domain: tuple[float | None, float | None] = (None, None),
    name: str = "the length",
) -> dict[str, Any]:
    """Return the log-uniform prior of an angular range, in pixels.

    Parameters
    ----------
    pixel_scale : float
        Pixel scale in arcseconds per pixel.
    angular_range : pair of float, default=(0.1, 120)
        Smallest and largest length, in arcseconds.
    domain : pair of float or None
        Validated domain in pixels, with None for an open side.
    name : str
        Name used in error messages.

    Returns
    -------
    dict
        ``{"type": "log_uniform", "lower": ..., "upper": ...}`` in pixels.
    """
    scale = _positive_finite(pixel_scale, "pixel_scale")
    smallest, largest = _ordered_range(angular_range, "angular_range")

    if smallest <= 0.0:
        raise ValueError("angular_range must be positive.")

    lower, upper = _clip_to_domain(
        smallest / scale,
        largest / scale,
        domain,
        name=name,
    )

    return {
        "type": "log_uniform",
        "lower": lower,
        "upper": upper,
    }


def clipped_reference_length_prior(
    reference: Mapping[str, Any],
    pixel_scale: float,
    *,
    angular_range: Sequence[float] = ANGULAR_RANGE_ARCSEC,
    domain: tuple[float | None, float | None] = (None, None),
    name: str = "the length",
) -> dict[str, Any]:
    """Return a reference length prior clipped to an angular range.

    The type of the reference prior is kept; its bounds, in pixels, are
    intersected with the angular range converted through the pixel scale and
    with the validated domain.
    """
    scale = _positive_finite(pixel_scale, "pixel_scale")
    smallest, largest = _ordered_range(angular_range, "angular_range")
    prior = dict(reference)

    if prior.get("type") not in ("uniform", "log_uniform"):
        raise ValueError(
            f"The reference prior of {name} is not uniform or log-uniform; "
            "declare its prior."
        )

    lower, upper = _clip_to_domain(
        max(float(prior["lower"]), smallest / scale),
        min(float(prior["upper"]), largest / scale),
        domain,
        name=name,
    )
    prior["lower"] = lower
    prior["upper"] = upper

    return prior


def centre_priors(
    fwhm_pixels: float,
    *,
    multiple: float = CENTRE_FWHM_MULTIPLE,
) -> dict[str, dict[str, Any]]:
    """Return the centre priors relative to the target position.

    Parameters
    ----------
    fwhm_pixels : float
        FWHM of the PSF in pixels.
    multiple : float, default=3
        Half-width of each prior in units of the FWHM.

    Returns
    -------
    dict
        Uniform priors of ``x0`` and ``y0`` relative to ``target_x`` and
        ``target_y``.
    """
    half_width = _positive_finite(multiple, "multiple") * _positive_finite(
        fwhm_pixels,
        "fwhm_pixels",
    )

    return {
        axis: {
            "type": "uniform",
            "lower": -half_width,
            "upper": half_width,
            "relative_to": f"target_{axis[0]}",
        }
        for axis in ("x0", "y0")
    }


def background_priors(
    model: str,
    *,
    sky_scale: float,
    cutout_shape: Sequence[int],
    multiple: float = BACKGROUND_SKY_SCALE_MULTIPLE,
) -> dict[str, dict[str, Any]]:
    """Return the residual background priors in units of the sky scale.

    The level lies within ``multiple * sky_scale``; each gradient of a plane
    lies within the value that changes the plane by as much over half the
    cutout along its axis, since the plane is defined about the centre of
    the cutout.

    Parameters
    ----------
    model : str
        Background model, ``"constant"`` or ``"plane"``.
    sky_scale : float
        Prepared sky scale, in image units.
    cutout_shape : pair of int
        Shape ``(rows, columns)`` of the prepared cutout.
    multiple : float, default=5
        Half-width in units of the sky scale.
    """
    bound = _positive_finite(multiple, "multiple") * _positive_finite(
        sky_scale,
        "sky_scale",
    )
    rows, columns = (int(value) for value in cutout_shape)

    if rows < 2 or columns < 2:
        raise ValueError("cutout_shape must hold at least two pixels per axis.")

    half_widths = {
        "background_level": None,
        "background_gradient_x": 0.5 * columns,
        "background_gradient_y": 0.5 * rows,
    }
    priors = {}

    for name in background_parameter_names(model):
        half = half_widths[name]
        limit = bound if half is None else bound / half
        priors[name] = {
            "type": "uniform",
            "lower": -limit,
            "upper": limit,
        }

    return priors


def reference_priors(model_name: str) -> dict[str, dict[str, Any]]:
    """Return a copy of the reference priors of a built-in model."""
    return deepcopy(DEFAULT_CONFIG["fit"]["priors"].get(model_name, {}))


def propose_model_priors(
    model_name: str,
    *,
    zeropoint: float | None,
    pixel_scale: float | None,
    psf_fwhm_pixels: float | None,
    decided: Mapping[str, PriorProposal] | None = None,
    magnitudes: Sequence[float] = MAGNITUDE_RANGE,
    angular_range: Sequence[float] = ANGULAR_RANGE_ARCSEC,
    centre_multiple: float = CENTRE_FWHM_MULTIPLE,
    photometric_system: str | None = None,
    zeropoint_declared: bool = False,
    sky_scale: float | None = None,
    psf_noise_equivalent_area: float | None = None,
) -> dict[str, PriorProposal]:
    """Propose a prior for every parameter of a model.

    Parameters
    ----------
    model_name : str
        Name of the built-in model.
    zeropoint : float or None
        Usable zeropoint, from the header or declared. Without it, every flux
        not in ``decided`` takes the prior of
        :func:`point_source_flux_prior`, which needs ``sky_scale`` and
        ``psf_noise_equivalent_area``.
    pixel_scale : float or None
        Pixel scale in arcseconds per pixel. Without it a length keeps its
        reference prior in pixels, and the proposal says so.
    psf_fwhm_pixels : float or None
        FWHM of the PSF in pixels, required for the centre unless both
        coordinates are in ``decided``.
    decided : mapping of str to PriorProposal, optional
        Priors already chosen by the user or by a calibration; they are kept
        as they are.
    magnitudes, angular_range, centre_multiple
        The physical ranges of the rules; see the module description.
    photometric_system : str, optional
        Photometric system of the zeropoint, recorded in the provenance.
    zeropoint_declared : bool, default=False
        Whether the zeropoint is declared in ``data.calibration`` rather than
        read from the header; it sets the source of the flux priors.
    sky_scale : float, optional
        Prepared sky scale, in image units, a fixed input.
    psf_noise_equivalent_area : float, optional
        Noise-equivalent area of the PSF, in square pixels, a fixed input.

    Returns
    -------
    dict of str to PriorProposal
        One proposal per parameter, in the order of the model.

    Raises
    ------
    ValueError
        If a prior cannot be proposed, because its input is missing or the
        model has no reference prior for it; the message says what to
        declare.
    """
    if model_name not in PARAMETER_KINDS:
        raise ValueError(f"Unknown model: {model_name}")

    decided = dict(decided or {})
    unknown = sorted(set(decided) - set(parameter_names(model_name)))

    if unknown:
        raise ValueError(
            f"Model {model_name!r} has no parameters {unknown}."
        )

    kinds = PARAMETER_KINDS[model_name]
    references = reference_priors(model_name)
    physical_lengths = model_name in PHYSICAL_LENGTH_MODELS
    proposals: dict[str, PriorProposal] = {}

    for name in parameter_names(model_name):
        kind = kinds[name]

        if name in decided:
            proposals[name] = decided[name]
            continue

        if kind == "flux":
            proposals[name] = _flux_proposal(
                name,
                zeropoint=zeropoint,
                magnitudes=magnitudes,
                photometric_system=photometric_system,
                zeropoint_declared=zeropoint_declared,
                sky_scale=sky_scale,
                psf_noise_equivalent_area=psf_noise_equivalent_area,
            )
            continue

        if kind == "length":
            if pixel_scale is None:
                if name not in references:
                    raise ValueError(
                        f"The pixel scale is unknown and BayMorph has no "
                        f"reference prior for {name!r} of {model_name!r}; "
                        "declare it."
                    )

                proposals[name] = PriorProposal(
                    parameter=name,
                    kind=kind,
                    prior=references[name],
                    source="reference",
                    detail=(
                        "reference prior in pixels: the header gives no pixel "
                        "scale to convert an angular range"
                    ),
                )
                continue

            smallest, largest = _ordered_range(angular_range, "angular_range")
            domain = length_domain(model_name, name)
            domain_text = (
                ""
                if domain == (None, None)
                else " within the validated rendering domain"
            )

            if physical_lengths:
                prior = angular_length_prior(
                    pixel_scale,
                    angular_range=angular_range,
                    domain=domain,
                    name=repr(name),
                )
                source = "header"
                detail = (
                    f"{smallest:g} to {largest:g} arcsec at "
                    f"{float(pixel_scale):.6g} arcsec/px, log-uniform in "
                    f"pixels{domain_text}"
                )
            else:
                if name not in references:
                    raise ValueError(
                        f"BayMorph has no reference prior for {name!r} of "
                        f"{model_name!r}; declare it."
                    )

                prior = clipped_reference_length_prior(
                    references[name],
                    pixel_scale,
                    angular_range=angular_range,
                    domain=domain,
                    name=repr(name),
                )
                source = "reference"
                detail = (
                    "reference prior, whose range tells the components of "
                    f"the model apart, clipped to {smallest:g} to {largest:g} "
                    f"arcsec{domain_text}"
                )

            proposals[name] = PriorProposal(
                parameter=name,
                kind=kind,
                prior=prior,
                source=source,
                detail=detail,
            )
            continue

        if kind == "position":
            if psf_fwhm_pixels is None:
                raise ValueError(
                    "The FWHM of the PSF is unknown, so the centre priors "
                    "cannot be proposed; declare them."
                )

            prior = centre_priors(psf_fwhm_pixels, multiple=centre_multiple)[name]
            proposals[name] = PriorProposal(
                parameter=name,
                kind=kind,
                prior=prior,
                source="fixed_inputs",
                detail=(
                    f"within {float(centre_multiple):g} times the PSF FWHM of "
                    f"{float(psf_fwhm_pixels):.4g} px of the target position"
                ),
            )
            continue

        prior, detail = _reference_prior(
            model_name,
            name,
            references,
        )
        proposals[name] = PriorProposal(
            parameter=name,
            kind=kind,
            prior=prior,
            source="reference",
            detail=detail,
        )

    return proposals


def _flux_proposal(
    name: str,
    *,
    zeropoint: float | None,
    magnitudes: Sequence[float],
    photometric_system: str | None,
    zeropoint_declared: bool,
    sky_scale: float | None,
    psf_noise_equivalent_area: float | None,
) -> PriorProposal:
    """Propose the prior of one flux, with the rule and its numbers."""
    bright, faint = _ordered_range(magnitudes, "magnitudes")

    if zeropoint is not None:
        point = float(zeropoint)
        prior = magnitude_flux_prior(point, magnitudes=magnitudes)
        system = f" {photometric_system}" if photometric_system else ""
        origin = "declared zeropoint" if zeropoint_declared else "zeropoint"

        return PriorProposal(
            parameter=name,
            kind="flux",
            prior=prior,
            source="declaration" if zeropoint_declared else "header",
            detail=(
                f"magnitudes {bright:g} to {faint:g}{system} through the "
                f"{origin} {point:.6g}: {prior['lower']:.4g} to "
                f"{prior['upper']:.4g} in image units, log-uniform"
            ),
        )

    if sky_scale is None or psf_noise_equivalent_area is None:
        raise ValueError(
            f"No usable zeropoint is read or declared, and the prior of "
            f"{name!r} from a point source at signal-to-noise "
            f"{POINT_SOURCE_SNR:g} needs the prepared sky scale and the "
            "noise-equivalent area of the PSF. Declare the zeropoint in "
            "data.calibration, or the prior in image units."
        )

    span = faint - bright
    prior = point_source_flux_prior(
        sky_scale, psf_noise_equivalent_area, magnitude_span=span
    )

    return PriorProposal(
        parameter=name,
        kind="flux",
        prior=prior,
        source="fixed_inputs",
        detail=(
            f"no usable zeropoint: from {prior['lower']:.4g}, the flux of a "
            f"point source at signal-to-noise {POINT_SOURCE_SNR:g} (sky scale "
            f"{float(sky_scale):.4g} times the square root of the PSF "
            f"noise-equivalent area {float(psf_noise_equivalent_area):.4g} "
            f"px^2), to {span:g} magnitudes brighter, {prior['upper']:.4g}, "
            "in image units, log-uniform"
        ),
    )


def _reference_prior(
    model_name: str,
    name: str,
    references: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Return the reference prior of a parameter and how it was found.

    A parameter without its own reference prior takes the default that the
    components of its model share, as the fit does: the axis ratios and
    position angles of the two components of ``double_sersic`` take the
    defaults ``q`` and ``theta``.
    """
    try:
        prior = prior_config_for_parameter(
            name,
            references,
        )
    except ValueError:
        raise ValueError(
            f"BayMorph has no reference prior for {name!r} of "
            f"{model_name!r}; declare it."
        ) from None

    detail = (
        "reference prior of BayMorph"
        if name in references
        else "reference prior of BayMorph, the default that the components "
        "of the model share"
    )

    return deepcopy(dict(prior)), detail


def _normalized_unit(value: Any) -> str | None:
    """Return a unit string lower-cased and without spaces, or None."""
    if value is None:
        return None

    return str(value).strip().lower().replace(" ", "")


def _origin(quantity: Any) -> str:
    """Return where a quantity of a header reading comes from."""
    if quantity.status == "declared":
        return "declared in data.calibration"

    return f"{quantity.keyword}, {quantity.convention}"


def _gain_keyword_in_image_units(
    reading: HeaderReading,
) -> tuple[float | None, str]:
    """Return the gain keyword in the units of the image, or a reason."""
    quantity = reading["gain"]
    label = (
        "data.calibration.gain" if quantity.status == "declared" else quantity.keyword
    )

    if not quantity.usable:
        return None, f"the gain keyword is {quantity.status}"

    if _normalized_unit(quantity.unit) not in ("electron/adu", "e-/adu"):
        return None, (
            f"the gain keyword is in {quantity.unit!r}, not electrons per ADU"
        )

    gain = float(quantity.value)

    if not math.isfinite(gain) or gain <= 0.0:
        return None, "the gain keyword is not a positive number"

    unit = _normalized_unit(reading.value("image_unit"))

    if unit in _ADU_UNITS:
        return gain, f"{label} = {gain:.6g} e-/ADU, image in ADU"

    if unit in _ADU_RATE_UNITS:
        exposure = reading["exposure_time"]

        if not exposure.usable or _normalized_unit(exposure.unit) != "s":
            return None, (
                "the image is a rate and its exposure time in seconds is "
                "not established"
            )

        seconds = float(exposure.value)

        if not math.isfinite(seconds) or seconds <= 0.0:
            return None, "the exposure time is not a positive number"

        return gain * seconds, (
            f"{label} = {gain:.6g} e-/ADU times the exposure time "
            f"{seconds:.6g} s, image in ADU/s"
        )

    return None, (
        f"the image unit {reading.value('image_unit')!r} has no rule that "
        "converts a gain in electrons per ADU"
    )


def _resampled(reading: HeaderReading) -> bool:
    """Return whether the header records a resampling or a combination."""
    if reading["resampling_kernel"].status != "missing":
        return True

    combined = reading.value("n_combined")

    return combined is not None and int(combined) > 1


def propose_effective_gain(
    reading: HeaderReading,
    *,
    sky_scale: float,
    measured_sky_level: float | None = None,
    resampling_variance_factor: float | None = None,
    tolerance: float = GAIN_RATIO_TOLERANCE,
) -> GainProposal:
    """Propose the source of the effective gain of AD-047.

    The sources are proposed in this order: the subtracted sky level that
    the header records, when it passes the guard of AD-047; the sky that the
    preparation measured, when it is still in the image; the gain keyword,
    converted to the units of the image, only for an image that was not
    resampled or whose resampling variance factor is declared, the gain then
    divided by that factor. When two sources exist, their ratio is reported,
    with a warning when it departs from the ratio the declared resampling
    predicts by more than ``tolerance``.

    Parameters
    ----------
    reading : HeaderReading
        Reading of the header of the image.
    sky_scale : float
        Prepared sky scale, in image units.
    measured_sky_level : float, optional
        Background level measured by the preparation.
    resampling_variance_factor : float, optional
        Factor by which a resampling of the image multiplied the variance of
        its noise, declared by the user.
    tolerance : float, default=0.15
        Relative departure that triggers a warning.

    Returns
    -------
    GainProposal
        The proposal; its declaration is None when no source qualifies.
    """
    scale = _positive_finite(sky_scale, "sky_scale")
    factor = (
        None
        if resampling_variance_factor is None
        else _positive_finite(
            resampling_variance_factor,
            "resampling_variance_factor",
        )
    )

    if factor is not None and factor > 1.0:
        raise ValueError(
            "resampling_variance_factor must not exceed 1: a resampling "
            "smooths the noise."
        )

    candidates: dict[str, float] = {}
    notes: list[str] = []
    declarations: dict[str, dict[str, Any]] = {}

    sky = reading["sky_level"]

    if not sky.usable:
        notes.append(f"the header sky level is {sky.status}")
    elif sky.unit != IMAGE_UNIT:
        notes.append(
            f"the header sky level is in {sky.unit!r}, not in the units of "
            "the image"
        )
    else:
        level = float(sky.value)

        if math.isfinite(level) and level >= SKY_LEVEL_GUARD * scale:
            candidates["sky_level"] = level / scale**2
            declarations["sky_level"] = {"source": "sky_level", "sky_level": level}
            notes.append(
                f"sky level {level:.6g} ({sky.keyword}, {sky.convention}) over "
                f"the square of the sky scale {scale:.6g}"
            )
        else:
            notes.append(
                f"the sky level {level:.6g} of the header is below "
                f"{SKY_LEVEL_GUARD:g} times the sky scale"
            )

    if measured_sky_level is not None:
        level = float(measured_sky_level)

        if math.isfinite(level) and level >= SKY_LEVEL_GUARD * scale:
            candidates["measured_sky"] = level / scale**2
            declarations["measured_sky"] = {"source": "measured_sky"}
            notes.append(
                f"measured sky level {level:.6g} over the square of the sky "
                f"scale {scale:.6g}"
            )
        else:
            notes.append(
                "the measured sky level is below "
                f"{SKY_LEVEL_GUARD:g} times the sky scale: the sky was "
                "subtracted before the image reached BayMorph"
            )

    keyword_gain, keyword_note = _gain_keyword_in_image_units(reading)
    resampled = _resampled(reading)

    if keyword_gain is not None:
        candidates["gain_keyword"] = keyword_gain

        if not resampled:
            declarations["gain_keyword"] = {
                "source": "declared",
                "gain": keyword_gain,
            }
            notes.append(keyword_note + "; the header records no resampling")
        elif factor is not None:
            declarations["gain_keyword"] = {
                "source": "declared",
                "gain": keyword_gain / factor,
            }
            notes.append(
                keyword_note
                + f", divided by the declared resampling variance factor "
                f"{factor:.6g}"
            )
        else:
            notes.append(
                keyword_note
                + "; the image was resampled and no resampling variance "
                "factor is declared, so the keyword is not proposed"
            )
    else:
        notes.append(f"gain keyword not used: {keyword_note}")

    ratio = None
    expected = None
    warnings: list[str] = []

    sky_gain = candidates.get("sky_level", candidates.get("measured_sky"))

    if sky_gain is not None and "gain_keyword" in candidates:
        ratio = sky_gain / candidates["gain_keyword"]

        if not resampled:
            expected = 1.0
        elif factor is not None:
            expected = 1.0 / factor
    elif "sky_level" in candidates and "measured_sky" in candidates:
        ratio = candidates["sky_level"] / candidates["measured_sky"]
        expected = 1.0

    if ratio is not None and expected is not None:
        departure = abs(ratio / expected - 1.0)

        if departure > tolerance:
            warnings.append(
                f"The ratio of the gains of two sources is {ratio:.3g}, "
                f"{departure:.0%} from the {expected:.3g} that the declared "
                "resampling predicts. Check the sky level, the gain keyword "
                "and the resampling before accepting the proposed gain."
            )
    elif ratio is not None:
        notes.append(
            f"ratio of the sky gain to the gain keyword {ratio:.3g}, reported "
            "without a prediction because no resampling variance factor is "
            "declared"
        )

    for source in ("sky_level", "measured_sky", "gain_keyword"):
        if source in declarations:
            declaration = declarations[source]
            value = (
                declaration["gain"]
                if declaration["source"] == "declared"
                else candidates[source]
            )

            return GainProposal(
                declaration=declaration,
                value=value,
                candidates=candidates,
                ratio=ratio,
                expected_ratio=expected,
                notes=tuple(notes),
                warnings=tuple(warnings),
            )

    warnings.append(
        "No source of the effective gain qualifies. Declare the gain, or "
        "declare the source variance absent, before running the fit."
    )

    return GainProposal(
        declaration=None,
        value=None,
        candidates=candidates,
        ratio=ratio,
        expected_ratio=expected,
        notes=tuple(notes),
        warnings=tuple(warnings),
    )


def propose_psf_max_peak(
    reading: HeaderReading,
    *,
    fraction: float = PSF_PEAK_SATURATION_FRACTION,
) -> tuple[float | None, str]:
    """Propose the ceiling on the peak of the stars of the empirical PSF.

    The ceiling is a fraction of the saturation level, proposed only when
    the unit of that level against the image is established.

    Returns
    -------
    tuple
        The ceiling in image units, or None, and how it was obtained or why
        it was not.
    """
    quantity = reading["saturation"]

    if not quantity.usable:
        note = f": {quantity.note}" if quantity.note else ""

        return None, f"not proposed: the saturation level is {quantity.status}{note}"

    if quantity.unit != IMAGE_UNIT:
        return None, (
            f"not proposed: the saturation level is in {quantity.unit!r}, not "
            "in the units of the image"
        )

    level = float(quantity.value)

    if not math.isfinite(level) or level <= 0.0:
        return None, "not proposed: the saturation level is not positive"

    share = _positive_finite(fraction, "fraction")

    return share * level, (
        f"{share:g} times the saturation level {level:.6g} ({_origin(quantity)})"
    )
