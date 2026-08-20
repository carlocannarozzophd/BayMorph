"""Evidence-aware observation inspection and configuration guidance."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from baymorph.core.observation import Observation
from baymorph.pipeline.config import DEFAULT_CONFIG
from baymorph.pipeline.config_maker import make_config_from_fits
from baymorph.pipeline.image_preparation import prepare_real_fits_image
from baymorph.pipeline.preparation_steps import (
    estimate_target_geometry_from_image,
)
from baymorph.pipeline.segmentation import select_segment_near_position


InputFunction = Callable[[str], str]
TargetPosition = (
    str
    | Mapping[str, float]
    | Sequence[float]
    | None
)


@dataclass(frozen=True, slots=True)
class PriorGuidance:
    """Evidence-aware guidance for one model parameter.

    Data-derived quantities may be exposed as diagnostic numerical scales.
    They never become scientific-prior information automatically.
    """

    parameter: str
    family: str
    rationale: str
    observed_scale: float | None = None
    observed_scale_source: str | None = None
    observed_scale_role: str | None = None
    use_observed_scale_for_prior: bool = False
    evidence_safe: bool = True


@dataclass(frozen=True, slots=True)
class ObservationInspection:
    """Non-destructive summary of one astronomical FITS observation."""

    path: Path
    shape: tuple[int, int]

    finite_fraction: float
    minimum: float
    maximum: float
    median: float

    background: float
    noise_sigma: float

    pixel_scale: float | None
    seeing_fwhm: float | None
    saturation: float | None
    bunit: str | None

    target_x: float | None
    target_y: float | None

    target_flux_scale: float | None
    target_half_light_radius: float | None
    target_axis_ratio: float | None
    target_position_angle: float | None
    target_area_pixels: int | None


class ObservationAdvisor:
    """Inspect a FITS observation and guide BayMorph configuration.

    Image-derived quantities are diagnostic estimates and numerical scales.
    They are never inserted automatically into scientific priors.

    Configuration generation always delegates to
    :func:`make_config_from_fits`.
    """

    def __init__(
        self,
        fits_path: str | Path,
        *,
        target_position: TargetPosition = None,
    ) -> None:
        self.fits_path = Path(
            fits_path
        )

        if not self.fits_path.exists():
            raise FileNotFoundError(
                f"FITS file not found: {self.fits_path}"
            )

        self.target_position = target_position
        self._inspection: ObservationInspection | None = None

    def inspect(
        self,
    ) -> ObservationInspection:
        """Inspect metadata, image scales, and optional target morphology."""
        observation = Observation.from_fits(
            self.fits_path
        )

        data = np.asarray(
            observation.image.data,
            dtype=float,
        )

        if data.ndim != 2:
            raise ValueError(
                "ObservationAdvisor requires a two-dimensional FITS image."
            )

        finite = np.isfinite(
            data
        )
        finite_values = data[
            finite
        ]

        if finite_values.size == 0:
            raise ValueError(
                "FITS image contains no finite pixels."
            )

        prepared = prepare_real_fits_image(
            data
        )

        noise_sigma = float(
            np.nanmedian(
                prepared.noise
            )
        )

        target_x, target_y = self._resolve_target(
            observation
        )

        target_flux_scale = None
        target_half_light_radius = None
        target_axis_ratio = None
        target_position_angle = None
        target_area_pixels = None

        segmentation = (
            prepared.detection.segmentation
        )

        if (
            target_x is not None
            and segmentation is not None
        ):
            label = select_segment_near_position(
                segmentation,
                x=target_x,
                y=target_y,
            )

            target_segment = (
                segmentation.data
                == label
            )

            target_area_pixels = int(
                np.count_nonzero(
                    target_segment
                )
            )

            geometry = (
                estimate_target_geometry_from_image(
                    prepared.data,
                    target_segment,
                )
            )

            target_axis_ratio = float(
                geometry.q
            )

            target_position_angle = float(
                geometry.theta
            )

            source_signal = np.where(
                target_segment,
                np.clip(
                    prepared.data,
                    0.0,
                    None,
                ),
                0.0,
            )

            flux = float(
                np.sum(
                    source_signal
                )
            )

            if (
                np.isfinite(
                    flux
                )
                and flux > 0.0
            ):
                target_flux_scale = flux

                target_half_light_radius = (
                    _half_light_radius(
                        source_signal,
                        x0=geometry.x0,
                        y0=geometry.y0,
                        q=geometry.q,
                        theta=geometry.theta,
                    )
                )

        metadata = observation.metadata

        inspection = ObservationInspection(
            path=self.fits_path.resolve(),
            shape=(
                int(
                    data.shape[0]
                ),
                int(
                    data.shape[1]
                ),
            ),
            finite_fraction=float(
                np.mean(
                    finite
                )
            ),
            minimum=float(
                np.nanmin(
                    data
                )
            ),
            maximum=float(
                np.nanmax(
                    data
                )
            ),
            median=float(
                np.nanmedian(
                    data
                )
            ),
            background=float(
                prepared.background
            ),
            noise_sigma=noise_sigma,
            pixel_scale=_finite_or_none(
                metadata.pixel_scale
            ),
            seeing_fwhm=_finite_or_none(
                metadata.seeing_fwhm
            ),
            saturation=_finite_or_none(
                metadata.saturation
            ),
            bunit=metadata.bunit,
            target_x=target_x,
            target_y=target_y,
            target_flux_scale=target_flux_scale,
            target_half_light_radius=target_half_light_radius,
            target_axis_ratio=target_axis_ratio,
            target_position_angle=target_position_angle,
            target_area_pixels=target_area_pixels,
        )

        self._inspection = inspection

        return inspection

    def prior_guidance(
        self,
        model_name: str = "single_sersic",
    ) -> dict[str, PriorGuidance]:
        """Return evidence-aware prior guidance.

        Image-derived estimates are reported only as diagnostic or numerical
        scales. They never define scientific-prior bounds automatically.
        """
        if model_name != "single_sersic":
            raise NotImplementedError(
                "ObservationAdvisor prior guidance currently supports "
                "only 'single_sersic'."
            )

        inspection = (
            self._inspection
            or self.inspect()
        )

        image_scale = {
            "observed_scale_source": "fitted_image",
            "observed_scale_role": "diagnostic_numerical_scale",
            "use_observed_scale_for_prior": False,
        }

        return {
            "flux": PriorGuidance(
                parameter="flux",
                family="log_uniform",
                observed_scale=inspection.target_flux_scale,
                rationale=(
                    "Flux is a positive scale parameter. "
                    "The segmented positive-flux scale may guide numerical "
                    "parameterization, but scientific prior bounds must be "
                    "declared independently when Bayesian evidence is used."
                ),
                **image_scale,
            ),
            "re": PriorGuidance(
                parameter="re",
                family="positive_scale",
                observed_scale=inspection.target_half_light_radius,
                rationale=(
                    "Effective radius is positive. "
                    "The segmented half-light radius is a diagnostic scale, "
                    "not an automatically inferred scientific prior."
                ),
                **image_scale,
            ),
            "n": PriorGuidance(
                parameter="n",
                family="bounded",
                rationale=(
                    "Use an explicit bounded morphology prior. "
                    "Do not derive its bounds from a preliminary fit "
                    "to the same image."
                ),
            ),
            "q": PriorGuidance(
                parameter="q",
                family="bounded_0_1",
                observed_scale=inspection.target_axis_ratio,
                rationale=(
                    "Axis ratio has physical support 0 < q <= 1. "
                    "The image-moment estimate is diagnostic only."
                ),
                **image_scale,
            ),
            "theta": PriorGuidance(
                parameter="theta",
                family="uniform_axial",
                observed_scale=inspection.target_position_angle,
                rationale=(
                    "Position angle is axial: theta == theta + pi. "
                    "Use a complete declared axial domain rather than "
                    "data-derived bounds."
                ),
                **image_scale,
            ),
            "x0": PriorGuidance(
                parameter="x0",
                family="uniform_relative_to_target",
                rationale=(
                    "Centre priors may be expressed relative to the "
                    "explicit target-identification seed."
                ),
            ),
            "y0": PriorGuidance(
                parameter="y0",
                family="uniform_relative_to_target",
                rationale=(
                    "Centre priors may be expressed relative to the "
                    "explicit target-identification seed."
                ),
            ),
        }

    def summary(
        self,
    ) -> str:
        """Return a concise human-readable inspection summary."""
        inspection = (
            self._inspection
            or self.inspect()
        )

        lines = [
            "BayMorph Observation Advisor",
            "",
            "OBSERVATION",
            f"FITS: {inspection.path}",
            (
                "Shape: "
                f"{inspection.shape[1]} x "
                f"{inspection.shape[0]} pixels"
            ),
            f"Finite pixels: {inspection.finite_fraction:.3%}",
            (
                "Pixel scale: "
                + _format_optional(
                    inspection.pixel_scale,
                    "arcsec/pixel",
                )
            ),
            (
                "Seeing FWHM: "
                + _format_optional(
                    inspection.seeing_fwhm,
                    "arcsec",
                )
            ),
            f"Background: {inspection.background:.6g}",
            f"Noise sigma: {inspection.noise_sigma:.6g}",
        ]

        if inspection.target_x is not None:
            radius_text = _format_optional(
                inspection.target_half_light_radius,
                "pixels",
            )

            if (
                inspection.target_half_light_radius is not None
                and inspection.pixel_scale is not None
            ):
                radius_arcsec = (
                    inspection.target_half_light_radius
                    * inspection.pixel_scale
                )
                radius_text += (
                    f" ({radius_arcsec:.6g} arcsec)"
                )

            theta_text = "unavailable"

            if inspection.target_position_angle is not None:
                theta_degrees = (
                    np.degrees(
                        inspection.target_position_angle
                    )
                    % 180.0
                )
                theta_text = f"{theta_degrees:.6g} deg"

            lines.extend(
                [
                    "",
                    "TARGET DIAGNOSTICS",
                    (
                        "Target seed: "
                        f"({inspection.target_x:.3f}, "
                        f"{inspection.target_y:.3f})"
                    ),
                    (
                        "Segmented positive-flux scale: "
                        + _format_optional(
                            inspection.target_flux_scale,
                            (
                                inspection.bunit
                                or "image units"
                            ),
                        )
                    ),
                    (
                        "Segmented half-light radius scale: "
                        + radius_text
                    ),
                    (
                        "Axis-ratio moment estimate: "
                        + _format_optional(
                            inspection.target_axis_ratio,
                            "",
                        )
                    ),
                    (
                        "Position-angle moment estimate: "
                        + theta_text
                    ),
                ]
            )

        lines.extend(
            [
                "",
                "EVIDENCE SAFETY",
                (
                    "Target diagnostics are derived from the fitted image."
                ),
                (
                    "Their role is diagnostic and numerical only."
                ),
                (
                    "BayMorph does not use them automatically to define "
                    "scientific prior bounds."
                ),
            ]
        )

        return "\n".join(
            lines
        )

    def make_config(
        self,
        *,
        model_priors: Mapping[str, Any] | None = None,
        **config_options: Any,
    ) -> dict[str, Any]:
        """Create a config using the canonical BayMorph config maker."""
        target_position = config_options.pop(
            "target_position",
            self.target_position,
        )

        if target_position is None:
            raise ValueError(
                "target_position is required to create a configuration."
            )

        return make_config_from_fits(
            self.fits_path,
            target_position=target_position,
            model_priors=model_priors,
            **config_options,
        )

    def configure(
        self,
        *,
        output_path: str | Path | None = None,
        input_fn: InputFunction = input,
        **config_options: Any,
    ) -> dict[str, Any] | None:
        """Interactively create an evidence-safe single-Sersic config.

        The advisor reports image-derived numerical scales but never converts
        them automatically into scientific-prior bounds.

        Configuration choices that would invalidate the standard normalized
        Bayesian-evidence interpretation are rejected unless corrected during
        the interactive dialogue. Expert users who intentionally require such
        configurations may use ``make_config`` or ``make_config_from_fits``
        directly.
        """
        print(
            self.summary()
        )

        answer = input_fn(
            "Create a BayMorph configuration now? [y/N]: "
        ).strip().lower()

        if answer not in {
            "y",
            "yes",
        }:
            return None

        guidance = self.prior_guidance(
            "single_sersic"
        )

        print()
        print("PRIOR PLAN")
        print()

        reference_priors = DEFAULT_CONFIG[
            "fit"
        ][
            "priors"
        ][
            "single_sersic"
        ]

        for parameter in (
            "flux",
            "re",
            "n",
            "q",
            "theta",
            "x0",
            "y0",
            "sky",
        ):
            item = guidance.get(
                parameter
            )

            print(parameter)

            if item is not None:
                print(
                    f"  suggested family: {item.family}"
                )

                if item.observed_scale is not None:
                    if parameter == "theta":
                        observed_text = (
                            f"{np.degrees(item.observed_scale) % 180.0:.6g} deg"
                        )
                    else:
                        observed_text = (
                            f"{item.observed_scale:.6g}"
                        )

                    print(
                        "  observed numerical scale: "
                        f"{observed_text}"
                    )
                    print(
                        "  data used to define prior: NO"
                    )

            if parameter == "flux":
                print(
                    "  prior bounds: USER DECISION"
                )
            else:
                prior = reference_priors[
                    parameter
                ]

                prior_text = (
                    f"{prior['type']}("
                    f"{prior['lower']}, "
                    f"{prior['upper']})"
                )

                if "relative_to" in prior:
                    prior_text += (
                        " relative to "
                        f"{prior['relative_to']}"
                    )

                print(
                    "  BayMorph reference prior: "
                    f"{prior_text}"
                )

            print()

        likelihood_config = config_options.get(
            "likelihood",
            None,
        )

        if isinstance(
            likelihood_config,
            Mapping,
        ):
            include_normalization = (
                likelihood_config.get(
                    "include_normalization",
                    True,
                )
            )

            if include_normalization is False:
                print(
                    "EVIDENCE WARNING"
                )
                print(
                    "The requested likelihood excludes its "
                    "normalization term."
                )
                print(
                    "ObservationAdvisor requires a normalized "
                    "likelihood for standard Bayesian-evidence "
                    "interpretation."
                )

                normalize = input_fn(
                    "Enable likelihood normalization? [Y/n]: "
                ).strip().lower()

                if normalize in {
                    "",
                    "y",
                    "yes",
                }:
                    likelihood_config = dict(
                        likelihood_config
                    )
                    likelihood_config[
                        "include_normalization"
                    ] = True

                    config_options[
                        "likelihood"
                    ] = likelihood_config
                else:
                    raise ValueError(
                        "ObservationAdvisor.configure() requires a "
                        "normalized likelihood. Use make_config() or "
                        "make_config_from_fits() directly only if an "
                        "unnormalized likelihood is scientifically "
                        "intentional."
                    )

        flux_scale = guidance[
            "flux"
        ].observed_scale

        if flux_scale is not None:
            print(
                "Observed target flux scale "
                "(diagnostic only): "
                f"{flux_scale:.6g}"
            )

        prior_type = input_fn(
            "Flux prior type [log_uniform]: "
        ).strip().lower()

        if not prior_type:
            prior_type = "log_uniform"

        if prior_type not in {
            "uniform",
            "log_uniform",
        }:
            raise ValueError(
                "Flux prior type must be "
                "'uniform' or 'log_uniform'."
            )

        lower = _prompt_float(
            input_fn,
            "Flux prior lower bound "
            "(scientific choice): ",
        )

        upper = _prompt_float(
            input_fn,
            "Flux prior upper bound "
            "(scientific choice): ",
        )

        if not lower < upper:
            raise ValueError(
                "Flux prior requires lower < upper."
            )

        if (
            prior_type == "log_uniform"
            and lower <= 0.0
        ):
            raise ValueError(
                "Log-uniform flux prior requires lower > 0."
            )

        model_priors = {
            "flux": {
                "type": prior_type,
                "lower": lower,
                "upper": upper,
            }
        }

        return self.make_config(
            output_path=output_path,
            model_name="single_sersic",
            model_priors=model_priors,
            **config_options,
        )

    def _resolve_target(
        self,
        observation: Observation,
    ) -> tuple[
        float | None,
        float | None,
    ]:
        """Resolve the optional target seed for inspection."""
        target = self.target_position

        if target is None:
            return None, None

        if isinstance(
            target,
            str,
        ):
            if (
                target.strip().lower()
                != "center"
            ):
                raise ValueError(
                    "String target_position must be exactly 'center'."
                )

            return (
                float(
                    observation.metadata.center_x
                ),
                float(
                    observation.metadata.center_y
                ),
            )

        if isinstance(
            target,
            Mapping,
        ):
            if set(
                target
            ) != {
                "x",
                "y",
            }:
                raise ValueError(
                    "Target-position mappings must contain "
                    "exactly 'x' and 'y'."
                )

            return (
                float(
                    target["x"]
                ),
                float(
                    target["y"]
                ),
            )

        if len(
            target
        ) != 2:
            raise ValueError(
                "target_position must contain "
                "exactly two coordinates."
            )

        return (
            float(
                target[0]
            ),
            float(
                target[1]
            ),
        )


def _half_light_radius(
    source_signal: np.ndarray,
    *,
    x0: float,
    y0: float,
    q: float,
    theta: float,
) -> float | None:
    """Estimate an elliptical half-light radius from positive signal."""
    y, x = np.indices(
        source_signal.shape,
        dtype=float,
    )

    dx = x - float(
        x0
    )
    dy = y - float(
        y0
    )

    cosine = np.cos(
        theta
    )
    sine = np.sin(
        theta
    )

    major = (
        cosine * dx
        + sine * dy
    )

    minor = (
        -sine * dx
        + cosine * dy
    )

    safe_q = max(
        float(
            q
        ),
        np.finfo(
            float
        ).eps,
    )

    radius = np.sqrt(
        major**2
        + (
            minor
            / safe_q
        ) ** 2
    )

    signal = np.asarray(
        source_signal,
        dtype=float,
    )

    valid = (
        np.isfinite(
            signal
        )
        & (
            signal > 0.0
        )
    )

    if not np.any(
        valid
    ):
        return None

    order = np.argsort(
        radius[
            valid
        ]
    )

    ordered_radius = radius[
        valid
    ][
        order
    ]

    ordered_signal = signal[
        valid
    ][
        order
    ]

    cumulative = np.cumsum(
        ordered_signal
    )

    half_total = (
        0.5
        * cumulative[-1]
    )

    index = int(
        np.searchsorted(
            cumulative,
            half_total,
            side="left",
        )
    )

    return float(
        ordered_radius[
            min(
                index,
                ordered_radius.size - 1,
            )
        ]
    )


def _finite_or_none(
    value: Any,
) -> float | None:
    """Return a finite float or None."""
    if value is None:
        return None

    result = float(
        value
    )

    if not np.isfinite(
        result
    ):
        return None

    return result


def _format_optional(
    value: float | None,
    unit: str,
) -> str:
    """Format an optional scalar quantity."""
    if value is None:
        return "unavailable"

    suffix = (
        f" {unit}"
        if unit
        else ""
    )

    return (
        f"{value:.6g}"
        f"{suffix}"
    )


def _prompt_float(
    input_fn: InputFunction,
    prompt: str,
) -> float:
    """Read one finite floating-point value."""
    value = float(
        input_fn(
            prompt
        ).strip()
    )

    if not np.isfinite(
        value
    ):
        raise ValueError(
            "Prior bounds must be finite."
        )

    return value
