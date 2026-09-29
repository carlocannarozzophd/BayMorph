"""Prior-predictive diagnostics for BayMorph morphology models.

This module evaluates observable properties of model images generated from
declared scientific priors. Image-derived measurements are never used to
modify those priors.

The current implementation samples the latent prior predictive distribution
of the expected image. It does not add stochastic observational-noise
realizations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from baymorph.fitting.model_factories import (
    make_single_sersic_psf_model,
)
from baymorph.pipeline.prior_config import parse_prior_config


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_SINGLE_SERSIC_PARAMETERS = (
    "flux",
    "re",
    "n",
    "q",
    "theta",
    "x0",
    "y0",
)


@dataclass(frozen=True, slots=True)
class PriorPredictiveReport:
    """Diagnostics from latent single-Sersic prior-predictive draws."""

    parameter_samples: Mapping[str, FloatArray]

    peak_surface_brightness: FloatArray
    frame_peak_surface_brightness: FloatArray
    frame_flux_fraction: FloatArray
    usable_flux_fraction: FloatArray
    re_to_half_frame: FloatArray
    peak_snr: FloatArray
    quadrature_snr: FloatArray

    render_valid: BoolArray

    @property
    def n_draws(self) -> int:
        """Return the number of requested prior draws."""
        return int(
            self.render_valid.size
        )

    @property
    def n_valid(self) -> int:
        """Return the number of successfully rendered draws."""
        return int(
            np.count_nonzero(
                self.render_valid
            )
        )

    @property
    def valid_fraction(self) -> float:
        """Return the fraction of successfully rendered draws."""
        return float(
            np.mean(
                self.render_valid
            )
        )

    def summary(
        self,
    ) -> str:
        """Return a concise quantitative report without policy judgments."""
        lines = [
            "BayMorph Latent Prior-Predictive Diagnostics",
            "",
            f"Draws: {self.n_draws}",
            f"Valid renders: {self.n_valid} ({self.valid_fraction:.3%})",
            "",
            _quantile_line(
                "Peak surface brightness",
                self.peak_surface_brightness,
            ),
            _quantile_line(
                "Frame peak surface brightness",
                self.frame_peak_surface_brightness,
            ),
            _quantile_line(
                "Frame flux fraction",
                self.frame_flux_fraction,
            ),
            _quantile_line(
                "Usable flux fraction",
                self.usable_flux_fraction,
            ),
            _quantile_line(
                "Re / half-frame",
                self.re_to_half_frame,
            ),
            _quantile_line(
                "Peak S/N",
                self.peak_snr,
            ),
            _quantile_line(
                "Quadrature S/N",
                self.quadrature_snr,
            ),
            "",
            (
                "No SAFE/UNSAFE classification is applied. "
                "Interpretation requires an explicit scientific policy."
            ),
        ]

        return "\n".join(
            lines
        )


def evaluate_single_sersic_prior_predictive(
    *,
    priors: Mapping[str, Mapping[str, Any]],
    image_shape: tuple[int, int],
    psf: Any,
    noise_sigma: float | FloatArray,
    references: Mapping[str, float] | None = None,
    mask: BoolArray | None = None,
    n_draws: int = 500,
    random_state: int | None = 0,
) -> PriorPredictiveReport:
    """Evaluate latent prior-predictive diagnostics for single-Sersic models.

    Parameters
    ----------
    priors : mapping
        Resolved prior configuration for all single-Sersic parameters.
    image_shape : tuple of int
        Prepared image shape as ``(ny, nx)``.
    psf : array-like
        PSF kernel used by the production single-Sersic model factory.
    noise_sigma : float or ndarray
        Positive observational-noise scale used only to express signal
        diagnostics in noise units.
    references : mapping, optional
        Deterministic reference coordinates required by relative priors.
    mask : ndarray of bool, optional
        Pixels excluded from inference. ``True`` means excluded.
    n_draws : int, default=500
        Number of independent prior draws.
    random_state : int or None, default=0
        Seed for reproducible unit-cube sampling.

    Returns
    -------
    PriorPredictiveReport
        Quantitative observable-space diagnostics.

    Notes
    -----
    The observation is used only through fixed instrumental or preparation
    information such as image shape, PSF, mask, and noise scale. The function
    never changes the supplied scientific priors.
    """
    shape = _validate_image_shape(
        image_shape
    )

    if not isinstance(
        n_draws,
        int,
    ) or n_draws < 1:
        raise ValueError(
            "n_draws must be a positive integer."
        )

    missing = [
        name
        for name in _SINGLE_SERSIC_PARAMETERS
        if name not in priors
    ]

    if missing:
        raise ValueError(
            "Single-Sersic prior-predictive diagnostics require priors for "
            f"all parameters. Missing: {missing}."
        )

    usable = _resolve_usable_pixels(
        shape,
        mask,
    )

    noise = _resolve_noise(
        noise_sigma,
        shape,
    )

    if not np.any(
        usable
    ):
        raise ValueError(
            "Prior-predictive diagnostics require at least one usable pixel."
        )

    if np.any(
        ~np.isfinite(
            noise[
                usable
            ]
        )
    ) or np.any(
        noise[
            usable
        ]
        <= 0.0
    ):
        raise ValueError(
            "noise_sigma must be finite and positive on usable pixels."
        )

    typed_priors = {
        name: parse_prior_config(
            priors[
                name
            ],
            references=references,
        )
        for name in _SINGLE_SERSIC_PARAMETERS
    }

    rng = np.random.default_rng(
        random_state
    )

    parameter_samples = {
        name: np.asarray(
            [
                prior.transform(
                    float(
                        value
                    )
                )
                for value in rng.random(
                    n_draws
                )
            ],
            dtype=float,
        )
        for name, prior in typed_priors.items()
    }

    peak_surface_brightness = _nan_array(
        n_draws
    )
    frame_peak_surface_brightness = _nan_array(
        n_draws
    )
    frame_flux_fraction = _nan_array(
        n_draws
    )
    usable_flux_fraction = _nan_array(
        n_draws
    )
    re_to_half_frame = _nan_array(
        n_draws
    )
    peak_snr = _nan_array(
        n_draws
    )
    quadrature_snr = _nan_array(
        n_draws
    )

    render_valid = np.zeros(
        n_draws,
        dtype=bool,
    )

    half_frame = (
        0.5
        * min(
            shape
        )
    )

    for index in range(
        n_draws
    ):
        parameters = {
            name: float(
                values[
                    index
                ]
            )
            for name, values in parameter_samples.items()
        }

        re_to_half_frame[
            index
        ] = (
            parameters[
                "re"
            ]
            / half_frame
        )

        try:
            model = make_single_sersic_psf_model(
                **parameters,
                psf=psf,
            )

            source_image = _render_source_components(
                model,
                shape,
            )
        except (
            FloatingPointError,
            OverflowError,
            ValueError,
        ):
            continue

        if (
            source_image.shape
            != shape
            or not np.all(
                np.isfinite(
                    source_image
                )
            )
        ):
            continue

        flux = parameters[
            "flux"
        ]

        source_usable = source_image[
            usable
        ]

        frame_peak_surface_brightness[
            index
        ] = float(
            np.max(
                source_image
            )
        )

        frame_flux_fraction[
            index
        ] = float(
            np.sum(
                source_image
            )
            / flux
        )

        usable_flux_fraction[
            index
        ] = float(
            np.sum(
                source_usable
            )
            / flux
        )

        peak_surface_brightness[
            index
        ] = float(
            np.max(
                source_usable
            )
        )

        signal_to_noise = (
            source_usable
            / noise[
                usable
            ]
        )

        peak_snr[
            index
        ] = float(
            np.max(
                signal_to_noise
            )
        )

        quadrature_snr[
            index
        ] = float(
            np.sqrt(
                np.sum(
                    signal_to_noise**2
                )
            )
        )

        render_valid[
            index
        ] = True

    return PriorPredictiveReport(
        parameter_samples=parameter_samples,
        peak_surface_brightness=peak_surface_brightness,
        frame_peak_surface_brightness=frame_peak_surface_brightness,
        frame_flux_fraction=frame_flux_fraction,
        usable_flux_fraction=usable_flux_fraction,
        re_to_half_frame=re_to_half_frame,
        peak_snr=peak_snr,
        quadrature_snr=quadrature_snr,
        render_valid=render_valid,
    )


def _render_source_components(
    model: Any,
    shape: tuple[int, int],
) -> FloatArray:
    """Render and combine source components without additive background."""
    components = model.render_components(
        shape
    )

    if isinstance(
        components,
        Mapping,
    ):
        arrays = [
            np.asarray(
                component,
                dtype=float,
            )
            for component in components.values()
        ]

        if not arrays:
            raise ValueError(
                "Model returned no source components."
            )

        image = np.sum(
            arrays,
            axis=0,
        )

    elif isinstance(
        components,
        np.ndarray,
    ):
        image = np.asarray(
            components,
            dtype=float,
        )

    else:
        raise TypeError(
            "render_components() must return a mapping of named "
            "component images or a NumPy array."
        )

    return np.asarray(
        image,
        dtype=float,
    )


def _resolve_noise(
    noise_sigma: float | FloatArray,
    shape: tuple[int, int],
) -> FloatArray:
    """Return a full noise map."""
    noise = np.asarray(
        noise_sigma,
        dtype=float,
    )

    if noise.ndim == 0:
        return np.full(
            shape,
            float(
                noise
            ),
            dtype=float,
        )

    if noise.shape != shape:
        raise ValueError(
            "noise_sigma array must match image_shape."
        )

    return noise


def _resolve_usable_pixels(
    shape: tuple[int, int],
    mask: BoolArray | None,
) -> BoolArray:
    """Return the pixels participating in inference."""
    if mask is None:
        return np.ones(
            shape,
            dtype=bool,
        )

    mask_array = np.asarray(
        mask,
        dtype=bool,
    )

    if mask_array.shape != shape:
        raise ValueError(
            "mask must match image_shape."
        )

    return ~mask_array


def _validate_image_shape(
    image_shape: tuple[int, int],
) -> tuple[int, int]:
    """Validate a two-dimensional positive image shape."""
    if (
        len(
            image_shape
        )
        != 2
        or any(
            int(
                value
            )
            != value
            or value <= 0
            for value in image_shape
        )
    ):
        raise ValueError(
            "image_shape must contain two positive integer dimensions."
        )

    return (
        int(
            image_shape[
                0
            ]
        ),
        int(
            image_shape[
                1
            ]
        ),
    )


def _nan_array(
    size: int,
) -> FloatArray:
    """Return one floating array initialized to NaN."""
    return np.full(
        size,
        np.nan,
        dtype=float,
    )


def _quantile_line(
    label: str,
    values: FloatArray,
) -> str:
    """Return robust 5/50/95-percentile text for finite values."""
    finite = np.asarray(
        values,
        dtype=float,
    )

    finite = finite[
        np.isfinite(
            finite
        )
    ]

    if finite.size == 0:
        return f"{label}: unavailable"

    q05, q50, q95 = np.quantile(
        finite,
        [
            0.05,
            0.50,
            0.95,
        ],
    )

    return (
        f"{label}: "
        f"p05={q05:.6g}, "
        f"median={q50:.6g}, "
        f"p95={q95:.6g}"
    )


@dataclass(frozen=True, slots=True)
class FluxPriorCalibration:
    """Joint prior-predictive calibration of total-flux bounds.

    One ensemble of morphology draws is rendered at a fixed positive
    reference flux. Linearity of Sérsic intensity and PSF convolution is then
    used to derive both the detectability lower bound and the detector-
    saturation upper bound without additional image rendering.
    """

    detection_flux: FloatArray
    saturation_flux: FloatArray

    snr_threshold: float
    required_detectable_fraction: float
    required_unsaturated_fraction: float

    saturation_level: float
    subtracted_background: float
    prepared_saturation_level: float

    reference_flux: float
    render_valid_fraction: float

    chosen_flux_lower: float | None
    chosen_flux_upper: float | None

    lower_confidence_interval: tuple[float, float] | None
    upper_confidence_interval: tuple[float, float] | None
    confidence_level: float

    unit: str | None = None

    @property
    def n_draws(self) -> int:
        """Return the number of morphology draws."""
        return int(
            self.detection_flux.size
        )

    @property
    def width_dex(self) -> float | None:
        """Return the calibrated logarithmic prior width."""
        if (
            self.chosen_flux_lower is None
            or self.chosen_flux_upper is None
            or self.chosen_flux_lower <= 0.0
            or self.chosen_flux_upper <= self.chosen_flux_lower
        ):
            return None

        return float(
            np.log10(
                self.chosen_flux_upper
                / self.chosen_flux_lower
            )
        )

    @property
    def candidate_prior(
        self,
    ) -> dict[str, float | str] | None:
        """Return the calibrated log-uniform prior configuration."""
        if self.width_dex is None:
            return None

        return {
            "type": "log_uniform",
            "lower": float(
                self.chosen_flux_lower
            ),
            "upper": float(
                self.chosen_flux_upper
            ),
        }

    def detectable_fraction(
        self,
        flux: float,
    ) -> float:
        """Return the morphology fraction detectable at one total flux."""
        value = float(
            flux
        )

        return float(
            np.mean(
                self.detection_flux
                <= value
            )
        )

    def unsaturated_fraction(
        self,
        flux: float,
    ) -> float:
        """Return the morphology fraction not exceeding saturation."""
        value = float(
            flux
        )

        return float(
            np.mean(
                self.saturation_flux
                >= value
            )
        )

    def summary(self) -> str:
        """Return the complete evidence-oriented calibration summary."""
        unit = (
            ""
            if self.unit is None
            else f" {self.unit}"
        )

        lines = [
            "BayMorph Flux Prior Calibration",
            "",
            f"Morphology draws: {self.n_draws}",
            (
                "Valid morphology renders: "
                f"{self.render_valid_fraction:.3%}"
            ),
            (
                "Numerical reference flux: "
                f"{self.reference_flux:.6g}{unit}"
            ),
            "",
            "LOWER BOUND",
            (
                "Method: prior-predictive detectability"
            ),
            (
                "Detectability metric: quadrature model S/N"
            ),
            f"S/N threshold: {self.snr_threshold:.6g}",
            (
                "Required detectable fraction: "
                f"{self.required_detectable_fraction:.3%}"
            ),
        ]

        if self.chosen_flux_lower is None:
            lines.append(
                "Candidate lower bound: unavailable"
            )
        else:
            lines.append(
                "Candidate lower bound: "
                f"{self.chosen_flux_lower:.6g}{unit}"
            )

        if self.lower_confidence_interval is not None:
            lower, upper = self.lower_confidence_interval

            lines.append(
                (
                    f"{self.confidence_level:.1%} bootstrap interval: "
                    f"[{lower:.6g}, {upper:.6g}]{unit}"
                )
            )

        lines.extend(
            [
                "",
                "UPPER BOUND",
                "Method: detector saturation",
                (
                    "Raw saturation level: "
                    f"{self.saturation_level:.6g}{unit}/pixel"
                ),
                (
                    "Subtracted background: "
                    f"{self.subtracted_background:.6g}{unit}/pixel"
                ),
                (
                    "Prepared saturation level: "
                    f"{self.prepared_saturation_level:.6g}{unit}/pixel"
                ),
                (
                    "Required unsaturated fraction: "
                    f"{self.required_unsaturated_fraction:.3%}"
                ),
            ]
        )

        if self.chosen_flux_upper is None:
            lines.append(
                "Candidate upper bound: unavailable"
            )
        else:
            lines.append(
                "Candidate upper bound: "
                f"{self.chosen_flux_upper:.6g}{unit}"
            )

        if self.upper_confidence_interval is not None:
            lower, upper = self.upper_confidence_interval

            lines.append(
                (
                    f"{self.confidence_level:.1%} bootstrap interval: "
                    f"[{lower:.6g}, {upper:.6g}]{unit}"
                )
            )

        lines.extend(
            [
                "",
                "CANDIDATE PRIOR",
            ]
        )

        if self.candidate_prior is None:
            lines.append(
                "No valid calibrated log-uniform interval."
            )
        else:
            lines.append(
                "flux ~ log_uniform("
                f"{self.chosen_flux_lower:.6g}, "
                f"{self.chosen_flux_upper:.6g})"
            )
            lines.append(
                f"Prior width: {self.width_dex:.3f} dex"
            )

        lines.extend(
            [
                "",
                (
                    "Measured target flux used to define bounds: NO"
                ),
                (
                    "Bounds are conditional on the declared morphology "
                    "priors and observational calibration policy."
                ),
            ]
        )

        return "\n".join(
            lines
        )


def calibrate_single_sersic_flux_prior(
    *,
    priors: Mapping[str, Mapping[str, Any]],
    image_shape: tuple[int, int],
    psf: Any,
    noise_sigma: float | FloatArray,
    saturation_level: float,
    subtracted_background: float,
    references: Mapping[str, float] | None = None,
    mask: BoolArray | None = None,
    snr_threshold: float = 5.0,
    required_detectable_fraction: float = 0.95,
    required_unsaturated_fraction: float = 0.95,
    n_draws: int = 256,
    random_state: int | None = 0,
    bootstrap_samples: int = 512,
    confidence_level: float = 0.95,
    unit: str | None = None,
) -> FluxPriorCalibration:
    """Jointly calibrate single-Sérsic flux-prior bounds.

    The morphology ensemble is rendered exactly once at a positive numerical
    reference flux. The detectability and saturation fluxes are then derived
    analytically from linear flux scaling.

    The lower bound is the smallest total flux at which the requested
    fraction of morphology draws reaches the declared S/N threshold.

    The upper bound is the largest total flux at which the requested fraction
    of morphology draws remains at or below the detector saturation level.

    Bootstrap intervals quantify Monte Carlo sensitivity of the finite
    morphology ensemble. They do not represent uncertainty in the scientific
    calibration policy itself.
    """
    if not isinstance(
        n_draws,
        int,
    ) or n_draws < 1:
        raise ValueError(
            "n_draws must be a positive integer."
        )

    if not isinstance(
        bootstrap_samples,
        int,
    ) or bootstrap_samples < 0:
        raise ValueError(
            "bootstrap_samples must be a non-negative integer."
        )

    if not np.isfinite(
        snr_threshold
    ) or snr_threshold <= 0.0:
        raise ValueError(
            "snr_threshold must be finite and positive."
        )

    if not (
        0.0
        < required_detectable_fraction
        <= 1.0
    ):
        raise ValueError(
            "required_detectable_fraction must lie in (0, 1]."
        )

    if not (
        0.0
        < required_unsaturated_fraction
        <= 1.0
    ):
        raise ValueError(
            "required_unsaturated_fraction must lie in (0, 1]."
        )

    if not (
        0.0
        < confidence_level
        <= 1.0
    ):
        raise ValueError(
            "confidence_level must lie in (0, 1]."
        )

    saturation = float(
        saturation_level
    )

    background = float(
        subtracted_background
    )

    if not np.isfinite(
        saturation
    ) or saturation <= 0.0:
        raise ValueError(
            "saturation_level must be finite and positive."
        )

    if not np.isfinite(
        background
    ):
        raise ValueError(
            "subtracted_background must be finite."
        )

    prepared_saturation = (
        saturation
        - background
    )

    if prepared_saturation <= 0.0:
        raise ValueError(
            "Background-subtracted saturation level must be positive."
        )

    shape = _validate_image_shape(
        image_shape
    )

    usable = _resolve_usable_pixels(
        shape,
        mask,
    )

    noise = _resolve_noise(
        noise_sigma,
        shape,
    )

    usable_noise = noise[
        usable
    ]

    if (
        usable_noise.size == 0
        or np.any(
            ~np.isfinite(
                usable_noise
            )
        )
        or np.any(
            usable_noise <= 0.0
        )
    ):
        raise ValueError(
            "noise_sigma must be finite and positive on usable pixels."
        )

    reference_flux = float(
        np.median(
            usable_noise
        )
        * np.sqrt(
            usable_noise.size
        )
    )

    candidate_priors = {
        name: dict(
            config
        )
        for name, config in priors.items()
    }

    candidate_priors[
        "flux"
    ] = {
        "type": "fixed",
        "value": reference_flux,
    }

    report = evaluate_single_sersic_prior_predictive(
        priors=candidate_priors,
        image_shape=shape,
        psf=psf,
        noise_sigma=noise,
        references=references,
        mask=mask,
        n_draws=n_draws,
        random_state=random_state,
    )

    detection_flux = np.full(
        n_draws,
        np.inf,
        dtype=float,
    )

    saturation_flux = np.zeros(
        n_draws,
        dtype=float,
    )

    detectable_scaling = (
        report.render_valid
        & np.isfinite(
            report.quadrature_snr
        )
        & (
            report.quadrature_snr
            > 0.0
        )
    )

    detection_flux[
        detectable_scaling
    ] = (
        reference_flux
        * snr_threshold
        / report.quadrature_snr[
            detectable_scaling
        ]
    )

    saturation_scaling = (
        report.render_valid
        & np.isfinite(
            report.frame_peak_surface_brightness
        )
        & (
            report.frame_peak_surface_brightness
            > 0.0
        )
    )

    saturation_flux[
        saturation_scaling
    ] = (
        reference_flux
        * prepared_saturation
        / report.frame_peak_surface_brightness[
            saturation_scaling
        ]
    )

    chosen_flux_lower = _detectability_order_statistic(
        detection_flux,
        required_detectable_fraction,
    )

    chosen_flux_upper = _saturation_order_statistic(
        saturation_flux,
        required_unsaturated_fraction,
    )

    (
        lower_confidence_interval,
        upper_confidence_interval,
    ) = _bootstrap_flux_prior_bounds(
        detection_flux=detection_flux,
        saturation_flux=saturation_flux,
        required_detectable_fraction=required_detectable_fraction,
        required_unsaturated_fraction=required_unsaturated_fraction,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        random_state=random_state,
    )

    return FluxPriorCalibration(
        detection_flux=detection_flux,
        saturation_flux=saturation_flux,
        snr_threshold=float(
            snr_threshold
        ),
        required_detectable_fraction=float(
            required_detectable_fraction
        ),
        required_unsaturated_fraction=float(
            required_unsaturated_fraction
        ),
        saturation_level=saturation,
        subtracted_background=background,
        prepared_saturation_level=prepared_saturation,
        reference_flux=reference_flux,
        render_valid_fraction=report.valid_fraction,
        chosen_flux_lower=chosen_flux_lower,
        chosen_flux_upper=chosen_flux_upper,
        lower_confidence_interval=lower_confidence_interval,
        upper_confidence_interval=upper_confidence_interval,
        confidence_level=float(
            confidence_level
        ),
        unit=unit,
    )


def _detectability_order_statistic(
    detection_flux: FloatArray,
    required_fraction: float,
) -> float | None:
    """Return the empirical detectability lower-bound order statistic."""
    values = np.asarray(
        detection_flux,
        dtype=float,
    )

    required_count = int(
        np.ceil(
            required_fraction
            * values.size
        )
    )

    candidate = float(
        np.sort(
            values
        )[
            required_count - 1
        ]
    )

    if (
        not np.isfinite(
            candidate
        )
        or candidate <= 0.0
    ):
        return None

    return candidate


def _saturation_order_statistic(
    saturation_flux: FloatArray,
    required_fraction: float,
) -> float | None:
    """Return the empirical saturation upper-bound order statistic."""
    values = np.asarray(
        saturation_flux,
        dtype=float,
    )

    required_count = int(
        np.ceil(
            required_fraction
            * values.size
        )
    )

    allowed_below = (
        values.size
        - required_count
    )

    candidate = float(
        np.sort(
            values
        )[
            allowed_below
        ]
    )

    if (
        not np.isfinite(
            candidate
        )
        or candidate <= 0.0
    ):
        return None

    return candidate


def _bootstrap_flux_prior_bounds(
    *,
    detection_flux: FloatArray,
    saturation_flux: FloatArray,
    required_detectable_fraction: float,
    required_unsaturated_fraction: float,
    bootstrap_samples: int,
    confidence_level: float,
    random_state: int | None,
) -> tuple[
    tuple[float, float] | None,
    tuple[float, float] | None,
]:
    """Bootstrap Monte Carlo sensitivity of calibrated flux bounds."""
    if bootstrap_samples == 0:
        return (
            None,
            None,
        )

    n_draws = int(
        detection_flux.size
    )

    seed = (
        None
        if random_state is None
        else int(
            random_state
        )
        + 1
    )

    rng = np.random.default_rng(
        seed
    )

    lower_samples: list[float] = []
    upper_samples: list[float] = []

    for _ in range(
        bootstrap_samples
    ):
        indices = rng.integers(
            0,
            n_draws,
            size=n_draws,
        )

        lower = _detectability_order_statistic(
            detection_flux[
                indices
            ],
            required_detectable_fraction,
        )

        upper = _saturation_order_statistic(
            saturation_flux[
                indices
            ],
            required_unsaturated_fraction,
        )

        if lower is not None:
            lower_samples.append(
                lower
            )

        if upper is not None:
            upper_samples.append(
                upper
            )

    alpha = (
        1.0
        - confidence_level
    ) / 2.0

    lower_interval = _finite_quantile_interval(
        lower_samples,
        alpha,
    )

    upper_interval = _finite_quantile_interval(
        upper_samples,
        alpha,
    )

    return (
        lower_interval,
        upper_interval,
    )


def _finite_quantile_interval(
    values: Sequence[float],
    alpha: float,
) -> tuple[float, float] | None:
    """Return a finite equal-tail interval from bootstrap values."""
    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(
            array
        )
    ]

    if array.size == 0:
        return None

    lower, upper = np.quantile(
        array,
        [
            alpha,
            1.0 - alpha,
        ],
    )

    return (
        float(
            lower
        ),
        float(
            upper
        ),
    )
