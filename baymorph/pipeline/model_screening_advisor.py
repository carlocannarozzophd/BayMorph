"""Preparation-free single-observation model screening."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from baymorph.analysis.model_screening import (
    ModelScreeningReport,
)
from baymorph.analysis.model_screening_features import (
    ModelScreeningFeatureVector,
    extract_model_screening_features,
)
from baymorph.core.observation import (
    Observation,
    ObservationMetadata,
)
from baymorph.pipeline.advisor import ObservationAdvisor
from baymorph.analysis.model_screening_classifier import (
    LinearDiscriminantScreeningClassifier,
)


@dataclass(frozen=True, slots=True)
class SingleObservationScreeningResult:
    """Preparation-free model-screening result for one FITS image."""

    path: Path
    shape: tuple[int, int]
    target_x: float
    target_y: float
    metadata: ObservationMetadata
    psf_fwhm_pixels: float
    psf_fwhm_source: str
    background_source: str
    noise_source: str
    features: ModelScreeningFeatureVector
    report: ModelScreeningReport

    def summary(self) -> str:
        """Return header context, assumptions, and ranked model scores."""
        metadata = self.metadata

        lines = [
            "BayMorph Single-Observation Model Screening",
            "",
            "OBSERVATION",
            f"FITS: {self.path}",
            f"Shape: {self.shape[0]} x {self.shape[1]} pixels",
            f"Filter: {_format_optional(metadata.filter_name)}",
            f"Telescope: {_format_optional(metadata.telescope)}",
            f"Instrument: {_format_optional(metadata.instrument)}",
            f"Image unit: {_format_optional(metadata.bunit)}",
            (
                "Pixel scale: "
                f"{_format_number(metadata.pixel_scale)} arcsec/pixel"
            ),
            (
                "Header seeing FWHM: "
                f"{_format_number(metadata.seeing_fwhm)}"
            ),
            (
                "Photometric zeropoint: "
                f"{_format_number(metadata.zeropoint)}"
            ),
            (
                "Exposure time: "
                f"{_format_number(metadata.exposure_time)}"
            ),
            (
                "Total exposure time: "
                f"{_format_number(metadata.total_exposure_time)}"
            ),
            (
                "Saturation level: "
                f"{_format_number(metadata.saturation)}"
            ),
            "",
            "SCREENING INPUTS",
            (
                "Target seed: "
                f"({self.target_x:.6f}, {self.target_y:.6f}) pixels"
            ),
            (
                "Diagnostic PSF FWHM: "
                f"{self.psf_fwhm_pixels:.6f} pixels"
            ),
            f"PSF scale source: {self.psf_fwhm_source}",
            f"Background source: {self.background_source}",
            f"Noise source: {self.noise_source}",
            "",
            self.report.summary(),
            "",
            "SCREENING SCOPE",
            (
                "This stage does not perform production cleaning, masking, "
                "background modelling, PSF construction, or inference."
            ),
            (
                "The diagnostic PSF scale is not a production PSF model."
            ),
            (
                "The user must choose the production model."
            ),
        ]

        return "\n".join(lines)


class SingleObservationModelAdvisor(ObservationAdvisor):
    """Rank registered BayMorph models before production preparation."""

    def screen_models(
        self,
        classifier: LinearDiscriminantScreeningClassifier,
        *,
        psf_fwhm_pixels: float | None = None,
        header_fwhm_unit: str | None = None,
        background: float | None = None,
        noise_sigma: float | None = None,
        target_mask: NDArray[np.bool_] | None = None,
    ) -> SingleObservationScreeningResult:
        """Screen all registered models without preparing the observation.

        The FITS ``FWHM`` keyword has no universal unit contract. BayMorph
        therefore never converts it silently. The caller must either supply
        ``psf_fwhm_pixels`` directly or declare whether the header value is
        expressed in ``"arcsec"`` or ``"pixels"``.
        """
        observation = Observation.from_fits(
            self.fits_path
        )
        data = np.asarray(
            observation.image.data,
            dtype=float,
        )

        if data.ndim != 2:
            raise ValueError(
                "Model screening requires a two-dimensional FITS image."
            )

        target_x, target_y = self._resolve_target(
            observation
        )
        if target_x is None or target_y is None:
            raise ValueError(
                "Model screening requires target_position. Supply pixel "
                "coordinates or use target_position='center'."
            )

        resolved_psf_fwhm, psf_source = _resolve_psf_fwhm_pixels(
            observation.metadata,
            psf_fwhm_pixels=psf_fwhm_pixels,
            header_fwhm_unit=header_fwhm_unit,
        )

        features = extract_model_screening_features(
            data,
            target_x=target_x,
            target_y=target_y,
            psf_fwhm=resolved_psf_fwhm,
            background=background,
            noise_sigma=noise_sigma,
            target_mask=target_mask,
        )
        report = classifier.screen(
            features
        )

        return SingleObservationScreeningResult(
            path=self.fits_path.resolve(),
            shape=(
                int(data.shape[0]),
                int(data.shape[1]),
            ),
            target_x=float(target_x),
            target_y=float(target_y),
            metadata=observation.metadata,
            psf_fwhm_pixels=resolved_psf_fwhm,
            psf_fwhm_source=psf_source,
            background_source=(
                "caller_supplied"
                if background is not None
                else "robust_image_border"
            ),
            noise_source=(
                "caller_supplied"
                if noise_sigma is not None
                else "robust_image_border"
            ),
            features=features,
            report=report,
        )


def _resolve_psf_fwhm_pixels(
    metadata: ObservationMetadata,
    *,
    psf_fwhm_pixels: float | None,
    header_fwhm_unit: str | None,
) -> tuple[float, str]:
    """Resolve an explicit, auditable diagnostic PSF scale."""
    if psf_fwhm_pixels is not None:
        value = float(
            psf_fwhm_pixels
        )
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                "psf_fwhm_pixels must be finite and positive."
            )

        return value, "caller_supplied_pixels"

    if metadata.seeing_fwhm is None:
        raise ValueError(
            "No FITS FWHM value is available. Supply psf_fwhm_pixels."
        )

    if header_fwhm_unit is None:
        raise ValueError(
            "The FITS FWHM unit is ambiguous. Supply psf_fwhm_pixels or "
            "set header_fwhm_unit to 'arcsec' or 'pixels'."
        )

    normalized_unit = str(
        header_fwhm_unit
    ).strip().lower()

    if normalized_unit == "pixels":
        value = float(
            metadata.seeing_fwhm
        )
        source = "fits_header_fwhm_declared_pixels"
    elif normalized_unit == "arcsec":
        if (
            metadata.pixel_scale is None
            or not np.isfinite(metadata.pixel_scale)
            or metadata.pixel_scale <= 0.0
        ):
            raise ValueError(
                "A positive celestial pixel scale is required to convert "
                "the FITS FWHM from arcseconds to pixels."
            )

        value = float(
            metadata.seeing_fwhm
            / metadata.pixel_scale
        )
        source = "fits_header_fwhm_declared_arcsec"
    else:
        raise ValueError(
            "header_fwhm_unit must be exactly 'arcsec' or 'pixels'."
        )

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            "The resolved diagnostic PSF FWHM must be finite and positive."
        )

    return value, source


def _format_optional(
    value: str | None,
) -> str:
    """Format optional textual metadata."""
    if value is None:
        return "unavailable"

    return value


def _format_number(
    value: float | None,
) -> str:
    """Format optional numerical metadata."""
    if value is None:
        return "unavailable"

    return f"{value:.8g}"


__all__ = [
    "SingleObservationModelAdvisor",
    "SingleObservationScreeningResult",
]
