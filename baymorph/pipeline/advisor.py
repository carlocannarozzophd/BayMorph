"""Evidence-aware observation inspection and configuration guidance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from baymorph.analysis.prior_predictive import (
    FluxPriorCalibration,
    calibrate_single_sersic_flux_prior,
)
from baymorph.core.header_metadata import (
    DECLARABLE_QUANTITIES,
    IMAGE_UNIT,
    HeaderReading,
    validate_declarations,
)
from baymorph.core.image import read_image_header
from baymorph.core.observation import Observation
from baymorph.models.labels import PARAMETER_KINDS
from baymorph.pipeline.config import (
    DEFAULT_CONFIG,
    load_resolved_config,
    resolve_config,
    validate_fit_priors,
)
from baymorph.pipeline.config_maker import make_config_from_fits
from baymorph.pipeline.image_preparation import (
    load_image_preparation_products,
    prepare_real_fits_image,
    prepare_real_image_from_config,
)
from baymorph.pipeline.opening import open_observation
from baymorph.pipeline.prior_proposals import (
    ANGULAR_RANGE_ARCSEC,
    BACKGROUND_SKY_SCALE_MULTIPLE,
    CENTRE_FWHM_MULTIPLE,
    GAIN_RATIO_TOLERANCE,
    MAGNITUDE_RANGE,
    PHYSICAL_LENGTH_MODELS,
    POINT_SOURCE_SNR,
    PSF_PEAK_SATURATION_FRACTION,
    PriorProposal,
    background_priors,
    noise_equivalent_area,
    propose_effective_gain,
    propose_model_priors,
    propose_psf_max_peak,
)
from baymorph.pipeline.psf_candidates import run_psf_candidate_selection_from_config
from baymorph.pipeline.psf_construction import (
    estimate_psf_core_fwhm,
    run_psf_construction_from_config,
)
from baymorph.observation.background import background_parameter_names
from baymorph.pipeline.real_fit import _load_psf_from_config
from baymorph.pipeline.preparation_steps import estimate_target_geometry_from_image
from baymorph.pipeline.segmentation import select_segment_near_position
from baymorph.simulation.psf import GaussianPSF

InputFunction = Callable[[str], str]
TargetPosition = str | Mapping[str, float] | Sequence[float] | None


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
    evidence_safe: bool | None = None


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
    seeing_fwhm_unit: str | None = None


@dataclass(frozen=True, slots=True)
class FixedInputs:
    """Fixed inputs of the observation model, measured before any prior.

    The prepared sky scale, the cutout and the PSF enter the observation
    model as fixed inputs (AD-046, AD-047). The advisor measures them by
    preparing the image and building its PSF, and derives from them the
    priors of the centre and of the residual background, and of the fluxes
    when no zeropoint is usable (AD-049).

    Parameters
    ----------
    sky_scale : float
        Prepared sky scale, in image units.
    measured_sky_level : float
        Background level that the preparation measured and subtracted.
    cutout_shape : tuple of int
        Shape ``(rows, columns)`` of the prepared cutout.
    psf_fwhm_pixels : float
        FWHM of the PSF, in pixels.
    psf_fwhm_source : str
        How the FWHM was obtained.
    psf_noise_equivalent_area : float
        Noise-equivalent area ``1 / sum(p**2)`` of the PSF kernel that the
        fit convolves with, normalized to unit sum, in square pixels.
    stage_identity : str
        Fingerprint of the configuration sections that the preparation and
        the PSF depend on; a proposal accepts only fixed inputs measured for
        its own sections.
    run_directory : str
        Directory of the products of the preparation and of the PSF.
    """

    sky_scale: float
    measured_sky_level: float
    cutout_shape: tuple[int, int]
    psf_fwhm_pixels: float
    psf_fwhm_source: str
    psf_noise_equivalent_area: float
    stage_identity: str
    run_directory: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description of the fixed inputs."""
        return {
            "sky_scale": float(self.sky_scale),
            "measured_sky_level": float(self.measured_sky_level),
            "cutout_shape": [int(value) for value in self.cutout_shape],
            "psf_fwhm_pixels": float(self.psf_fwhm_pixels),
            "psf_fwhm_source": self.psf_fwhm_source,
            "psf_noise_equivalent_area": float(self.psf_noise_equivalent_area),
            "stage_identity": self.stage_identity,
            "run_directory": self.run_directory,
        }


_STAGE_SECTIONS = ("object", "data", "preparation", "psf", "outputs")
"""Configuration sections that the preparation and the PSF depend on."""

_NON_STAGE_OPTIONS = ("likelihood", "background", "sampler")
"""Options of the configuration maker that leave the stages unchanged."""


def _stage_identity(resolved: Mapping[str, Any]) -> str:
    """Return the fingerprint of the sections the stages depend on."""
    sections = {name: resolved.get(name) for name in _STAGE_SECTIONS}
    text = json.dumps(sections, sort_keys=True, default=str)

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _baymorph_version() -> str:
    """Return the installed BayMorph version."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("baymorph")
    except PackageNotFoundError:
        from baymorph import __version__

        return __version__


PRIOR_WIDTH_WARNING_DEX = 6.0


def _numeric_prior_bound(value: Any) -> float | None:
    """Resolve numeric and supported symbolic prior bounds."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)

        return result if np.isfinite(result) else None

    symbolic = {
        "-pi": -np.pi,
        "-pi/2": -0.5 * np.pi,
        "0": 0.0,
        "pi/2": 0.5 * np.pi,
        "pi": np.pi,
    }

    return symbolic.get(str(value).strip().lower().replace(" ", ""))


def _prior_volume_position(
    prior: Mapping[str, Any], observed_scale: float | None
) -> float | None:
    """Return an observed scale's position in the declared prior measure."""
    if observed_scale is None:
        return None

    value = float(observed_scale)

    lower = _numeric_prior_bound(prior.get("lower"))
    upper = _numeric_prior_bound(prior.get("upper"))

    if lower is None or upper is None or not np.isfinite(value) or not lower < upper:
        return None

    prior_type = str(prior.get("type", "uniform")).strip().lower()

    if prior_type == "log_uniform":
        if value <= 0.0 or lower <= 0.0:
            return None

        return float(np.log(value / lower) / np.log(upper / lower))

    if prior_type == "uniform":
        return float((value - lower) / (upper - lower))

    return None


def _flux_prior_requires_review(
    prior: Mapping[str, Any],
    observed_scale: float | None,
    *,
    edge_fraction: float = 0.05,
) -> bool:
    """Return whether a flux diagnostic lies near or outside a prior edge."""
    position = _prior_volume_position(prior, observed_scale)

    if position is None:
        return False

    return bool(position <= edge_fraction or position >= (1.0 - edge_fraction))


_OBSERVED_SCALE_NOTES = {
    "flux": (
        "The positive flux of the segmented target is a diagnostic scale, "
        "not a prior bound."
    ),
    "re": (
        "The half-light radius of the segmented target is a diagnostic "
        "scale, not a prior bound."
    ),
    "q": "The image-moment estimate of the axis ratio is diagnostic only.",
    "theta": ("The image-moment estimate of the position angle is diagnostic only."),
}
"""Meaning of the scale measured on the target, for each parameter that
describes the whole target."""


def _guidance_rule(name: str, kind: str, *, one_component: bool) -> tuple[str, str]:
    """Return the prior family of a parameter and the rule that proposes it."""
    if kind == "flux":
        return "log_uniform", (
            "A flux is a positive scale. The advisor proposes magnitudes "
            f"{MAGNITUDE_RANGE[0]:g} to {MAGNITUDE_RANGE[1]:g} through the "
            "zeropoint, read or declared, log-uniform in flux; without a "
            "usable zeropoint, from the flux of a point source at "
            f"signal-to-noise {POINT_SOURCE_SNR:g} on the prepared sky to "
            f"{MAGNITUDE_RANGE[1] - MAGNITUDE_RANGE[0]:g} magnitudes above it."
        )

    if kind == "length":
        smallest, largest = ANGULAR_RANGE_ARCSEC

        if one_component:
            return "positive_scale", (
                "A length is a positive scale. The advisor proposes "
                f"{smallest:g} to {largest:g} arcsec through the pixel scale, "
                "log-uniform in pixels, within the validated rendering domain."
            )

        return "positive_scale", (
            "A length is a positive scale. Its reference range tells the "
            "components of the model apart, and the advisor clips it to "
            f"{smallest:g} to {largest:g} arcsec through the pixel scale."
        )

    if kind == "position":
        return "uniform_relative_to_target", (
            f"The centre lies within {CENTRE_FWHM_MULTIPLE:g} times the FWHM "
            "of the PSF of the confirmed target position."
        )

    if kind == "angle":
        if name.startswith("theta"):
            return "uniform_axial", (
                "A position angle is axial: theta == theta + pi. The advisor "
                "keeps the reference prior, the complete axial domain, "
                "rather than bounds read from the image."
            )

        return "uniform_periodic", (
            "A phase is periodic. The advisor keeps the reference prior, "
            "one complete period."
        )

    if name == "q" or name.startswith("q_"):
        return "bounded_0_1", (
            "An axis ratio has physical support 0 < q <= 1. The advisor "
            "keeps the reference prior of BayMorph."
        )

    return "bounded", (
        "A dimensionless shape parameter. The advisor keeps the reference "
        "prior of BayMorph; do not derive its bounds from a preliminary fit "
        "to the same image."
    )


def _settle_psf_max_peak(
    reading: HeaderReading, config_options: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """Settle the ceiling on the peak of the PSF stars in ``config_options``.

    The ceiling selects the stars of an empirical PSF, so the measurement of
    the fixed inputs and the proposal settle it in the same way before the
    stages run: the value the user declares, else half the saturation level
    when the header gives it in the units of the image or it is declared in
    ``data.calibration``, else the BayMorph default.

    Returns
    -------
    tuple
        The provenance record of the ceiling, or None for a Gaussian PSF,
        and a warning when the BayMorph default applies, or None.
    """
    psf_type = str(config_options.get("psf_type", "empirical")).strip().lower()

    if psf_type != "empirical":
        return None, None

    if config_options.get("psf_max_peak") is not None:
        return {
            "value": float(config_options["psf_max_peak"]),
            "source": "user",
            "detail": "declared by the user",
        }, None

    peak, peak_detail = propose_psf_max_peak(reading)

    if peak is not None:
        config_options["psf_max_peak"] = peak
        source = (
            "declaration" if reading["saturation"].status == "declared" else "header"
        )

        return {"value": peak, "source": source, "detail": peak_detail}, None

    return {"value": None, "source": "default", "detail": peak_detail}, (
        "No empirical-PSF candidate peak ceiling was declared, and the "
        f"header gives none ({peak_detail}). BayMorph will use its default "
        "max_peak, which is not checked against the detector saturation "
        "level or the image units. Declare psf_max_peak=... in "
        "science-image units."
    )


def _chosen_models(
    model_name: str | None,
    models: Sequence[str] | None,
    model_priors: Mapping[str, Any] | None,
) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    """Return the chosen models and the priors declared for each.

    ``model_name`` chooses one model, whose ``model_priors`` map parameters to
    priors; ``models`` chooses one or more, whose ``model_priors`` map each
    model to such a mapping. Without either, the model is ``single_sersic``.
    """
    if model_name is not None and models is not None:
        raise ValueError("Supply model_name or models, not both.")

    if models is None:
        chosen = [str("single_sersic" if model_name is None else model_name).strip()]
        keyed = {chosen[0]: model_priors or {}}
    else:
        if isinstance(models, str) or not isinstance(models, Sequence):
            raise TypeError(
                "models must be a sequence of model names, such as "
                "['single_sersic', 'bulge_disk']."
            )

        chosen = [str(name).strip() for name in models]

        if not chosen:
            raise ValueError("models must name at least one model.")

        duplicates = sorted({name for name in chosen if chosen.count(name) > 1})

        if duplicates:
            raise ValueError(f"models names {duplicates} more than once.")

        keyed = dict(model_priors or {})
        foreign = sorted(str(name) for name in set(keyed) - set(chosen))

        if foreign:
            raise ValueError(
                "With models=..., model_priors maps each chosen model to "
                f"its priors; {foreign} are not among the chosen models "
                f"{chosen}."
            )

    declared: dict[str, dict[str, dict[str, Any]]] = {}

    for name in chosen:
        if name not in PARAMETER_KINDS:
            raise ValueError(f"Unknown model: {name}")

        priors = keyed.get(name) or {}

        if not isinstance(priors, Mapping):
            raise TypeError(
                f"The declared priors of {name!r} must map parameters to " "priors."
            )

        unknown = sorted(set(map(str, priors)) - set(PARAMETER_KINDS[name]))

        if unknown:
            raise ValueError(f"Model {name!r} has no parameters {unknown}.")

        declared[name] = {
            str(parameter): dict(prior) for parameter, prior in priors.items()
        }

    return chosen, declared


def _undecided_parameters(
    decided: Mapping[str, Mapping[str, Any]], kind: str
) -> list[str]:
    """Return the parameters of one kind that have no decided prior.

    ``decided`` maps each chosen model to its decided priors. The names are
    qualified by their model, as ``model.parameter``, when several models are
    chosen.
    """
    several = len(decided) > 1

    return [
        f"{model}.{parameter}" if several else parameter
        for model, priors in decided.items()
        for parameter, parameter_kind in PARAMETER_KINDS[model].items()
        if parameter_kind == kind and parameter not in priors
    ]


@dataclass(frozen=True, slots=True)
class ObservationConfigProposal:
    """Human-reviewable proposal for one observed-image configuration.

    Creating a proposal never writes a file. The user must call
    :meth:`write` with ``accept=True`` after reviewing the chosen models,
    the priors with their sources, and the warnings. A proposal of the
    advisor carries the provenance of every proposed value, which
    :meth:`write` records in a JSON report beside the YAML, outside every
    checkpoint signature (AD-049).

    ``starting_model`` is the first of the chosen models, and ``models``
    holds all of them in the order of ``fit.models``; it is empty for a
    proposal built by hand, which then has the starting model only.
    """

    config: dict[str, Any]
    output_path: Path
    starting_model: str
    flux_prior: dict[str, Any] | None
    flux_prior_source: str
    requires_flux_review: bool
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] | None = None
    models: tuple[str, ...] = ()

    @property
    def chosen_models(self) -> tuple[str, ...]:
        """Return the chosen models, in the order of ``fit.models``."""
        return self.models or (self.starting_model,)

    @property
    def report_path(self) -> Path:
        """Return the path of the provenance report beside the YAML."""
        return self.output_path.with_name(f"{self.output_path.stem}.advisor.json")

    def summary(self) -> str:
        """Return a concise proposal summary for human review."""
        target = self.config["data"]["target_position"]
        sampler_name = (
            str(
                self.config.get("fit", {})
                .get("sampler", {})
                .get("name", "not specified")
            )
            .strip()
            .lower()
        )
        sampler_scope = f"Selected sampler: {sampler_name}."
        comparison_scope = (
            "A preferred model can be established only through "
            "appropriate evidence-producing fits and diagnostics."
        )

        if isinstance(target, Mapping):
            target_text = f"X={float(target['x']):.3f}, " f"Y={float(target['y']):.3f}"
        else:
            target_text = str(target)

        chosen = self.chosen_models
        several = len(chosen) > 1

        lines = [
            "BayMorph Configuration Proposal",
            "",
            f"Output YAML: {self.output_path}",
            (f"FITS: {self.config['data']['image_path']}"),
            f"Confirmed target: {target_text}",
            (
                f"User-selected models: {', '.join(chosen)}"
                if several
                else f"User-selected starting model: {self.starting_model}"
            ),
            (f"Flux prior source: {self.flux_prior_source}"),
        ]

        if self.flux_prior is not None:
            lines.append("Flux prior: " + _format_prior(self.flux_prior))

        lines.extend(
            [
                (
                    "Flux coverage review: "
                    + ("REQUIRED" if self.requires_flux_review else "clear")
                ),
                sampler_scope,
            ]
        )

        if self.provenance is not None:
            lines.extend(_provenance_summary(self.provenance))

        lines.extend(
            [
                "",
                "MODEL-SCREENING SCOPE",
                (
                    "The models are a user decision, not a pre-fit Bayesian "
                    "model preference. Each model is fitted separately and "
                    "has its own evidence."
                    if several
                    else "The starting model is a user decision, not a "
                    "pre-fit Bayesian model preference."
                ),
                comparison_scope,
            ]
        )

        if self.warnings:
            lines.extend(["", "WARNINGS"])
            lines.extend(f"- {warning}" for warning in self.warnings)

        lines.extend(["", "No file has been written."])

        if self.requires_flux_review:
            lines.extend(
                [
                    (
                        "Writing is blocked until a reviewed scientific "
                        "flux prior is supplied."
                    ),
                    ("Build a new proposal with flux_prior=... " "before writing."),
                ]
            )
        else:
            lines.append("Write only after review with proposal.write(accept=True).")

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Return a readable notebook representation."""
        from html import escape

        return "<pre>" + escape(self.summary()) + "</pre>"

    def write(self, *, accept: bool = False, overwrite: bool = False) -> Path:
        """Write the proposed YAML after explicit user acceptance.

        Parameters
        ----------
        accept : bool, default=False
            Must be True to confirm that the proposal has been reviewed.
        overwrite : bool, default=False
            Whether an existing file at the destination may be replaced.
            Replacing a configuration does not update products already
            created from the previous one.

        Returns
        -------
        pathlib.Path
            Resolved path of the written YAML.

        Notes
        -----
        Writing the same proposal again is accepted and changes nothing, so a
        notebook can be re-executed. A destination whose content differs is
        never replaced without ``overwrite=True``. When the proposal carries
        its provenance, the JSON report :attr:`report_path` is written with
        the YAML, and both destinations are checked before either is written.
        """
        if accept is not True:
            raise ValueError("Writing a configuration proposal requires accept=True.")

        if self.requires_flux_review:
            raise ValueError(
                "Writing is blocked because the proposed flux prior "
                "requires review. Build a new proposal with an explicit "
                "scientifically justified flux_prior."
            )

        import yaml

        destination = self.output_path.expanduser()

        documents = [
            (destination, yaml.safe_dump(self.config, sort_keys=False), "Configuration")
        ]

        if self.provenance is not None:
            documents.append(
                (
                    self.report_path.expanduser(),
                    json.dumps(self.provenance, indent=2) + "\n",
                    "Provenance report",
                )
            )

        for path, text, label in documents:
            if (
                path.exists()
                and overwrite is not True
                and path.read_text(encoding="utf-8") != text
            ):
                raise FileExistsError(
                    f"{label} already exists and differs: {path}. "
                    "Choose a new output path, or pass overwrite=True only if "
                    "replacing it is intended; products already created from "
                    "it are not updated."
                )

        for path, text, _ in documents:
            if path.exists() and path.read_text(encoding="utf-8") == text:
                continue

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

        return destination.resolve()


class ObservationAdvisor:
    """Inspect a FITS observation and guide BayMorph configuration.

    Image-derived quantities are diagnostic estimates and numerical scales.
    They are never inserted automatically into scientific priors.

    Configuration generation always delegates to
    :func:`make_config_from_fits`.

    Parameters
    ----------
    fits_path : str or pathlib.Path
        Two-dimensional FITS image.
    target_position : mapping, sequence, {"center"}, or None
        Optional initial target-identification seed.
    hdu : int or str, optional
        Index or ``EXTNAME`` of the HDU that holds the image, written to
        ``data.hdu`` of the proposed configuration.
    calibration : mapping, optional
        Calibration quantities declared where the header does not give them,
        or gives them wrongly (AD-053). Every reading of the advisor uses
        them, and the proposed configuration carries them in
        ``data.calibration``.
    """

    def __init__(
        self,
        fits_path: str | Path,
        *,
        target_position: TargetPosition = None,
        hdu: int | str | None = None,
        calibration: Mapping[str, Any] | None = None,
    ) -> None:
        self.fits_path = Path(fits_path)

        if not self.fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {self.fits_path}")

        self.target_position = target_position
        self.hdu = hdu
        self.calibration = validate_declarations(calibration)
        self._inspection: ObservationInspection | None = None

    @classmethod
    def from_project(
        cls,
        fits_path: str | Path,
        *,
        start: str | Path | None = None,
        target_position: TargetPosition = None,
        hdu: int | str | None = None,
        calibration: Mapping[str, Any] | None = None,
    ) -> ObservationAdvisor:
        """Create an Advisor using paths relative to a BayMorph project.

        Relative FITS paths are searched for from ``start`` toward its parent
        directories. The discovered project root is retained so that relative
        configuration destinations are written outside the notebook
        directory.

        Absolute FITS paths are also accepted. In that case BayMorph searches
        upward from ``start`` for a project containing both ``notebooks`` and
        ``configs`` directories.

        Parameters
        ----------
        fits_path : str or pathlib.Path
            Relative project path or absolute path to a two-dimensional FITS
            image.
        start : str or pathlib.Path, optional
            Directory from which to begin project discovery. The current
            working directory is used by default.
        target_position : mapping, sequence, {"center"}, or None
            Optional initial target-identification seed.
        hdu : int or str, optional
            Index or ``EXTNAME`` of the HDU that holds the image, written to
            ``data.hdu`` of the proposed configuration. By default BayMorph
            selects it as :meth:`ImageData.from_fits` describes.
        calibration : mapping, optional
            Calibration quantities declared in ``data.calibration``; see
            :class:`ObservationAdvisor`.

        Returns
        -------
        ObservationAdvisor
            Project-aware Advisor.
        """
        supplied_path = Path(fits_path).expanduser()
        search_start = (
            Path.cwd() if start is None else Path(start).expanduser()
        ).resolve()

        search_directories = (search_start, *search_start.parents)

        if supplied_path.is_absolute():
            resolved_fits_path = supplied_path.resolve()

            if not resolved_fits_path.is_file():
                raise FileNotFoundError(f"FITS file not found: {resolved_fits_path}")

            project_root = next(
                (
                    directory
                    for directory in search_directories
                    if (directory / "notebooks").is_dir()
                    and (directory / "configs").is_dir()
                ),
                None,
            )
        else:
            project_root = None
            resolved_fits_path = None

            for directory in search_directories:
                candidate = (directory / supplied_path).resolve()

                if candidate.is_file():
                    project_root = directory
                    resolved_fits_path = candidate
                    break

            if resolved_fits_path is None:
                raise FileNotFoundError(
                    "Could not find FITS image "
                    f"{str(supplied_path)!r} from {search_start} "
                    "or any parent directory."
                )

        advisor = cls(
            resolved_fits_path,
            target_position=target_position,
            hdu=hdu,
            calibration=calibration,
        )
        advisor.project_root = project_root

        return advisor

    def set_target_position(self, x: float, y: float) -> tuple[float, float]:
        """Set an explicit zero-based target position inside the FITS image.

        Parameters
        ----------
        x, y : float
            BayMorph pixel coordinates. ``x`` is the NumPy column coordinate
            and ``y`` is the NumPy row coordinate.

        Returns
        -------
        tuple of float
            Validated ``(x, y)`` target position.
        """
        observation = self._open()
        data = np.asarray(observation.image.data, dtype=float)

        if data.ndim != 2:
            raise ValueError("Target selection requires a two-dimensional FITS image.")

        x = float(x)
        y = float(y)

        if not np.all(np.isfinite([x, y])):
            raise ValueError("Target coordinates must be finite.")

        height, width = data.shape

        if not (0.0 <= x <= width - 1 and 0.0 <= y <= height - 1):
            raise ValueError(
                "Target coordinates must lie inside the FITS image: "
                f"0 <= x <= {width - 1} and "
                f"0 <= y <= {height - 1}."
            )

        position = (x, y)

        self.target_position = position
        self._inspection = None

        return position

    def clear_target_position(self) -> None:
        """Clear the target position and any cached target inspection."""
        self.target_position = None
        self._inspection = None

    def select_target(
        self,
        *,
        initial_position: tuple[float, float] | None = None,
        title: str = "Select the target center",
        max_preview_size: int = 1000,
    ):
        """Return an interactive notebook selector for the target center.

        The selector displays the original FITS image using Plotly. Clicking
        the image or editing the numerical fields moves a candidate crosshair.
        The candidate becomes the Advisor target only after the user presses
        ``Confirm center``.

        Opening a selector places the Advisor target in a pending state. This
        prevents a previous or geometric-center candidate from being used
        silently after the interactive selection has begun.

        Parameters
        ----------
        initial_position : tuple of float, optional
            Initial unconfirmed candidate. If omitted, the existing Advisor
            target is used when available; otherwise the geometric image
            center is shown.
        title : str, default="Select the target center"
            Interactive figure title.
        max_preview_size : int, default=1000
            Maximum preview dimension. Larger images are sampled for display
            while retaining coordinates in the original FITS pixel frame.

        Returns
        -------
        baymorph.plotting.target_selector.TargetSelector
            Interactive selector whose ``value`` remains None until explicit
            confirmation.
        """
        from baymorph.plotting.target_selector import TargetSelector

        observation = self._open()
        data = np.asarray(observation.image.data, dtype=float)

        if data.ndim != 2:
            raise ValueError(
                "Interactive target selection requires a two-dimensional FITS image."
            )

        if initial_position is None and self.target_position is not None:
            target_x, target_y = self._resolve_target(observation)

            if target_x is not None and target_y is not None:
                initial_position = (float(target_x), float(target_y))

        def confirm_target(position: tuple[float, float]) -> None:
            self.set_target_position(*position)

        self.clear_target_position()

        selector = TargetSelector(
            data,
            initial_position=initial_position,
            title=title,
            max_preview_size=max_preview_size,
            on_confirm=confirm_target,
            on_invalidate=self.clear_target_position,
        )

        self._target_selector = selector

        return selector

    def header_report(self) -> str:
        """Return the table of the quantities read from the FITS header.

        Each row gives a quantity, its value, its unit, its status and the
        keyword and convention it was read from (AD-049). A quantity whose
        status is ``ambiguous`` or ``conflicting`` shows its raw value and
        is never used without a declaration of the user; a quantity that
        the header does not hold is left out. A quantity declared through
        ``calibration`` shows the status ``declared`` and the source
        ``data.calibration`` (AD-053). Only the header is read, never the
        pixels, so the report of a mosaic of several gigabytes costs the
        same as the report of a small stamp.

        Returns
        -------
        str
            The report, ready to print.
        """
        hdu_text = "selected by BayMorph" if self.hdu is None else repr(self.hdu)
        lines = [
            "BayMorph Header Reading",
            "",
            f"FITS: {self.fits_path}",
            f"HDU: {hdu_text}",
            "",
            self._header_reading().report(),
        ]

        return "\n".join(lines)

    def inspect(self) -> ObservationInspection:
        """Inspect metadata, image scales, and optional target morphology."""
        observation = self._open()

        data = np.asarray(observation.image.data, dtype=float)

        if data.ndim != 2:
            raise ValueError(
                "ObservationAdvisor requires a two-dimensional FITS image."
            )

        finite = np.isfinite(data)
        finite_values = data[finite]

        if finite_values.size == 0:
            raise ValueError("FITS image contains no finite pixels.")

        prepared = prepare_real_fits_image(data)

        noise_sigma = float(np.nanmedian(prepared.noise))

        target_x, target_y = self._resolve_target(observation)

        target_flux_scale = None
        target_half_light_radius = None
        target_axis_ratio = None
        target_position_angle = None
        target_area_pixels = None

        segmentation = prepared.detection.segmentation

        if target_x is not None and segmentation is not None:
            label = select_segment_near_position(segmentation, x=target_x, y=target_y)

            target_segment = segmentation.data == label

            target_area_pixels = int(np.count_nonzero(target_segment))

            geometry = estimate_target_geometry_from_image(
                prepared.data, target_segment
            )

            target_axis_ratio = float(geometry.q)

            target_position_angle = float(geometry.theta)

            source_signal = np.where(
                target_segment, np.clip(prepared.data, 0.0, None), 0.0
            )

            flux = float(np.sum(source_signal))

            if np.isfinite(flux) and flux > 0.0:
                target_flux_scale = flux

                target_half_light_radius = _half_light_radius(
                    source_signal,
                    x0=geometry.x0,
                    y0=geometry.y0,
                    q=geometry.q,
                    theta=geometry.theta,
                )

        metadata = observation.metadata

        inspection = ObservationInspection(
            path=self.fits_path.resolve(),
            shape=(int(data.shape[0]), int(data.shape[1])),
            finite_fraction=float(np.mean(finite)),
            minimum=float(np.nanmin(data)),
            maximum=float(np.nanmax(data)),
            median=float(np.nanmedian(data)),
            background=float(prepared.background),
            noise_sigma=noise_sigma,
            pixel_scale=_finite_or_none(metadata.pixel_scale),
            seeing_fwhm=_finite_or_none(metadata.seeing_fwhm),
            seeing_fwhm_unit=metadata.seeing_fwhm_unit,
            saturation=_finite_or_none(metadata.saturation),
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
        self, model_name: str = "single_sersic"
    ) -> dict[str, PriorGuidance]:
        """Return evidence-aware prior guidance for every parameter of a model.

        The guidance names the prior family of each parameter and the rule
        by which :meth:`propose_config` proposes it, from the kind of the
        parameter (AD-049). A parameter that describes the whole target,
        ``flux``, ``re``, ``q`` or ``theta``, also carries the scale measured
        on the segmented target. Those scales come from the image being
        fitted: they are diagnostics, never prior bounds.

        Parameters
        ----------
        model_name : str, default="single_sersic"
            Any of the ten built-in models.

        Returns
        -------
        dict of str to PriorGuidance
            One entry per parameter, in the order of the model.
        """
        model_name = str(model_name).strip()

        if model_name not in PARAMETER_KINDS:
            raise ValueError(f"Unknown model: {model_name}")

        inspection = self._inspection or self.inspect()

        observed_scales = {
            "flux": inspection.target_flux_scale,
            "re": inspection.target_half_light_radius,
            "q": inspection.target_axis_ratio,
            "theta": (
                None
                if inspection.target_position_angle is None
                else float(
                    (inspection.target_position_angle + 0.5 * np.pi) % np.pi
                    - 0.5 * np.pi
                )
            ),
        }
        image_scale = {
            "observed_scale_source": "fitted_image",
            "observed_scale_role": "diagnostic_numerical_scale",
            "use_observed_scale_for_prior": False,
        }
        one_component = model_name in PHYSICAL_LENGTH_MODELS
        guidance: dict[str, PriorGuidance] = {}

        for name, kind in PARAMETER_KINDS[model_name].items():
            family, rationale = _guidance_rule(name, kind, one_component=one_component)

            if name in observed_scales:
                guidance[name] = PriorGuidance(
                    parameter=name,
                    family=family,
                    rationale=f"{rationale} {_OBSERVED_SCALE_NOTES[name]}",
                    observed_scale=observed_scales[name],
                    **image_scale,
                )
            else:
                guidance[name] = PriorGuidance(
                    parameter=name, family=family, rationale=rationale
                )

        return guidance

    def summary(self) -> str:
        """Return a concise human-readable inspection summary."""
        inspection = self._inspection or self.inspect()

        lines = [
            "BayMorph Observation Advisor",
            "",
            "OBSERVATION",
            f"FITS: {inspection.path}",
            (f"Shape: {inspection.shape[1]} x {inspection.shape[0]} pixels"),
            f"Finite pixels: {inspection.finite_fraction:.3%}",
            (
                "Pixel scale: "
                + _format_optional(inspection.pixel_scale, "arcsec/pixel")
            ),
            (
                "Header FWHM: "
                + _format_header_fwhm(
                    inspection.seeing_fwhm, inspection.seeing_fwhm_unit
                )
            ),
            f"Image units: {inspection.bunit or 'not established'}",
            (
                "Saturation: "
                + (
                    "not established (see the header report)"
                    if inspection.saturation is None
                    else f"{inspection.saturation:.6g} in the units of the image"
                )
            ),
            ("Background (whole-image diagnostic): " f"{inspection.background:.6g}"),
            ("Noise sigma (whole-image diagnostic): " f"{inspection.noise_sigma:.6g}"),
        ]

        if inspection.target_x is not None:
            radius_text = _format_optional(
                inspection.target_half_light_radius, "pixels"
            )

            if (
                inspection.target_half_light_radius is not None
                and inspection.pixel_scale is not None
            ):
                radius_arcsec = (
                    inspection.target_half_light_radius * inspection.pixel_scale
                )
                radius_text += f" ({radius_arcsec:.6g} arcsec)"

            theta_text = "unavailable"

            if inspection.target_position_angle is not None:
                theta_degrees = np.degrees(inspection.target_position_angle) % 180.0
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
                            (inspection.bunit or "image units"),
                        )
                    ),
                    ("Segmented half-light radius scale: " + radius_text),
                    (
                        "Axis-ratio moment estimate: "
                        + _format_optional(inspection.target_axis_ratio, "")
                    ),
                    ("Position-angle moment estimate: " + theta_text),
                ]
            )

        lines.extend(
            [
                "",
                "EVIDENCE SAFETY",
                ("Target diagnostics are derived from the fitted image."),
                ("Their role is diagnostic and numerical only."),
                (
                    "BayMorph does not use them automatically to define "
                    "scientific prior bounds."
                ),
            ]
        )

        return "\n".join(lines)

    def make_config(
        self, *, model_priors: Mapping[str, Any] | None = None, **config_options: Any
    ) -> dict[str, Any]:
        """Create a config using the canonical BayMorph config maker.

        The configuration carries the calibration declared to the advisor in
        ``data.calibration``.
        """
        target_position = config_options.pop("target_position", self.target_position)

        if target_position is None:
            raise ValueError("target_position is required to create a configuration.")

        if "calibration" in config_options:
            raise TypeError(
                "Declare the calibration when creating the advisor, "
                "ObservationAdvisor(..., calibration=...), so that every "
                "reading of the advisor uses it."
            )

        config_options.setdefault("hdu", self.hdu)
        config_options["calibration"] = self.calibration

        return make_config_from_fits(
            self.fits_path,
            target_position=target_position,
            model_priors=model_priors,
            **config_options,
        )

    def calibrate_flux_prior(
        self,
        config: str | Path | Mapping[str, Any] | None = None,
        *,
        prepare: bool = False,
        build_psf: bool = False,
        snr_threshold: float = 5.0,
        required_detectable_fraction: float = 0.95,
        required_unsaturated_fraction: float = 0.95,
        n_draws: int = 256,
        random_state: int | None = 0,
        bootstrap_samples: int = 512,
        confidence_level: float = 0.95,
        **config_options: Any,
    ) -> FluxPriorCalibration:
        """Calibrate single-Sersic flux bounds for one prepared observation.

        The calibration uses the declared morphology priors together with
        observation-level quantities required to translate those priors into
        detectable and non-saturating flux support.

        The measured target flux and image-derived target morphology
        diagnostics are not used to define the calibrated bounds.

        Parameters
        ----------
        config : mapping, str, pathlib.Path, or None
            Existing BayMorph configuration describing the observation.
            If None, the Advisor creates a calibration workspace internally.
            The temporary configuration contains a fixed positive flux
            placeholder only to satisfy configuration validation. That
            placeholder is removed before prior-predictive calibration and
            never defines the scientific flux prior.
        prepare : bool, default=False
            If True, run the canonical image-preparation pipeline before
            calibration. Otherwise load existing preparation products.
        build_psf : bool, default=False
            If True and the configuration declares an empirical PSF, run
            canonical PSF-candidate selection followed by empirical-PSF
            construction before calibration. Otherwise load the existing
            empirical PSF. Ignored for a Gaussian PSF, which requires no
            construction step.
        snr_threshold : float, default=5
            Quadrature model-S/N detectability threshold.
        required_detectable_fraction : float, default=0.95
            Required morphology fraction detectable at the lower bound.
        required_unsaturated_fraction : float, default=0.95
            Required morphology fraction remaining below saturation at the
            upper bound.
        n_draws : int, default=256
            Number of morphology draws.
        random_state : int or None, default=0
            Random seed for morphology sampling.
        bootstrap_samples : int, default=512
            Number of bootstrap resamples used to quantify finite-ensemble
            sensitivity.
        confidence_level : float, default=0.95
            Equal-tail bootstrap interval confidence level.

        Returns
        -------
        FluxPriorCalibration
            Joint detectability and saturation calibration.

        Notes
        -----
        This is an observationally calibrated prior construction, not an
        objective or universally evidence-safe prior. Its result is
        conditional on the declared morphology priors, PSF, noise model,
        footprint, mask, detector metadata, and calibration policy.
        """
        if config is None:
            model_name = str(config_options.pop("model_name", "single_sersic"))

            if model_name != "single_sersic":
                raise NotImplementedError(
                    "Flux prior calibration currently supports only 'single_sersic'."
                )

            # Resolve relative output paths before make_config() can bind
            # them to the notebook kernel's working directory.
            project_root = getattr(self, "project_root", None)
            output_root = Path(config_options.get("outputs_root", "runs")).expanduser()

            if project_root is not None and not output_root.is_absolute():
                config_options["outputs_root"] = str(
                    (Path(project_root) / output_root).resolve()
                )

            workspace = self.make_config(
                model_name=model_name,
                model_priors={"flux": {"type": "fixed", "value": 1.0}},
                **config_options,
            )

            resolved = resolve_config(workspace)

        else:
            if config_options:
                raise TypeError(
                    "Configuration options may be supplied only when config=None."
                )

            if isinstance(config, (str, Path)):
                resolved = load_resolved_config(config)
            else:
                resolved = resolve_config(dict(config))

        if prepare:
            products = prepare_real_image_from_config(resolved, save=True)
        else:
            products = load_image_preparation_products(resolved)

        psf_type = resolved.get("psf", {}).get("type", "empirical")

        if build_psf and psf_type == "empirical":
            run_psf_candidate_selection_from_config(resolved)
            run_psf_construction_from_config(resolved)

        psf = _load_psf_from_config(resolved)

        inspection = self._inspection or self.inspect()

        if inspection.saturation is None:
            raise ValueError(
                "Flux upper-bound calibration requires the saturation level in "
                "the units of the image, which the header does not establish; "
                "declare it in data.calibration.saturation, through "
                "ObservationAdvisor(..., calibration={'saturation': ...})."
            )

        try:
            priors = dict(resolved["fit"]["priors"]["single_sersic"])
        except KeyError as exc:
            raise ValueError(
                "Flux calibration requires declared single-Sersic priors."
            ) from exc

        # Flux support is the quantity being calibrated. Any flux prior
        # present in the workspace or an existing scientific configuration
        # must therefore not influence the calibration.
        priors.pop("flux", None)

        references = {
            "target_x": float(products.target_x),
            "target_y": float(products.target_y),
        }

        return calibrate_single_sersic_flux_prior(
            priors=priors,
            image_shape=products.prepared_data.shape,
            psf=psf,
            noise_sigma=products.noise,
            saturation_level=inspection.saturation,
            subtracted_background=products.background,
            references=references,
            mask=products.fitting_mask,
            snr_threshold=snr_threshold,
            required_detectable_fraction=required_detectable_fraction,
            required_unsaturated_fraction=required_unsaturated_fraction,
            n_draws=n_draws,
            random_state=random_state,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            unit=inspection.bunit,
        )

    def _stage_config(
        self, *, model_name: str, **config_options: Any
    ) -> dict[str, Any]:
        """Return the resolved configuration that the stages run on.

        It is the configuration :meth:`propose_config` writes with the same
        options, without priors, likelihood, background or sampler, which the
        preparation and the PSF do not depend on. A relative output root is
        anchored to the project root, as the runner anchors it for a
        configuration in the ``configs`` directory of the project.
        """
        options = {
            key: value
            for key, value in config_options.items()
            if key not in _NON_STAGE_OPTIONS
        }
        project_root = getattr(self, "project_root", None)
        output_root = Path(options.get("outputs_root", "runs")).expanduser()

        if project_root is not None and not output_root.is_absolute():
            options["outputs_root"] = str((Path(project_root) / output_root).resolve())

        config = self.make_config(
            output_path=None,
            model_name=model_name,
            model_priors=None,
            _require_complete_priors=False,
            **options,
        )

        return resolve_config(config)

    def _run_fixed_input_stages(self, resolved: Mapping[str, Any]) -> tuple[Any, Any]:
        """Run the preparation and PSF stages of the runner.

        The stages reuse products already accepted for this configuration,
        and fail closed when persisted products do not match it.

        Returns
        -------
        tuple
            The preparation products, and the PSF image of an empirical PSF
            or None for a Gaussian PSF.
        """
        # The runner orchestrates the canonical stages; importing it here
        # keeps the pipeline modules independent of the public entry point.
        from baymorph.runner import BayMorphRunner

        runner = BayMorphRunner(dict(resolved))
        products = runner.prepare_image()
        psf = None

        if str(resolved["psf"]["type"]).strip().lower() == "empirical":
            runner.select_psf_candidates()
            psf = runner.build_psf()

        return products, psf

    def measure_fixed_inputs(
        self, *, model_name: str = "single_sersic", **config_options: Any
    ) -> FixedInputs:
        """Prepare the image and build its PSF, and return the fixed inputs.

        The prepared sky scale, the cutout and the PSF are fixed inputs of
        the observation model. This method runs the preparation and PSF
        stages of :class:`baymorph.BayMorphRunner` on the configuration that
        :meth:`propose_config` writes with the same options. The products are
        written to the run directory of that configuration, and products
        already accepted for it are reused, so the fit that follows does not
        prepare the image again.

        The ceiling on the peak of the PSF stars is settled as
        :meth:`propose_config` settles it: when ``psf_max_peak`` is not
        declared, half the saturation level of the header, when its unit is
        established, selects the stars.

        Parameters
        ----------
        model_name : str, default="single_sersic"
            Model of the configuration; the stages do not depend on it.
        **config_options
            Options of :func:`baymorph.make_config_from_fits` that the
            proposal will use, such as ``object_name``, ``outputs_root``,
            ``cutout_mode``, ``psf_type``, ``psf_max_peak`` or
            ``preparation``.

        Returns
        -------
        FixedInputs
            The sky scale, the measured sky level, the cutout shape, and the
            FWHM and noise-equivalent area of the PSF.
        """
        if self.target_position is None:
            raise ValueError(
                "A confirmed target position is required before preparing " "the image."
            )

        _settle_psf_max_peak(self._header_reading(), config_options)

        resolved = self._stage_config(model_name=model_name, **config_options)
        products, psf = self._run_fixed_input_stages(resolved)

        noise = np.asarray(products.noise, dtype=float)
        finite_noise = noise[np.isfinite(noise)]

        if finite_noise.size == 0:
            raise ValueError("The preparation gave no finite sky scale.")

        sky_scale = float(np.median(finite_noise))

        psf_config = resolved["psf"]

        if str(psf_config["type"]).strip().lower() == "gaussian":
            psf_fwhm = float(psf_config["gaussian"]["fwhm"])
            psf_fwhm_source = "FWHM of the declared Gaussian PSF"
            # The kernel of the default size, the one the fit convolves with.
            psf_area = noise_equivalent_area(GaussianPSF(fwhm=psf_fwhm).kernel())
        else:
            if psf is None:
                raise ValueError("The PSF stage returned no empirical PSF.")

            psf_fwhm = float(estimate_psf_core_fwhm(np.asarray(psf, dtype=float)))
            psf_fwhm_source = "core FWHM of the empirical PSF"
            psf_area = noise_equivalent_area(psf)

        if not (np.isfinite(sky_scale) and sky_scale > 0.0):
            raise ValueError("The prepared sky scale must be positive and finite.")

        if not (np.isfinite(psf_fwhm) and psf_fwhm > 0.0):
            raise ValueError("The FWHM of the PSF must be positive and finite.")

        rows, columns = np.shape(products.prepared_data)

        return FixedInputs(
            sky_scale=sky_scale,
            measured_sky_level=float(products.background),
            cutout_shape=(int(rows), int(columns)),
            psf_fwhm_pixels=psf_fwhm,
            psf_fwhm_source=psf_fwhm_source,
            psf_noise_equivalent_area=psf_area,
            stage_identity=_stage_identity(resolved),
            run_directory=str(
                Path(resolved["outputs"]["root"]) / str(resolved["object"]["name"])
            ),
        )

    def propose_config(
        self,
        *,
        output_path: str | Path,
        model_name: str | None = None,
        models: Sequence[str] | None = None,
        flux_calibration: FluxPriorCalibration | None = None,
        flux_prior: Mapping[str, Any] | None = None,
        model_priors: Mapping[str, Any] | None = None,
        fixed_inputs: FixedInputs | None = None,
        background_model: str | None = None,
        resampling_variance_factor: float | None = None,
        **config_options: Any,
    ) -> ObservationConfigProposal:
        """Build a reviewable configuration proposal without writing it.

        The proposal follows AD-049: its priors come only from the header,
        from physics and from the fixed inputs of the observation model,
        never from the intensities of the image being fitted.

        - Every flux takes the magnitudes 8 to 30 through the zeropoint,
          read from the header or declared through ``calibration``, as a
          log-uniform prior in image units, unless a flux prior is supplied:
          ``flux_calibration``, ``flux_prior`` or an entry of
          ``model_priors``. Without a usable zeropoint, every flux is
          log-uniform in image units from the flux of a point source at
          signal-to-noise 1 on the prepared sky, the sky scale times the
          square root of the noise-equivalent area of the PSF, to 22
          magnitudes above it. Each flux prior records its rule and its
          numbers.
        - Every length of a model with one extended component takes 0.1 to
          120 arcsec through the pixel scale, as a log-uniform prior in
          pixels within the validated rendering domain; the lengths of the
          other models keep their reference ranges, which tell their
          components apart, clipped to that angular range.
        - The centre lies within three times the FWHM of the PSF of the
          target position, and the dimensionless parameters and angles keep
          the reference priors of BayMorph.
        - The residual background lies within five times the prepared sky
          scale.
        - The effective gain of AD-047 comes from the sky level the header
          records, the measured sky, or the gain keyword, in that order; the
          ceiling on the peak of the PSF stars is half the saturation level
          when its unit is established.

        One proposal may hold several models, chosen with ``models``. Each
        receives its priors by the same rules, and the fit runs one fit per
        model, each with its own evidence, on the same prepared image, PSF,
        likelihood and residual background.

        The fixed inputs come from :meth:`measure_fixed_inputs`, which
        prepares the image and builds its PSF when they are not supplied. A
        prior, background, likelihood or PSF ceiling that the caller supplies
        is kept as declared. The chosen models remain an explicit user
        decision, not a pre-fit evidence preference.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Destination used only if the returned proposal is explicitly
            accepted and written.
        model_name : str, optional
            One user-selected starting model; ``single_sersic`` when neither
            ``model_name`` nor ``models`` is given.
        models : sequence of str, optional
            One or more user-selected models, written to ``fit.models`` in
            this order. Not allowed together with ``model_name``.
        flux_calibration : FluxPriorCalibration, optional
            Observational detectability-and-saturation calibration of the
            total flux of a single chosen model with one flux.
        flux_prior : mapping, optional
            Explicit scientific prior of the total flux of a single chosen
            model with one flux.
        model_priors : mapping, optional
            Explicit priors of any parameters, kept as declared. With
            ``model_name`` it maps parameters to priors; with ``models`` it
            maps each chosen model to such a mapping.
        fixed_inputs : FixedInputs, optional
            Fixed inputs measured by :meth:`measure_fixed_inputs` with the
            same options.
        background_model : {"constant", "plane"}, optional
            Residual background model whose priors the advisor proposes; by
            default the BayMorph default model. Not allowed together with
            ``background``.
        resampling_variance_factor : float, optional
            Factor by which a resampling of the image multiplied the variance
            of its noise, which lets the gain keyword of a resampled image be
            proposed.
        **config_options
            Options forwarded to the canonical configuration maker.

        Returns
        -------
        ObservationConfigProposal
            Immutable review object. No YAML is written at this stage.
        """
        if self.target_position is None:
            raise ValueError(
                "A confirmed target position is required before proposing "
                "a configuration."
            )

        chosen, declared = _chosen_models(model_name, models, model_priors)
        several = len(chosen) > 1
        first = chosen[0]

        if several and (flux_calibration is not None or flux_prior is not None):
            raise ValueError(
                "flux_calibration and flux_prior apply to a proposal of one "
                "model; with several models, declare each flux through "
                "model_priors."
            )

        supplied_sources = sum(
            source is not None
            for source in (flux_calibration, flux_prior, declared[first].get("flux"))
        )

        if supplied_sources > 1:
            raise ValueError(
                "Supply exactly one flux-prior source: flux_calibration, "
                "flux_prior, or model_priors['flux'], or none to let the "
                "advisor propose the flux prior."
            )

        if (
            flux_calibration is not None or flux_prior is not None
        ) and "flux" not in PARAMETER_KINDS[first]:
            raise ValueError(
                "flux_calibration and flux_prior apply to models with one "
                f"total flux; declare the fluxes of {first!r} through "
                "model_priors."
            )

        if (
            config_options.get("background") is not None
            and background_model is not None
        ):
            raise ValueError("Supply background or background_model, not both.")

        likelihood = config_options.get("likelihood")

        if (
            isinstance(likelihood, Mapping)
            and likelihood.get("include_normalization", True) is False
        ):
            raise ValueError(
                "ObservationAdvisor.propose_config() requires a normalized "
                "likelihood for standard evidence interpretation."
            )

        warnings: list[str] = []
        decided: dict[str, dict[str, PriorProposal]] = {
            name: {
                parameter: PriorProposal(
                    parameter=parameter,
                    kind=PARAMETER_KINDS[name][parameter],
                    prior=prior,
                    source="user",
                    detail="declared by the user",
                )
                for parameter, prior in declared[name].items()
            }
            for name in chosen
        }

        if flux_calibration is not None:
            candidate = flux_calibration.candidate_prior

            if candidate is None:
                raise ValueError(
                    "Flux calibration did not produce a valid candidate prior."
                )

            decided[first]["flux"] = PriorProposal(
                parameter="flux",
                kind="flux",
                prior=dict(candidate),
                source="calibration",
                detail="observational detectability and saturation calibration",
            )
        elif flux_prior is not None:
            decided[first]["flux"] = PriorProposal(
                parameter="flux",
                kind="flux",
                prior=dict(flux_prior),
                source="user",
                detail="explicit user-declared scientific prior",
            )

        reading = self._header_reading()
        zeropoint = _finite_or_none(reading.value("zeropoint"))
        zeropoint_declared = reading["zeropoint"].status == "declared"
        pixel_scale = _finite_or_none(reading.value("pixel_scale"))
        photometric_system = reading.value("photometric_system")
        undecided_lengths = _undecided_parameters(decided, "length")

        if pixel_scale is None and undecided_lengths:
            warnings.append(
                "The header gives no pixel scale, so the lengths "
                f"{undecided_lengths} keep the reference priors of BayMorph in "
                "pixels instead of an angular range. Review them, or declare "
                "them through model_priors."
            )

        # The ceiling on the peak of the PSF stars shapes the PSF, so it is
        # settled before the stages run, as measure_fixed_inputs settles it.
        max_peak_record, peak_warning = _settle_psf_max_peak(reading, config_options)

        if peak_warning is not None:
            warnings.append(peak_warning)

        stage_identity = _stage_identity(
            self._stage_config(model_name=first, **config_options)
        )

        if fixed_inputs is None:
            fixed_inputs = self.measure_fixed_inputs(model_name=first, **config_options)

        if fixed_inputs.stage_identity != stage_identity:
            raise ValueError(
                "The fixed inputs were measured for another preparation or "
                "PSF configuration. Measure them again with the options of "
                "this proposal, or let the proposal measure them."
            )

        proposals = {
            name: propose_model_priors(
                name,
                zeropoint=zeropoint,
                pixel_scale=pixel_scale,
                psf_fwhm_pixels=fixed_inputs.psf_fwhm_pixels,
                decided=decided[name],
                photometric_system=(
                    None if photometric_system is None else str(photometric_system)
                ),
                zeropoint_declared=zeropoint_declared,
                sky_scale=fixed_inputs.sky_scale,
                psf_noise_equivalent_area=fixed_inputs.psf_noise_equivalent_area,
            )
            for name in chosen
        }

        if config_options.get("background") is not None:
            background_record = {"source": "user", "detail": "declared by the user"}
        else:
            chosen_background = (
                str(background_model or DEFAULT_CONFIG["fit"]["background"]["model"])
                .strip()
                .lower()
            )
            config_options["background"] = {
                "model": chosen_background,
                "priors": background_priors(
                    chosen_background,
                    sky_scale=fixed_inputs.sky_scale,
                    cutout_shape=fixed_inputs.cutout_shape,
                ),
            }
            background_record = {
                "source": "fixed_inputs",
                "detail": (
                    f"level within {BACKGROUND_SKY_SCALE_MULTIPLE:g} times "
                    f"the prepared sky scale {fixed_inputs.sky_scale:.4g}"
                    + (
                        ", and gradients that change the plane by as much "
                        "over half the cutout"
                        if chosen_background == "plane"
                        else ""
                    )
                ),
            }

        likelihood = config_options.get("likelihood")
        likelihood_type = (
            (
                str(likelihood.get("type", "correlated_gaussian"))
                if isinstance(likelihood, Mapping)
                else "correlated_gaussian"
            )
            .strip()
            .lower()
        )

        if isinstance(likelihood, Mapping) and "effective_gain" in likelihood:
            gain_record: dict[str, Any] = {
                "source": "user",
                "declaration": deepcopy(dict(likelihood["effective_gain"])),
            }
        elif likelihood_type != "correlated_gaussian":
            gain_record = {
                "source": None,
                "detail": (
                    f"not proposed: the {likelihood_type!r} likelihood "
                    "declares no effective gain"
                ),
            }
        else:
            gain = propose_effective_gain(
                reading,
                sky_scale=fixed_inputs.sky_scale,
                measured_sky_level=fixed_inputs.measured_sky_level,
                resampling_variance_factor=resampling_variance_factor,
            )
            warnings.extend(gain.warnings)
            declared_likelihood = (
                deepcopy(dict(likelihood))
                if isinstance(likelihood, Mapping)
                else {"type": "correlated_gaussian"}
            )

            if gain.declaration is not None:
                declared_likelihood["effective_gain"] = deepcopy(gain.declaration)

            config_options["likelihood"] = declared_likelihood
            gain_record = {
                "source": (
                    None if gain.declaration is None else "header and fixed_inputs"
                ),
                **gain.as_dict(),
            }

        sampler = config_options.get("sampler", "nautilus")
        sampler_name = (
            str(sampler.get("name", "") if isinstance(sampler, Mapping) else sampler)
            .strip()
            .lower()
        )

        if sampler_name == "pocomc" and any(
            proposal.prior.get("type") == "log_uniform"
            for model_proposals in proposals.values()
            for proposal in model_proposals.values()
        ):
            raise ValueError(
                "pocoMC accepts uniform, beta and normal priors only, and the "
                "proposal holds log-uniform priors. Choose nautilus or "
                "dynesty, or declare the priors through model_priors."
            )

        inspection = self._inspection or self.inspect()
        flux_review_models = [
            name
            for name in chosen
            if "flux" in proposals[name]
            and _flux_prior_requires_review(
                proposals[name]["flux"].prior, inspection.target_flux_scale
            )
        ]
        requires_flux_review = bool(flux_review_models)

        if requires_flux_review:
            warnings.append(
                "The observed target flux diagnostic lies outside or near "
                "an edge of the proposed flux-prior measure"
                + (f" of {flux_review_models}" if several else "")
                + ". The prior was not widened automatically; posterior "
                "boundary diagnostics must be reviewed before interpreting "
                "the posterior or any evidence comparison."
            )

        warnings.append(
            "No validated model-screening classifier is distributed with "
            "BayMorph. "
            + (
                "The choice of the models is therefore a user decision."
                if several
                else "The starting model is therefore a user decision."
            )
        )

        config = self.make_config(
            output_path=None,
            model_name=first,
            model_priors={
                parameter: deepcopy(proposal.prior)
                for parameter, proposal in proposals[first].items()
            },
            **config_options,
        )
        config["fit"]["models"] = list(chosen)
        config["fit"]["priors"] = {
            name: {
                parameter: deepcopy(proposal.prior)
                for parameter, proposal in proposals[name].items()
            }
            for name in chosen
        }

        # Validate every chosen model as the fit will read it, and record
        # the prior of every parameter the fit will sample, in the order of
        # the model, so the written YAML does not depend on BayMorph defaults
        # that a later version could change.
        resolved = resolve_config(config)
        validate_fit_priors(resolved)
        config["fit"]["priors"] = {
            name: {
                parameter: deepcopy(resolved["fit"]["priors"][name][parameter])
                for parameter in PARAMETER_KINDS[name]
            }
            for name in chosen
        }

        background_resolved = resolved["fit"]["background"]
        background_model_name = str(background_resolved["model"]).strip().lower()
        config["fit"]["background"] = {
            "model": background_model_name,
            "priors": {
                name: deepcopy(background_resolved["priors"][name])
                for name in background_parameter_names(background_model_name)
            },
        }

        destination = Path(output_path).expanduser()
        project_root = getattr(self, "project_root", None)

        if project_root is not None and not destination.is_absolute():
            destination = Path(project_root) / destination

        if project_root is not None:
            image_path = Path(config["data"]["image_path"]).expanduser()

            try:
                relative_image_path = image_path.resolve().relative_to(
                    Path(project_root).resolve()
                )
            except ValueError:
                pass
            else:
                config["data"]["image_path"] = str(relative_image_path)

        provenance = {
            "schema": "baymorph.advisor_proposal",
            "schema_version": 2,
            "decision": "AD-049",
            "baymorph_version": _baymorph_version(),
            "image_path": config["data"]["image_path"],
            "hdu": self.hdu,
            "calibration": {
                name: {
                    "value": value,
                    "unit": DECLARABLE_QUANTITIES[name],
                    "source": "user",
                }
                for name, value in self.calibration.items()
            },
            "target_position": deepcopy(config["data"]["target_position"]),
            "models": list(chosen),
            "rules": {
                "magnitudes": list(MAGNITUDE_RANGE),
                "point_source_snr": POINT_SOURCE_SNR,
                "angular_range_arcsec": list(ANGULAR_RANGE_ARCSEC),
                "physical_length_models": sorted(PHYSICAL_LENGTH_MODELS),
                "centre_fwhm_multiple": CENTRE_FWHM_MULTIPLE,
                "background_sky_scale_multiple": (BACKGROUND_SKY_SCALE_MULTIPLE),
                "psf_peak_saturation_fraction": PSF_PEAK_SATURATION_FRACTION,
                "gain_ratio_tolerance": GAIN_RATIO_TOLERANCE,
            },
            "header": reading.as_dict(),
            "fixed_inputs": fixed_inputs.as_dict(),
            "priors": {
                name: {
                    parameter: proposal.as_dict()
                    for parameter, proposal in proposals[name].items()
                }
                for name in chosen
            },
            "background": {
                **background_record,
                "model": background_model_name,
                "priors": deepcopy(config["fit"]["background"]["priors"]),
            },
            "effective_gain": gain_record,
            "psf_max_peak": max_peak_record,
            "warnings": list(warnings),
        }

        flux_proposal = None if several else proposals[first].get("flux")

        return ObservationConfigProposal(
            config=config,
            output_path=destination,
            starting_model=first,
            flux_prior=(
                None if flux_proposal is None else deepcopy(flux_proposal.prior)
            ),
            flux_prior_source=(
                "see the priors of each flux"
                if flux_proposal is None
                else flux_proposal.detail
            ),
            requires_flux_review=requires_flux_review,
            warnings=tuple(warnings),
            provenance=provenance,
            models=tuple(chosen),
        )

    def _open(self) -> Observation:
        """Open the image, with the metadata of the declared calibration."""
        return open_observation(
            self.fits_path, hdu=self.hdu, declarations=self.calibration
        )

    def _header_reading(self) -> HeaderReading:
        """Return the reading of the header of the image HDU.

        Only the header is read, with the declared calibration.
        """
        return read_image_header(
            self.fits_path, hdu=self.hdu, declarations=self.calibration
        )

    def _resolve_target(
        self, observation: Observation
    ) -> tuple[float | None, float | None]:
        """Resolve the optional target seed for inspection."""
        target = self.target_position

        if target is None:
            return None, None

        if isinstance(target, str):
            if target.strip().lower() != "center":
                raise ValueError("String target_position must be exactly 'center'.")

            return (
                float(observation.metadata.center_x),
                float(observation.metadata.center_y),
            )

        if isinstance(target, Mapping):
            if set(target) != {"x", "y"}:
                raise ValueError(
                    "Target-position mappings must contain exactly 'x' and 'y'."
                )

            return (float(target["x"]), float(target["y"]))

        if len(target) != 2:
            raise ValueError("target_position must contain exactly two coordinates.")

        return (float(target[0]), float(target[1]))


def _half_light_radius(
    source_signal: np.ndarray, *, x0: float, y0: float, q: float, theta: float
) -> float | None:
    """Estimate an elliptical half-light radius from positive signal."""
    y, x = np.indices(source_signal.shape, dtype=float)

    dx = x - float(x0)
    dy = y - float(y0)

    cosine = np.cos(theta)
    sine = np.sin(theta)

    major = cosine * dx + sine * dy

    minor = -sine * dx + cosine * dy

    safe_q = max(float(q), np.finfo(float).eps)

    radius = np.sqrt(major**2 + (minor / safe_q) ** 2)

    signal = np.asarray(source_signal, dtype=float)

    valid = np.isfinite(signal) & (signal > 0.0)

    if not np.any(valid):
        return None

    order = np.argsort(radius[valid])

    ordered_radius = radius[valid][order]

    ordered_signal = signal[valid][order]

    cumulative = np.cumsum(ordered_signal)

    half_total = 0.5 * cumulative[-1]

    index = int(np.searchsorted(cumulative, half_total, side="left"))

    return float(ordered_radius[min(index, ordered_radius.size - 1)])


def _finite_or_none(value: Any) -> float | None:
    """Return a finite float or None."""
    if value is None:
        return None

    result = float(value)

    if not np.isfinite(result):
        return None

    return result


def _format_header_fwhm(value: float | None, unit: str | None) -> str:
    """Format raw header FWHM metadata without inventing a unit."""
    if value is None:
        return "unavailable"

    value_text = f"{float(value):.6g}"

    if unit == "arcsec":
        return f"{value_text} arcsec"

    if unit == "pixels":
        return f"{value_text} pixels"

    return f"{value_text} " "(unit unresolved; not converted)"


def _format_optional(value: float | None, unit: str) -> str:
    """Format an optional scalar quantity."""
    if value is None:
        return "unavailable"

    suffix = f" {unit}" if unit else ""

    return f"{value:.6g}{suffix}"


def _format_bound(value: Any) -> str:
    """Format a prior bound, numeric or symbolic."""
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    ):
        return f"{float(value):.6g}"

    return str(value)


def _format_prior(prior: Mapping[str, Any]) -> str:
    """Format one prior declaration for a human reader."""
    prior_type = str(prior.get("type", "unspecified"))

    if "lower" in prior or "upper" in prior:
        text = (
            f"{prior_type}("
            f"{_format_bound(prior.get('lower'))}, "
            f"{_format_bound(prior.get('upper'))})"
        )
    elif "value" in prior:
        text = f"{prior_type}({_format_bound(prior['value'])})"
    else:
        text = prior_type

    if prior.get("relative_to"):
        text += f" relative to {prior['relative_to']}"

    return text


def _format_declaration(name: str, entry: Mapping[str, Any]) -> str:
    """Format one declared calibration quantity for a human reader."""
    value = entry["value"]
    text = f"{value:.6g}" if isinstance(value, float) else str(value)
    unit = entry.get("unit")

    if unit == IMAGE_UNIT:
        text += " in the units of the image"
    elif unit:
        text += f" {unit}"

    return f"{name} {text} [{entry['source']}]"


def _provenance_summary(provenance: Mapping[str, Any]) -> list[str]:
    """Return the summary lines of the provenance of a proposal."""
    lines = ["", "PRIORS AND THEIR SOURCES"]

    for model_name, entries in provenance.get("priors", {}).items():
        lines.append(f"{model_name}:")
        lines.extend(
            f"  {name}: {_format_prior(entry['prior'])} "
            f"[{entry['source']}] {entry['detail']}"
            for name, entry in entries.items()
        )

    background = provenance.get("background")

    if background:
        lines.extend(
            [
                "",
                "RESIDUAL BACKGROUND",
                (
                    f"Model: {background['model']} [{background['source']}] "
                    f"{background.get('detail', '')}"
                ).rstrip(),
            ]
        )
        lines.extend(
            f"{name}: {_format_prior(prior)}"
            for name, prior in background["priors"].items()
        )

    lines.extend(["", "OBSERVATION"])

    calibration = provenance.get("calibration") or {}

    if calibration:
        lines.append(
            "Declared calibration (data.calibration): "
            + ", ".join(
                _format_declaration(name, entry) for name, entry in calibration.items()
            )
        )

    gain = provenance.get("effective_gain") or {}
    declaration = gain.get("declaration")

    if gain.get("source") == "user":
        lines.append(
            "Effective gain: declared by the user "
            f"({_format_prior(declaration or {})})"
        )
    elif declaration:
        lines.append(
            f"Effective gain: {gain['value']:.6g} from the source "
            f"{declaration['source']!r}"
        )
    else:
        lines.append(
            "Effective gain: not proposed; "
            + str(gain.get("detail", "declare it before running the fit"))
        )

    if gain.get("ratio") is not None:
        expected = gain.get("expected_ratio")
        lines.append(
            f"Gain cross-check: ratio of two sources {gain['ratio']:.3g}"
            + (
                f", predicted {expected:.3g}"
                if expected is not None
                else ", no prediction without a declared resampling factor"
            )
        )

    peak = provenance.get("psf_max_peak")

    if peak:
        value = peak.get("value")
        lines.append(
            "PSF peak ceiling: "
            + ("BayMorph default" if value is None else f"{value:.6g}")
            + f" [{peak['source']}] {peak['detail']}"
        )

    fixed = provenance.get("fixed_inputs")

    if fixed:
        rows, columns = fixed["cutout_shape"]
        lines.extend(
            [
                "",
                "FIXED INPUTS (prepared before any prior)",
                f"Prepared sky scale: {fixed['sky_scale']:.6g}",
                f"Cutout: {columns} x {rows} pixels",
                (
                    f"PSF FWHM: {fixed['psf_fwhm_pixels']:.4g} pixels "
                    f"({fixed['psf_fwhm_source']})"
                ),
                (
                    "PSF noise-equivalent area: "
                    f"{fixed['psf_noise_equivalent_area']:.4g} px^2"
                ),
                f"Products: {fixed['run_directory']}",
            ]
        )

    return lines
