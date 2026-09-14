"""Evidence-aware observation inspection and configuration guidance."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from baymorph.analysis.prior_predictive import (
    FluxPriorCalibration,
    calibrate_single_sersic_flux_prior,
)
from baymorph.core.observation import Observation
from baymorph.pipeline.config import (
    DEFAULT_CONFIG,
    load_resolved_config,
    resolve_config,
)
from baymorph.pipeline.config_maker import make_config_from_fits
from baymorph.pipeline.image_preparation import (
    load_image_preparation_products,
    prepare_real_fits_image,
    prepare_real_image_from_config,
)
from baymorph.pipeline.psf_candidates import (
    run_psf_candidate_selection_from_config,
)
from baymorph.pipeline.psf_construction import (
    load_empirical_psf,
    run_psf_construction_from_config,
)
from baymorph.pipeline.preparation_steps import (
    estimate_target_geometry_from_image,
)
from baymorph.pipeline.segmentation import select_segment_near_position


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


PRIOR_WIDTH_WARNING_DEX = 6.0


@dataclass(frozen=True, slots=True)
class _PriorAudit:
    """Internal audit of one declared scientific prior."""

    status: str
    width_dex: float | None = None
    observed_inside_bounds: bool | None = None
    lower_margin_dex: float | None = None
    upper_margin_dex: float | None = None
    warnings: tuple[str, ...] = ()


def _numeric_prior_bound(
    value: Any,
) -> float | None:
    """Resolve numeric and supported symbolic prior bounds."""
    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        result = float(value)

        return result if np.isfinite(result) else None

    symbolic = {
        "-pi": -np.pi,
        "-pi/2": -0.5 * np.pi,
        "0": 0.0,
        "pi/2": 0.5 * np.pi,
        "pi": np.pi,
    }

    return symbolic.get(
        str(value)
        .strip()
        .lower()
        .replace(
            " ",
            "",
        )
    )


def _audit_prior(
    prior: Mapping[str, Any],
    *,
    observed_scale: float | None = None,
) -> _PriorAudit:
    """Audit prior volume without using the observation to define bounds."""
    prior_type = (
        str(
            prior.get(
                "type",
                "",
            )
        )
        .strip()
        .lower()
    )

    lower = _numeric_prior_bound(prior.get("lower"))

    upper = _numeric_prior_bound(prior.get("upper"))

    warnings: list[str] = []

    width_dex = None
    observed_inside_bounds = None
    lower_margin_dex = None
    upper_margin_dex = None

    if (
        prior_type == "log_uniform"
        and lower is not None
        and upper is not None
        and lower > 0.0
        and upper > lower
    ):
        width_dex = float(np.log10(upper / lower))

        if width_dex >= PRIOR_WIDTH_WARNING_DEX:
            warnings.append(
                "prior spans "
                f"{width_dex:.3f} dex; "
                "Bayesian evidence may be strongly prior-volume sensitive"
            )

    if (
        observed_scale is not None
        and np.isfinite(observed_scale)
        and lower is not None
        and upper is not None
    ):
        observed_inside_bounds = bool(lower <= observed_scale <= upper)

        if not observed_inside_bounds:
            warnings.append(
                "observed diagnostic scale lies outside the declared prior support"
            )

        if (
            prior_type == "log_uniform"
            and observed_scale > 0.0
            and lower > 0.0
            and upper > 0.0
            and observed_inside_bounds
        ):
            lower_margin_dex = float(np.log10(observed_scale / lower))

            upper_margin_dex = float(np.log10(upper / observed_scale))

    return _PriorAudit(
        status=("WARNING" if warnings else "REVIEWED"),
        width_dex=width_dex,
        observed_inside_bounds=observed_inside_bounds,
        lower_margin_dex=lower_margin_dex,
        upper_margin_dex=upper_margin_dex,
        warnings=tuple(warnings),
    )


def _prior_volume_position(
    prior: Mapping[str, Any],
    observed_scale: float | None,
) -> float | None:
    """Return an observed scale's position in the declared prior measure."""
    if observed_scale is None:
        return None

    value = float(observed_scale)

    lower = _numeric_prior_bound(prior.get("lower"))
    upper = _numeric_prior_bound(prior.get("upper"))

    if lower is None or upper is None or not np.isfinite(value) or not lower < upper:
        return None

    prior_type = (
        str(
            prior.get(
                "type",
                "uniform",
            )
        )
        .strip()
        .lower()
    )

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
    position = _prior_volume_position(
        prior,
        observed_scale,
    )

    if position is None:
        return False

    return bool(position <= edge_fraction or position >= (1.0 - edge_fraction))


def _print_prior_audit(
    prior: Mapping[str, Any],
    *,
    observed_scale: float | None = None,
) -> _PriorAudit:
    """Print an evidence-oriented audit of one declared prior."""
    audit = _audit_prior(
        prior,
        observed_scale=observed_scale,
    )

    print(f"  bounds: [{prior.get('lower')}, {prior.get('upper')}]")

    if audit.width_dex is not None:
        print(f"  prior width: {audit.width_dex:.3f} dex")

    if audit.observed_inside_bounds is not None:
        print(
            "  observed diagnostic scale inside bounds: "
            + ("YES" if audit.observed_inside_bounds else "NO")
        )

    if audit.lower_margin_dex is not None and audit.upper_margin_dex is not None:
        print(
            "  observed-scale position: "
            f"{audit.lower_margin_dex:.3f} dex above lower bound, "
            f"{audit.upper_margin_dex:.3f} dex below upper bound"
        )

    print("  observed diagnostic used to define bounds: NO")

    print(f"  evidence review: {audit.status}")

    for warning in audit.warnings:
        print(f"  warning: {warning}")

    return audit


@dataclass(frozen=True, slots=True)
class ObservationConfigProposal:
    """Human-reviewable proposal for one observed-image configuration.

    Creating a proposal never writes a file. The user must call
    :meth:`write` with ``accept=True`` after reviewing the starting model,
    prior source, and warnings.
    """

    config: dict[str, Any]
    output_path: Path
    starting_model: str
    flux_prior: dict[str, Any]
    flux_prior_source: str
    requires_flux_review: bool
    warnings: tuple[str, ...] = ()

    def summary(
        self,
    ) -> str:
        """Return a concise proposal summary for human review."""
        target = self.config["data"]["target_position"]

        lines = [
            "BayMorph Configuration Proposal",
            "",
            f"Output YAML: {self.output_path}",
            (f"FITS: {self.config['data']['image_path']}"),
            (
                "Confirmed target: "
                f"X={float(target['x']):.3f}, "
                f"Y={float(target['y']):.3f}"
            ),
            (f"User-selected starting model: {self.starting_model}"),
            (f"Flux prior source: {self.flux_prior_source}"),
            (
                "Flux prior: "
                f"{self.flux_prior['type']}("
                f"{self.flux_prior['lower']}, "
                f"{self.flux_prior['upper']})"
            ),
            (
                "Flux coverage review: "
                + ("REQUIRED" if self.requires_flux_review else "clear")
            ),
            "",
            "MODEL-SCREENING SCOPE",
            (
                "The starting model is a user decision, not a pre-fit "
                "Bayesian model preference."
            ),
            (
                "A preferred model can be established only through the "
                "configured inference and evidence comparison."
            ),
        ]

        if self.warnings:
            lines.extend(
                [
                    "",
                    "WARNINGS",
                ]
            )
            lines.extend(f"- {warning}" for warning in self.warnings)

        lines.extend(
            [
                "",
                "No file has been written.",
            ]
        )

        if self.requires_flux_review:
            lines.extend(
                [
                    (
                        "Writing is blocked until a reviewed scientific "
                        "flux prior is supplied."
                    ),
                    (
                        "Build a new proposal with flux_prior=... "
                        "before writing."
                    ),
                ]
            )
        else:
            lines.append(
                "Write only after review with proposal.write(accept=True)."
            )

        return "\n".join(lines)

    def _repr_html_(
        self,
    ) -> str:
        """Return a readable notebook representation."""
        from html import escape

        return "<pre>" + escape(self.summary()) + "</pre>"

    def write(
        self,
        *,
        accept: bool = False,
    ) -> Path:
        """Write the proposed YAML after explicit user acceptance."""
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
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with destination.open(
            "w",
            encoding="utf-8",
        ) as stream:
            yaml.safe_dump(
                self.config,
                stream,
                sort_keys=False,
            )

        return destination.resolve()


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
        self.fits_path = Path(fits_path)

        if not self.fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {self.fits_path}")

        self.target_position = target_position
        self._inspection: ObservationInspection | None = None

    @classmethod
    def from_project(
        cls,
        fits_path: str | Path,
        *,
        start: str | Path | None = None,
        target_position: TargetPosition = None,
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

        Returns
        -------
        ObservationAdvisor
            Project-aware Advisor.
        """
        supplied_path = Path(fits_path).expanduser()
        search_start = (
            Path.cwd() if start is None else Path(start).expanduser()
        ).resolve()

        search_directories = (
            search_start,
            *search_start.parents,
        )

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
        )
        advisor.project_root = project_root

        return advisor

    def set_target_position(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
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
        observation = Observation.from_fits(self.fits_path)
        data = np.asarray(
            observation.image.data,
            dtype=float,
        )

        if data.ndim != 2:
            raise ValueError("Target selection requires a two-dimensional FITS image.")

        x = float(x)
        y = float(y)

        if not np.all(
            np.isfinite(
                [
                    x,
                    y,
                ]
            )
        ):
            raise ValueError("Target coordinates must be finite.")

        height, width = data.shape

        if not (0.0 <= x <= width - 1 and 0.0 <= y <= height - 1):
            raise ValueError(
                "Target coordinates must lie inside the FITS image: "
                f"0 <= x <= {width - 1} and "
                f"0 <= y <= {height - 1}."
            )

        position = (
            x,
            y,
        )

        self.target_position = position
        self._inspection = None

        return position

    def clear_target_position(
        self,
    ) -> None:
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

        observation = Observation.from_fits(self.fits_path)
        data = np.asarray(
            observation.image.data,
            dtype=float,
        )

        if data.ndim != 2:
            raise ValueError(
                "Interactive target selection requires a two-dimensional FITS image."
            )

        if initial_position is None and self.target_position is not None:
            target_x, target_y = self._resolve_target(observation)

            if target_x is not None and target_y is not None:
                initial_position = (
                    float(target_x),
                    float(target_y),
                )

        def confirm_target(
            position: tuple[float, float],
        ) -> None:
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

    def inspect(
        self,
    ) -> ObservationInspection:
        """Inspect metadata, image scales, and optional target morphology."""
        observation = Observation.from_fits(self.fits_path)

        data = np.asarray(
            observation.image.data,
            dtype=float,
        )

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
            label = select_segment_near_position(
                segmentation,
                x=target_x,
                y=target_y,
            )

            target_segment = segmentation.data == label

            target_area_pixels = int(np.count_nonzero(target_segment))

            geometry = estimate_target_geometry_from_image(
                prepared.data,
                target_segment,
            )

            target_axis_ratio = float(geometry.q)

            target_position_angle = float(geometry.theta)

            source_signal = np.where(
                target_segment,
                np.clip(
                    prepared.data,
                    0.0,
                    None,
                ),
                0.0,
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
            shape=(
                int(data.shape[0]),
                int(data.shape[1]),
            ),
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

        inspection = self._inspection or self.inspect()

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
                observed_scale=(
                    None
                    if inspection.target_position_angle is None
                    else float(
                        (inspection.target_position_angle + 0.5 * np.pi) % np.pi
                        - 0.5 * np.pi
                    )
                ),
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
                + _format_optional(
                    inspection.pixel_scale,
                    "arcsec/pixel",
                )
            ),
            (
                "Header FWHM: "
                + _format_header_fwhm(
                    inspection.seeing_fwhm,
                    inspection.seeing_fwhm_unit,
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
                        + _format_optional(
                            inspection.target_axis_ratio,
                            "",
                        )
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
            raise ValueError("target_position is required to create a configuration.")

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
            If True, run canonical PSF-candidate selection followed by
            empirical-PSF construction before calibration. Otherwise load
            the existing empirical PSF.
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
            model_name = str(
                config_options.pop(
                    "model_name",
                    "single_sersic",
                )
            )

            if model_name != "single_sersic":
                raise NotImplementedError(
                    "Flux prior calibration currently supports only 'single_sersic'."
                )

            # Resolve relative output paths before make_config() can bind
            # them to the notebook kernel's working directory.
            project_root = getattr(self, "project_root", None)
            output_root = Path(
                config_options.get("outputs_root", "runs")
            ).expanduser()

            if project_root is not None and not output_root.is_absolute():
                config_options["outputs_root"] = str(
                    (Path(project_root) / output_root).resolve()
                )

            workspace = self.make_config(
                model_name=model_name,
                model_priors={
                    "flux": {
                        "type": "fixed",
                        "value": 1.0,
                    },
                },
                **config_options,
            )

            resolved = resolve_config(workspace)

        else:
            if config_options:
                raise TypeError(
                    "Configuration options may be supplied only when config=None."
                )

            if isinstance(
                config,
                (str, Path),
            ):
                resolved = load_resolved_config(config)
            else:
                resolved = resolve_config(dict(config))

        if prepare:
            products = prepare_real_image_from_config(
                resolved,
                save=True,
            )
        else:
            products = load_image_preparation_products(resolved)

        if build_psf:
            run_psf_candidate_selection_from_config(resolved)
            run_psf_construction_from_config(resolved)

        psf = load_empirical_psf(resolved)

        inspection = self._inspection or self.inspect()

        if inspection.saturation is None:
            raise ValueError(
                "Flux upper-bound calibration requires finite detector "
                "saturation metadata."
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
        priors.pop(
            "flux",
            None,
        )

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

    def propose_config(
        self,
        *,
        output_path: str | Path,
        model_name: str = "single_sersic",
        flux_calibration: FluxPriorCalibration | None = None,
        flux_prior: Mapping[str, Any] | None = None,
        model_priors: Mapping[str, Any] | None = None,
        **config_options: Any,
    ) -> ObservationConfigProposal:
        """Build a reviewable configuration proposal without writing it.

        Exactly one total-flux prior source is required: either an
        observational :class:`FluxPriorCalibration`, an explicit
        ``flux_prior``, or a ``flux`` entry in ``model_priors``.

        The chosen model remains an explicit user-selected starting model.
        BayMorph does not interpret it as a pre-fit evidence preference.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Destination used only if the returned proposal is explicitly
            accepted and written.
        model_name : str, default="single_sersic"
            User-selected starting model.
        flux_calibration : FluxPriorCalibration, optional
            Observational detectability-and-saturation calibration.
        flux_prior : mapping, optional
            Explicit scientific total-flux prior.
        model_priors : mapping, optional
            Additional explicit model-prior overrides.
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

        priors = dict(model_priors or {})
        embedded_flux_prior = priors.pop(
            "flux",
            None,
        )

        supplied_sources = sum(
            source is not None
            for source in (
                flux_calibration,
                flux_prior,
                embedded_flux_prior,
            )
        )

        if supplied_sources != 1:
            raise ValueError(
                "Supply exactly one flux-prior source: flux_calibration, "
                "flux_prior, or model_priors['flux']."
            )

        warnings = []
        flux_prior_source: str

        if flux_calibration is not None:
            candidate = flux_calibration.candidate_prior

            if candidate is None:
                raise ValueError(
                    "Flux calibration did not produce a valid candidate prior."
                )

            selected_flux_prior = dict(candidate)
            flux_prior_source = "observational detectability and saturation calibration"
        elif flux_prior is not None:
            selected_flux_prior = dict(flux_prior)
            flux_prior_source = "explicit user-declared scientific prior"
        else:
            selected_flux_prior = dict(embedded_flux_prior)
            flux_prior_source = "explicit user-declared scientific prior"

        likelihood = config_options.get("likelihood")

        if (
            isinstance(
                likelihood,
                Mapping,
            )
            and likelihood.get(
                "include_normalization",
                True,
            )
            is False
        ):
            raise ValueError(
                "ObservationAdvisor.propose_config() requires a normalized "
                "likelihood for standard evidence interpretation."
            )

        inspection = self._inspection or self.inspect()

        requires_flux_review = _flux_prior_requires_review(
            selected_flux_prior,
            inspection.target_flux_scale,
        )

        if requires_flux_review:
            warnings.append(
                "The observed target flux diagnostic lies outside or near "
                "an edge of the proposed flux-prior measure. The prior was "
                "not widened automatically; posterior boundary diagnostics "
                "must be reviewed before interpreting the evidence."
            )

        warnings.append(
            "No validated model-screening classifier is distributed with "
            "BayMorph. The starting model is therefore a user decision."
        )

        priors["flux"] = selected_flux_prior

        config = self.make_config(
            output_path=None,
            model_name=model_name,
            model_priors=priors,
            **config_options,
        )

        destination = Path(output_path).expanduser()
        project_root = getattr(
            self,
            "project_root",
            None,
        )

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

        return ObservationConfigProposal(
            config=config,
            output_path=destination,
            starting_model=str(model_name),
            flux_prior=selected_flux_prior,
            flux_prior_source=flux_prior_source,
            requires_flux_review=requires_flux_review,
            warnings=tuple(warnings),
        )

    def configure(
        self,
        *,
        output_path: str | Path | None = None,
        input_fn: InputFunction = input,
        flux_calibration: FluxPriorCalibration | None = None,
        **config_options: Any,
    ) -> dict[str, Any] | None:
        """Interactively create an evidence-aware single-Sersic config.

        Image-derived numerical scales are used only to audit declared priors.
        They never determine scientific-prior bounds automatically.

        A ``WARNING`` status identifies a prior that deserves explicit review;
        it does not imply that the prior is mathematically invalid.
        """
        answer = (
            input_fn("Create a BayMorph configuration now? [y/N]: ").strip().lower()
        )

        if answer not in {
            "y",
            "yes",
        }:
            return None

        guidance = self.prior_guidance("single_sersic")

        print()
        print("PRIOR PLAN")
        print()

        reference_priors = DEFAULT_CONFIG["fit"]["priors"]["single_sersic"]

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
            item = guidance.get(parameter)

            print(parameter)

            if item is not None:
                print(f"  suggested family: {item.family}")

                if item.observed_scale is not None:
                    if parameter == "theta":
                        observed_text = (
                            f"{np.degrees(item.observed_scale):.6g} deg "
                            "(axial equivalent in prior domain)"
                        )
                    else:
                        observed_text = f"{item.observed_scale:.6g}"

                    print(f"  observed numerical scale: {observed_text}")

            if parameter == "flux":
                if (
                    flux_calibration is not None
                    and flux_calibration.candidate_prior is not None
                ):
                    calibrated = flux_calibration.candidate_prior

                    print(
                        "  calibrated candidate: "
                        f"{calibrated['type']}("
                        f"{calibrated['lower']:.6g}, "
                        f"{calibrated['upper']:.6g})"
                    )
                    print("  measured target diagnostic used to define bounds: NO")
                    print("  calibration status: CALIBRATED")
                else:
                    print("  prior bounds: USER DECISION")
                    print("  observed diagnostic used to define bounds: NO")
                    print("  evidence review: PENDING USER BOUNDS")
            else:
                prior = reference_priors[parameter]

                prior_text = f"{prior['type']}({prior['lower']}, {prior['upper']})"

                if "relative_to" in prior:
                    prior_text += f" relative to {prior['relative_to']}"

                print(f"  BayMorph reference prior: {prior_text}")

                _print_prior_audit(
                    prior,
                    observed_scale=(None if item is None else item.observed_scale),
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
            include_normalization = likelihood_config.get(
                "include_normalization",
                True,
            )

            if include_normalization is False:
                print("EVIDENCE WARNING")
                print("The requested likelihood excludes its normalization term.")
                print(
                    "ObservationAdvisor requires a normalized "
                    "likelihood for standard Bayesian-evidence "
                    "interpretation."
                )

                normalize = (
                    input_fn("Enable likelihood normalization? [Y/n]: ").strip().lower()
                )

                if normalize in {
                    "",
                    "y",
                    "yes",
                }:
                    likelihood_config = dict(likelihood_config)
                    likelihood_config["include_normalization"] = True

                    config_options["likelihood"] = likelihood_config
                else:
                    raise ValueError(
                        "ObservationAdvisor.configure() requires a "
                        "normalized likelihood. Use make_config() or "
                        "make_config_from_fits() directly only if an "
                        "unnormalized likelihood is scientifically "
                        "intentional."
                    )

        flux_scale = guidance["flux"].observed_scale

        if flux_scale is not None:
            print(f"Observed target flux scale (diagnostic only): {flux_scale:.6g}")

        use_calibrated = False

        if flux_calibration is not None:
            calibrated_prior = flux_calibration.candidate_prior

            if calibrated_prior is None:
                raise ValueError(
                    "Flux calibration did not produce a valid ordered "
                    "log-uniform interval."
                )

            print()
            print("CALIBRATED FLUX PRIOR")
            print(f"  lower bound: {flux_calibration.chosen_flux_lower:.6g}")
            print(f"  upper bound: {flux_calibration.chosen_flux_upper:.6g}")

            if flux_calibration.width_dex is not None:
                print(f"  prior width: {flux_calibration.width_dex:.3f} dex")

            if flux_calibration.lower_confidence_interval is not None:
                lower_ci = flux_calibration.lower_confidence_interval

                print(
                    f"  lower-bound MC interval: [{lower_ci[0]:.6g}, {lower_ci[1]:.6g}]"
                )

            if flux_calibration.upper_confidence_interval is not None:
                upper_ci = flux_calibration.upper_confidence_interval

                print(
                    f"  upper-bound MC interval: [{upper_ci[0]:.6g}, {upper_ci[1]:.6g}]"
                )

            print("  measured target diagnostic used to define bounds: NO")
            print("  calibration status: CALIBRATED")

            prior_position = _prior_volume_position(
                calibrated_prior,
                flux_scale,
            )

            requires_review = _flux_prior_requires_review(
                calibrated_prior,
                flux_scale,
            )

            if prior_position is not None:
                print(
                    f"  observed diagnostic prior-volume position: {prior_position:.3%}"
                )

            if requires_review:
                print()
                print("CALIBRATION COVERAGE WARNING")
                print(
                    "The observed target flux diagnostic lies within "
                    "the outer 5% of the calibrated prior measure or "
                    "outside its bounds."
                )
                print(
                    "Automatic acceptance is unsafe because the fitted "
                    "posterior may contact a prior boundary."
                )
                print(
                    "BayMorph will not widen the prior using the target "
                    "diagnostic. Supply a sample-level scientific prior "
                    "or explicitly accept this candidate for diagnostic "
                    "use only."
                )

                answer = (
                    input_fn(
                        "Use calibrated flux prior for a diagnostic run "
                        "despite this warning? [y/N]: "
                    )
                    .strip()
                    .lower()
                )

                use_calibrated = answer in {
                    "y",
                    "yes",
                }

            else:
                answer = input_fn("Use calibrated flux prior? [Y/n]: ").strip().lower()

                use_calibrated = answer in {
                    "",
                    "y",
                    "yes",
                }

        if use_calibrated:
            flux_prior = dict(flux_calibration.candidate_prior)

            print()
            print("FLUX PRIOR AUDIT")
            print("  source: observational prior calibration")
            print("  measured target diagnostic used to define bounds: NO")
            print("  evidence review: CALIBRATED")
        else:
            prior_type = input_fn("Flux prior type [log_uniform]: ").strip().lower()

            if not prior_type:
                prior_type = "log_uniform"

            if prior_type not in {
                "uniform",
                "log_uniform",
            }:
                raise ValueError("Flux prior type must be 'uniform' or 'log_uniform'.")

            lower = _prompt_float(
                input_fn,
                "Flux prior lower bound (scientific choice): ",
            )

            upper = _prompt_float(
                input_fn,
                "Flux prior upper bound (scientific choice): ",
            )

            if not lower < upper:
                raise ValueError("Flux prior requires lower < upper.")

            if prior_type == "log_uniform" and lower <= 0.0:
                raise ValueError("Log-uniform flux prior requires lower > 0.")

            flux_prior = {
                "type": prior_type,
                "lower": lower,
                "upper": upper,
            }

            print()
            print("FLUX PRIOR AUDIT")

            if prior_type == "log_uniform":
                print(
                    "  interpretation: equal prior probability per logarithmic interval"
                )

            flux_audit = _print_prior_audit(
                flux_prior,
                observed_scale=flux_scale,
            )

            if flux_audit.status == "WARNING":
                print()
                proceed = (
                    input_fn("Proceed with this warned flux prior? [y/N]: ")
                    .strip()
                    .lower()
                )

                if proceed not in {
                    "y",
                    "yes",
                }:
                    print(
                        "Configuration cancelled: warned flux prior was not accepted."
                    )
                    return None

        model_priors = {
            "flux": flux_prior,
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
            if target.strip().lower() != "center":
                raise ValueError("String target_position must be exactly 'center'.")

            return (
                float(observation.metadata.center_x),
                float(observation.metadata.center_y),
            )

        if isinstance(
            target,
            Mapping,
        ):
            if set(target) != {
                "x",
                "y",
            }:
                raise ValueError(
                    "Target-position mappings must contain exactly 'x' and 'y'."
                )

            return (
                float(target["x"]),
                float(target["y"]),
            )

        if len(target) != 2:
            raise ValueError("target_position must contain exactly two coordinates.")

        return (
            float(target[0]),
            float(target[1]),
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

    dx = x - float(x0)
    dy = y - float(y0)

    cosine = np.cos(theta)
    sine = np.sin(theta)

    major = cosine * dx + sine * dy

    minor = -sine * dx + cosine * dy

    safe_q = max(
        float(q),
        np.finfo(float).eps,
    )

    radius = np.sqrt(major**2 + (minor / safe_q) ** 2)

    signal = np.asarray(
        source_signal,
        dtype=float,
    )

    valid = np.isfinite(signal) & (signal > 0.0)

    if not np.any(valid):
        return None

    order = np.argsort(radius[valid])

    ordered_radius = radius[valid][order]

    ordered_signal = signal[valid][order]

    cumulative = np.cumsum(ordered_signal)

    half_total = 0.5 * cumulative[-1]

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

    result = float(value)

    if not np.isfinite(result):
        return None

    return result


def _format_header_fwhm(
    value: float | None,
    unit: str | None,
) -> str:
    """Format raw header FWHM metadata without inventing a unit."""
    if value is None:
        return "unavailable"

    value_text = f"{float(value):.6g}"

    if unit == "arcsec":
        return f"{value_text} arcsec"

    if unit == "pixels":
        return f"{value_text} pixels"

    return (
        f"{value_text} "
        "(unit unresolved; not converted)"
    )


def _format_optional(
    value: float | None,
    unit: str,
) -> str:
    """Format an optional scalar quantity."""
    if value is None:
        return "unavailable"

    suffix = f" {unit}" if unit else ""

    return f"{value:.6g}{suffix}"


def _prompt_float(
    input_fn: InputFunction,
    prompt: str,
) -> float:
    """Read one finite floating-point value."""
    value = float(input_fn(prompt).strip())

    if not np.isfinite(value):
        raise ValueError("Prior bounds must be finite.")

    return value
