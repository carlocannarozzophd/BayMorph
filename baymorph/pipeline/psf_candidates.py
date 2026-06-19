from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

from .config import load_resolved_config
from .outputs import make_output_tree, pipeline_paths
from baymorph.core.image import ImageData
from baymorph.pipeline.image_preparation import prepare_real_fits_image

@dataclass
class PSFCandidate:
    x: float
    y: float
    label: int | None = None
    area: float | None = None
    max_value: float | None = None
    flux: float | None = None
    eccentricity: float | None = None
    elongation: float | None = None
    semimajor_sigma: float | None = None
    semiminor_sigma: float | None = None
    fwhm: float | None = None
    sharpness: float | None = None
    roundness: float | None = None
    half_size: int | None = None


def save_psf_candidates(
    config: dict[str, Any],
    candidates: list[PSFCandidate],
) -> Path:
    make_output_tree(config)
    path = pipeline_paths(config)["psf"]["candidates"]

    with path.open("w") as f:
        json.dump([asdict(candidate) for candidate in candidates], f, indent=2)

    return path


def load_psf_candidates(
    config: str | Path | dict[str, Any],
) -> list[PSFCandidate]:
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = config

    path = pipeline_paths(resolved)["psf"]["candidates"]

    with path.open("r") as f:
        raw_candidates = json.load(f)

    return [PSFCandidate(**candidate) for candidate in raw_candidates]


def run_psf_candidate_selection_from_config(
    config: str | Path | dict[str, Any],
) -> list[PSFCandidate]:
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = config

    make_output_tree(resolved)

    image = ImageData.from_fits(resolved["data"]["image_path"])

    detection_cfg = resolved["preparation"]["detection"]
    background_cfg = resolved["preparation"]["background"]

    prepared = prepare_real_fits_image(
        image.data,
        background_box_size=background_cfg["box_size"],
        background_filter_size=background_cfg["filter_size"],
        detection_sigma=detection_cfg["n_sigma"],
    )

    segmentation = prepared.detection.segmentation
    source_catalog = prepared.detection.source_catalog

    if segmentation is None:
        raise RuntimeError("No segmentation map was produced.")

    if source_catalog is None:
        raise RuntimeError("No source catalog was produced.")

    rows = source_catalog_to_candidate_rows(source_catalog)

    candidate_cfg = resolved["psf"]["empirical"]["candidate_selection"]

    candidates = select_psf_candidates_from_rows(
        rows,
        segmentation.data,
        target_n_candidates=candidate_cfg["target_n_candidates"],
        min_area=candidate_cfg["min_area"],
        max_area=candidate_cfg["max_area"],
        min_peak=candidate_cfg["min_peak"],
        max_peak=candidate_cfg["max_peak"],
        max_eccentricity=candidate_cfg["max_eccentricity"],
        max_elongation=candidate_cfg["max_elongation"],
        edge_margin=candidate_cfg["edge_margin"],
        galaxy_exclusion_box=candidate_cfg["galaxy_exclusion_box"],
        sigma_factor=candidate_cfg["sigma_factor"],
        size_margin=candidate_cfg["size_margin"],
        min_half_size=candidate_cfg["min_half_size"],
        max_half_size=candidate_cfg["max_half_size"],
        isolation_margin=candidate_cfg["isolation_margin"],
    )

    save_psf_candidates(resolved, candidates)

    return candidates


def estimate_candidate_half_size(
    fwhm: float,
    *,
    scale_factor: float = 4.0,
    min_half_size: int = 12,
    max_half_size: int = 40,
) -> int:
    if fwhm <= 0:
        raise ValueError("fwhm must be positive.")

    half_size = int(np.ceil(scale_factor * fwhm))
    return int(np.clip(half_size, min_half_size, max_half_size))


def is_isolated_candidate(
    segmentation: np.ndarray,
    *,
    label: int,
    x: float,
    y: float,
    half_size: int,
) -> bool:
    if half_size <= 0:
        raise ValueError("half_size must be positive.")

    x0 = int(round(x))
    y0 = int(round(y))

    y_min = max(0, y0 - half_size)
    y_max = min(segmentation.shape[0], y0 + half_size + 1)
    x_min = max(0, x0 - half_size)
    x_max = min(segmentation.shape[1], x0 + half_size + 1)

    local = segmentation[y_min:y_max, x_min:x_max]

    labels = set(np.unique(local))
    labels.discard(0)
    labels.discard(label)

    return len(labels) == 0


def select_psf_candidates_from_rows(
    rows: list[dict[str, Any]],
    segmentation: np.ndarray,
    *,
    target_n_candidates: int,
    min_area: float,
    max_area: float,
    min_peak: float,
    max_peak: float,
    max_eccentricity: float,
    max_elongation: float,
    edge_margin: int,
    galaxy_exclusion_box: dict[str, float] | None,
    sigma_factor: float,
    size_margin: int,
    min_half_size: int,
    max_half_size: int,
    isolation_margin: int,
) -> list[PSFCandidate]:
    if target_n_candidates <= 0:
        raise ValueError("target_n_candidates must be positive.")

    selected: list[PSFCandidate] = []

    for row in rows:
        if not candidate_row_passes_quality_cuts(
            row,
            segmentation.shape,
            min_area=min_area,
            max_area=max_area,
            min_peak=min_peak,
            max_peak=max_peak,
            max_eccentricity=max_eccentricity,
            max_elongation=max_elongation,
            edge_margin=edge_margin,
            galaxy_exclusion_box=galaxy_exclusion_box,
        ):
            continue

        half_size = estimate_candidate_half_size_from_sigmas(
            float(row["semimajor_sigma"]),
            float(row["semiminor_sigma"]),
            sigma_factor=sigma_factor,
            size_margin=size_margin,
            min_half_size=min_half_size,
            max_half_size=max_half_size,
        )

        isolation_half_size = half_size + int(isolation_margin)

        if not is_isolated_candidate(
            segmentation,
            label=int(row["label"]),
            x=float(row["x"]),
            y=float(row["y"]),
            half_size=isolation_half_size,
        ):
            continue

        selected.append(
            PSFCandidate(
                x=float(row["x"]),
                y=float(row["y"]),
                label=int(row["label"]),
                area=float(row["area"]),
                max_value=float(row["max_value"]),
                flux=row.get("flux"),
                eccentricity=float(row["eccentricity"]),
                elongation=float(row["elongation"]),
                semimajor_sigma=float(row["semimajor_sigma"]),
                semiminor_sigma=float(row["semiminor_sigma"]),
                half_size=half_size,
            )
        )

    selected = sorted(
        selected,
        key=lambda candidate: (
            candidate.eccentricity,
            candidate.elongation,
            -candidate.max_value,
        ),
    )

    return selected[:target_n_candidates]


def is_inside_exclusion_box(
    x: float,
    y: float,
    box: dict[str, float] | None,
) -> bool:
    if box is None:
        return False

    return (
        box["xmin"] <= x <= box["xmax"]
        and box["ymin"] <= y <= box["ymax"]
    )


def estimate_candidate_half_size_from_sigmas(
    semimajor_sigma: float,
    semiminor_sigma: float,
    *,
    sigma_factor: float,
    size_margin: int,
    min_half_size: int,
    max_half_size: int,
) -> int:
    if semimajor_sigma <= 0 or semiminor_sigma <= 0:
        raise ValueError("semimajor_sigma and semiminor_sigma must be positive.")

    sigma = max(semimajor_sigma, semiminor_sigma)
    half_size = int(np.ceil(sigma_factor * sigma)) + int(size_margin)

    return int(np.clip(half_size, min_half_size, max_half_size))


def candidate_row_passes_quality_cuts(
    row: dict[str, Any],
    image_shape: tuple[int, int],
    *,
    min_area: float,
    max_area: float,
    min_peak: float,
    max_peak: float,
    max_eccentricity: float,
    max_elongation: float,
    edge_margin: int,
    galaxy_exclusion_box: dict[str, float] | None,
) -> bool:
    x = float(row["x"])
    y = float(row["y"])

    ny, nx = image_shape

    if x <= edge_margin or x >= nx - edge_margin:
        return False

    if y <= edge_margin or y >= ny - edge_margin:
        return False

    if is_inside_exclusion_box(x, y, galaxy_exclusion_box):
        return False

    area = float(row["area"])
    max_value = float(row["max_value"])
    eccentricity = float(row["eccentricity"])
    elongation = float(row["elongation"])

    return (
        min_area <= area <= max_area
        and min_peak <= max_value <= max_peak
        and eccentricity < max_eccentricity
        and elongation < max_elongation
    )


def source_catalog_to_candidate_rows(source_catalog) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for source in source_catalog:
        rows.append(
            {
                "label": int(source.label),
                "x": float(source.xcentroid.value),
                "y": float(source.ycentroid.value),
                "area": float(source.area.value),
                "max_value": float(source.max_value),
                "eccentricity": float(source.eccentricity.value),
                "elongation": float(source.elongation.value),
                "semimajor_sigma": float(source.semimajor_sigma.value),
                "semiminor_sigma": float(source.semiminor_sigma.value),
                "flux": float(source.segment_flux.value),
            }
        )

    return rows