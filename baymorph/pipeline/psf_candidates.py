from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

from .config import load_resolved_config
from .outputs import make_output_tree, pipeline_paths


@dataclass
class PSFCandidate:
    x: float
    y: float
    label: int | None = None
    flux: float | None = None
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

    raise NotImplementedError(
        "PSF candidate selection orchestration is not implemented yet."
    )


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
    isolation_scale_factor: float,
    min_half_size: int,
    max_half_size: int,
) -> list[PSFCandidate]:
    if target_n_candidates <= 0:
        raise ValueError("target_n_candidates must be positive.")

    candidates: list[PSFCandidate] = []

    for row in rows:
        label = int(row["label"])
        x = float(row["x"])
        y = float(row["y"])
        fwhm = float(row["fwhm"])

        half_size = estimate_candidate_half_size(
            fwhm,
            scale_factor=isolation_scale_factor,
            min_half_size=min_half_size,
            max_half_size=max_half_size,
        )

        if not is_isolated_candidate(
            segmentation,
            label=label,
            x=x,
            y=y,
            half_size=half_size,
        ):
            continue

        candidates.append(
            PSFCandidate(
                x=x,
                y=y,
                label=label,
                flux=row.get("flux"),
                fwhm=fwhm,
                sharpness=row.get("sharpness"),
                roundness=row.get("roundness"),
                half_size=half_size,
            )
        )

    candidates = sorted(
        candidates,
        key=lambda candidate: (
            -np.inf if candidate.flux is None else -candidate.flux,
            candidate.fwhm if candidate.fwhm is not None else np.inf,
        ),
    )

    return candidates[:target_n_candidates]