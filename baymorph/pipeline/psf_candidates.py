from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

from .config import load_resolved_config
from .outputs import make_output_tree, pipeline_paths


@dataclass
class PSFCandidate:
    x: float
    y: float
    flux: float | None = None
    fwhm: float | None = None
    sharpness: float | None = None
    roundness: float | None = None


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