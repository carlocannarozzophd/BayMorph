from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_resolved_config
from .outputs import make_output_tree, save_resolved_config
from .image_preparation import prepare_real_image_from_config
from .psf_candidates import run_psf_candidate_selection_from_config
from .psf_construction import run_psf_construction_from_config

def _coerce_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        return load_resolved_config(config)

    if isinstance(config, dict):
        return config

    raise TypeError("config must be a path or a dictionary.")


def initialize_pipeline_run(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    resolved = _coerce_config(config)
    make_output_tree(resolved)
    save_resolved_config(resolved)
    return resolved


def run_image_preparation(config: str | Path | dict[str, Any]):
    return prepare_real_image_from_config(config)


def run_psf_candidate_selection(config: str | Path | dict[str, Any]):
    return run_psf_candidate_selection_from_config(config)


def run_psf_construction(config: str | Path | dict[str, Any]):
    return run_psf_construction_from_config(config)


def run_real_fit(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    return initialize_pipeline_run(config)