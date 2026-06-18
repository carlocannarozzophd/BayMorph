from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from .config import load_resolved_config
from .outputs import make_output_tree, pipeline_paths


def save_empirical_psf(
    config: dict[str, Any],
    psf: np.ndarray,
) -> Path:
    make_output_tree(config)
    path = pipeline_paths(config)["psf"]["psf_fits"]

    overwrite = bool(config["outputs"].get("overwrite", False))

    fits.writeto(path, psf, overwrite=overwrite)

    return path


def load_empirical_psf(
    config: str | Path | dict[str, Any],
) -> np.ndarray:
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = config

    path = pipeline_paths(resolved)["psf"]["psf_fits"]

    return fits.getdata(path)


def run_psf_construction_from_config(
    config: str | Path | dict[str, Any],
) -> np.ndarray:
    raise NotImplementedError(
        "Empirical PSF construction orchestration is not implemented yet."
    )