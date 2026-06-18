from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
from astropy.io import fits

from .config import load_resolved_config
from .outputs import make_output_tree, pipeline_paths

from baymorph.core.image import ImageData

from .psf_candidates import load_psf_candidates

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
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = config

    image = ImageData.from_fits(
        resolved["data"]["image_path"]
    )

    candidates = load_psf_candidates(resolved)

    if len(candidates) == 0:
        raise ValueError("No PSF candidates available.")

    cutouts = []
    used_candidates = []

    for candidate in candidates:
        if candidate.half_size is None:
            raise ValueError(
                "PSF candidate is missing half_size."
            )

        cutout = extract_centered_cutout(
            image.data,
            x=candidate.x,
            y=candidate.y,
            half_size=candidate.half_size,
        )

        cutout = subtract_outer_background(cutout)

        cutout = recenter_cutout(cutout)

        cutouts.append(cutout)

        used_candidates.append(
            {
                "label": candidate.label,
                "x": candidate.x,
                "y": candidate.y,
                "half_size": candidate.half_size,
            }
        )

    psf = stack_psf_cutouts(cutouts)

    save_empirical_psf(
        resolved,
        psf,
    )

    save_empirical_psf_metadata(
        resolved,
        {
            "n_candidates_input": len(candidates),
            "n_candidates_used": len(cutouts),
            "psf_shape": list(psf.shape),
            "psf_sum": float(np.sum(psf)),
            "used_candidates": used_candidates,
        },
    )

    return psf


def normalize_psf(psf: np.ndarray) -> np.ndarray:
    total = np.sum(psf)

    if not np.isfinite(total) or total <= 0:
        raise ValueError("PSF sum must be positive and finite.")

    return psf / total


def extract_centered_cutout(
    image: np.ndarray,
    x: float,
    y: float,
    half_size: int,
) -> np.ndarray:
    x0 = int(round(x))
    y0 = int(round(y))

    y_min = y0 - half_size
    y_max = y0 + half_size + 1
    x_min = x0 - half_size
    x_max = x0 + half_size + 1

    if x_min < 0 or y_min < 0 or x_max > image.shape[1] or y_max > image.shape[0]:
        raise ValueError("Requested PSF cutout extends outside the image boundaries.")

    return image[y_min:y_max, x_min:x_max]


def subtract_outer_background(
    cutout: np.ndarray,
    border_width: int = 3,
) -> np.ndarray:
    if border_width <= 0:
        raise ValueError("border_width must be positive.")

    if 2 * border_width >= min(cutout.shape):
        raise ValueError("border_width is too large for this cutout.")

    border_pixels = np.concatenate(
        [
            cutout[:border_width, :].ravel(),
            cutout[-border_width:, :].ravel(),
            cutout[:, :border_width].ravel(),
            cutout[:, -border_width:].ravel(),
        ]
    )

    background = np.nanmedian(border_pixels)

    return cutout - background


def stack_psf_cutouts(cutouts: list[np.ndarray]) -> np.ndarray:
    if len(cutouts) == 0:
        raise ValueError("At least one PSF cutout is required.")

    shapes = {cutout.shape for cutout in cutouts}
    if len(shapes) != 1:
        raise ValueError("All PSF cutouts must have the same shape.")

    stack = np.nanmedian(np.stack(cutouts, axis=0), axis=0)

    return normalize_psf(stack)


def centroid_from_flux(cutout: np.ndarray) -> tuple[float, float]:
    data = np.asarray(cutout, dtype=float)
    data = np.where(np.isfinite(data), data, 0.0)
    data = data - np.min(data)

    total = np.sum(data)
    if total <= 0:
        raise ValueError("Cannot compute centroid for non-positive flux cutout.")

    yy, xx = np.indices(data.shape)

    x_centroid = float(np.sum(xx * data) / total)
    y_centroid = float(np.sum(yy * data) / total)

    return x_centroid, y_centroid


def recenter_cutout(
    cutout: np.ndarray,
    *,
    target_x: float | None = None,
    target_y: float | None = None,
) -> np.ndarray:
    from scipy.ndimage import shift

    y_size, x_size = cutout.shape

    if target_x is None:
        target_x = (x_size - 1) / 2

    if target_y is None:
        target_y = (y_size - 1) / 2

    x_centroid, y_centroid = centroid_from_flux(cutout)

    dx = target_x - x_centroid
    dy = target_y - y_centroid

    return shift(
        cutout,
        shift=(dy, dx),
        order=3,
        mode="nearest",
        prefilter=True,
    )


def save_empirical_psf_metadata(
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    make_output_tree(config)
    path = pipeline_paths(config)["psf"]["metadata"]

    with path.open("w") as f:
        json.dump(metadata, f, indent=2)

    return path