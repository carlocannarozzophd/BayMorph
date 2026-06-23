from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from .config import load_resolved_config
from .outputs import make_output_tree, pipeline_paths

from baymorph.core.image import ImageData
from baymorph.simulation.psf import EmpiricalPSF

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
) -> EmpiricalPSF:
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = config

    path = pipeline_paths(resolved)["psf"]["psf_fits"]

    return EmpiricalPSF(
        fits.getdata(path)
    )


def estimate_psf_fwhm(
    psf: np.ndarray,
) -> float | None:
    data = np.asarray(psf, dtype=float)

    if data.ndim != 2:
        return None

    if not np.all(np.isfinite(data)):
        return None

    peak = np.nanmax(data)

    if not np.isfinite(peak) or peak <= 0:
        return None

    half_max = 0.5 * peak

    yy, xx = np.indices(data.shape)

    y0, x0 = np.unravel_index(
        np.nanargmax(data),
        data.shape,
    )

    radius = np.sqrt(
        (xx - x0) ** 2
        + (yy - y0) ** 2
    )

    inside = data >= half_max

    if not np.any(inside):
        return None

    fwhm = 2.0 * np.nanmax(
        radius[inside]
    )

    return float(fwhm)


def psf_radial_profile(
    psf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(psf, dtype=float)

    yy, xx = np.indices(data.shape)

    y0, x0 = np.unravel_index(
        np.nanargmax(data),
        data.shape,
    )

    radius = np.sqrt(
        (xx - x0) ** 2
        + (yy - y0) ** 2
    )

    max_radius = 0.5 * min(data.shape)

    radii = np.arange(
        0.0,
        max_radius + 1.0,
        1.0,
    )

    radius_mid = 0.5 * (
        radii[:-1]
        + radii[1:]
    )

    profile = np.full(
        radius_mid.shape,
        np.nan,
        dtype=float,
    )

    valid = np.isfinite(data)

    for i in range(radii.size - 1):
        annulus = (
            (radius >= radii[i])
            & (radius < radii[i + 1])
            & valid
        )

        if np.any(annulus):
            profile[i] = np.nanmedian(
                data[annulus]
            )

    return radius_mid, profile


def save_empirical_psf_diagnostics(
    config: dict[str, Any],
    psf: np.ndarray,
    cutouts: list[np.ndarray],
) -> None:
    paths = pipeline_paths(config)["psf"]

    diagnostics_dir = paths["dir"] / "diagnostics"

    diagnostics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(30, 15),
    )

    image = axes[0].imshow(
        psf,
        origin="lower",
        cmap="turbo",
    )

    axes[0].set_title(
        "Empirical PSF"
    )

    fig.colorbar(
        image,
        ax=axes[0],
    )

    radius, profile = psf_radial_profile(
        psf,
    )

    axes[1].plot(
        radius,
        profile,
    )

    axes[1].set_xlabel(
        "Radius [pix]"
    )

    axes[1].set_ylabel(
        "Median PSF value"
    )

    axes[1].set_title(
        "PSF radial profile"
    )

    fig.savefig(
        diagnostics_dir / "empirical_psf_diagnostic.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

    if len(cutouts) == 0:
        return

    n_show = min(
        len(cutouts),
        9,
    )

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(45, 45),
    )

    axes = np.ravel(axes)

    for ax in axes:
        ax.axis("off")

    for i in range(n_show):
        ax = axes[i]

        ax.imshow(
            cutouts[i],
            origin="lower",
            cmap="turbo",
        )

        ax.set_title(
            f"PSF cutout {i + 1}"
        )

        ax.axis("on")

    fig.savefig(
        diagnostics_dir / "empirical_psf_cutouts.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


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

    psf_fwhm = estimate_psf_fwhm(
        psf,
    )

    save_empirical_psf_diagnostics(
        resolved,
        psf,
        cutouts,
    )

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
            "psf_peak": float(np.nanmax(psf)),
            "psf_min": float(np.nanmin(psf)),
            "psf_max": float(np.nanmax(psf)),
            "psf_fwhm_pix": psf_fwhm,
            "used_candidates": used_candidates,
            "warnings": _empirical_psf_warnings(
                psf,
                n_candidates_used=len(cutouts),
                psf_fwhm_pix=psf_fwhm,
            ),
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


def clean_empirical_psf(
    psf: np.ndarray,
    *,
    background_radius_fraction: float = 0.65,
    taper_start_fraction: float = 0.55,
) -> np.ndarray:
    data = np.asarray(
        psf,
        dtype=float,
    ).copy()

    if data.ndim != 2:
        raise ValueError(
            "PSF must be 2D."
        )

    if not np.all(
        np.isfinite(data)
    ):
        data = np.nan_to_num(
            data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    y0, x0 = np.unravel_index(
        np.nanargmax(data),
        data.shape,
    )

    yy, xx = np.indices(
        data.shape,
    )

    radius = np.sqrt(
        (xx - x0) ** 2
        + (yy - y0) ** 2
    )

    max_radius = 0.5 * min(
        data.shape
    )

    background_mask = (
        radius >= background_radius_fraction * max_radius
    )

    if np.any(background_mask):
        background = np.nanmedian(
            data[background_mask]
        )
    else:
        background = 0.0

    data = data - background

    data[data < 0.0] = 0.0

    taper_start = (
        taper_start_fraction
        * max_radius
    )

    taper_end = max_radius

    taper = np.ones_like(
        data,
        dtype=float,
    )

    outer = radius > taper_start

    taper[outer] = 0.5 * (
        1.0
        + np.cos(
            np.pi
            * (radius[outer] - taper_start)
            / (taper_end - taper_start)
        )
    )

    taper[radius >= taper_end] = 0.0

    data = data * taper

    return normalize_psf(
        data
    )


def stack_psf_cutouts(cutouts: list[np.ndarray]) -> np.ndarray:
    if len(cutouts) == 0:
        raise ValueError("At least one PSF cutout is required.")

    half_sizes = [
        min(cutout.shape) // 2
        for cutout in cutouts
    ]

    common_half_size = int(
        np.min(half_sizes)
    )

    cropped_cutouts = [
        crop_center(
            cutout,
            common_half_size,
        )
        for cutout in cutouts
    ]

    stack = np.nanmedian(
        np.stack(
            cropped_cutouts,
            axis=0,
        ),
        axis=0,
    )

    psf = normalize_psf(
        stack
    )

    psf = clean_empirical_psf(
        psf,
    )

    return psf

def _empirical_psf_warnings(
    psf: np.ndarray,
    *,
    n_candidates_used: int,
    psf_fwhm_pix: float | None,
) -> list[str]:
    warnings = []

    data = np.asarray(
        psf,
        dtype=float,
    )

    if n_candidates_used < 5:
        warnings.append(
            "Empirical PSF is based on fewer than 5 stars and is likely unstable."
        )

    elif n_candidates_used < 10:
        warnings.append(
            "Empirical PSF is based on fewer than 10 stars; use with caution."
        )

    elif n_candidates_used < 20:
        warnings.append(
            "Empirical PSF is based on fewer than 20 stars; validation is recommended."
        )

    psf_sum = np.nansum(
        data,
    )

    if not np.isfinite(psf_sum) or not np.isclose(psf_sum, 1.0, rtol=1e-3):
        warnings.append(
            "PSF is not normalized to unit sum."
        )

    if not np.all(
        np.isfinite(data)
    ):
        warnings.append(
            "PSF contains non-finite values."
        )

    peak = np.nanmax(
        data,
    )

    if not np.isfinite(peak) or peak <= 0:
        warnings.append(
            "PSF peak is not positive."
        )

    if psf_fwhm_pix is None:
        warnings.append(
            "Could not estimate empirical PSF FWHM."
        )
    elif psf_fwhm_pix <= 0:
        warnings.append(
            "Empirical PSF FWHM is non-positive."
        )

    return warnings


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


def estimate_psf_half_size(
    candidates,
    *,
    fallback_half_size: int = 25,
) -> int:
    half_sizes = [
        candidate.half_size
        for candidate in candidates
        if candidate.half_size is not None
    ]

    if len(half_sizes) == 0:
        return fallback_half_size

    return int(np.max(half_sizes))


def crop_center(
    image: np.ndarray,
    half_size: int,
) -> np.ndarray:
    if half_size <= 0:
        raise ValueError("half_size must be positive.")

    y_size, x_size = image.shape
    x0 = x_size // 2
    y0 = y_size // 2

    y_min = y0 - half_size
    y_max = y0 + half_size + 1
    x_min = x0 - half_size
    x_max = x0 + half_size + 1

    if x_min < 0 or y_min < 0 or x_max > x_size or y_max > y_size:
        raise ValueError("Requested center crop extends outside the image boundaries.")

    return image[y_min:y_max, x_min:x_max]