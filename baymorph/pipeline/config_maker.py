"""Configuration generation from astronomical FITS observations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from baymorph.core.observation import Observation
from baymorph.pipeline.config import resolve_config


TargetPosition = (
    str
    | Mapping[str, float]
    | Sequence[float]
)


def _finite_positive(
    value: Any,
) -> float | None:
    """Return a finite positive float or None."""
    if value is None:
        return None

    try:
        result = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if (
        not math.isfinite(
            result
        )
        or result <= 0.0
    ):
        return None

    return result


def _normalize_target_position(
    target_position: TargetPosition | None,
) -> str | dict[str, float]:
    """Normalize an explicit target-identification seed."""
    if target_position is None:
        raise ValueError(
            "target_position is required. Supply (x, y), a mapping with "
            "'x' and 'y', or the explicit value 'center'."
        )

    if isinstance(
        target_position,
        str,
    ):
        value = target_position.strip().lower()

        if value != "center":
            raise ValueError(
                "String target_position must be exactly 'center'."
            )

        return "center"

    if isinstance(
        target_position,
        Mapping,
    ):
        if set(
            target_position
        ) != {
            "x",
            "y",
        }:
            raise ValueError(
                "Target-position mappings must contain exactly 'x' and 'y'."
            )

        x = float(
            target_position[
                "x"
            ]
        )
        y = float(
            target_position[
                "y"
            ]
        )

    else:
        if (
            isinstance(
                target_position,
                (str, bytes),
            )
            or len(
                target_position
            )
            != 2
        ):
            raise ValueError(
                "Target-position sequences must contain exactly two values."
            )

        x = float(
            target_position[
                0
            ]
        )
        y = float(
            target_position[
                1
            ]
        )

    if not (
        math.isfinite(
            x
        )
        and math.isfinite(
            y
        )
    ):
        raise ValueError(
            "Target-position coordinates must be finite."
        )

    return {
        "x": x,
        "y": y,
    }


def _cutout_config(
    mode: str,
    *,
    fixed_size: int | None,
) -> dict[str, Any]:
    """Build a standard cutout configuration."""
    mode = str(
        mode
    ).strip().lower()

    if mode == "auto":
        if fixed_size is not None:
            raise ValueError(
                "fixed_size cannot be supplied when cutout_mode='auto'."
            )

        return {
            "mode": "auto",
            "initial_factor": 4.0,
            "sky_margin": 20.0,
            "min_size": 100,
            "max_size": None,
        }

    if mode == "fixed":
        if fixed_size is None:
            raise ValueError(
                "fixed_size is required when cutout_mode='fixed'."
            )

        fixed_size = int(
            fixed_size
        )

        if fixed_size <= 0:
            raise ValueError(
                "fixed_size must be positive."
            )

        return {
            "mode": "fixed",
            "size": fixed_size,
        }

    raise ValueError(
        "cutout_mode must be 'auto' or 'fixed'."
    )


def _empirical_psf_config(
    observation: Observation,
    *,
    pixel_scale: float | None,
) -> dict[str, Any]:
    """Build empirical-PSF configuration from deterministic metadata."""
    candidate_selection: dict[
        str,
        Any,
    ] = {
        "auto": True,
    }

    seeing = _finite_positive(
        observation.metadata.seeing_fwhm
    )

    if (
        seeing is not None
        and pixel_scale is not None
    ):
        candidate_selection[
            "min_half_size"
        ] = "auto"

        candidate_selection[
            "max_half_size"
        ] = "auto"

    saturation = _finite_positive(
        observation.metadata.saturation
    )

    if saturation is not None:
        candidate_selection[
            "max_peak"
        ] = "auto"

    return {
        "type": "empirical",
        "empirical": {
            "candidate_selection": (
                candidate_selection
            ),
        },
    }


def _gaussian_psf_config(
    observation: Observation,
    *,
    pixel_scale: float | None,
    gaussian_fwhm_pixels: float | None,
) -> dict[str, Any]:
    """Build an explicitly requested Gaussian-PSF configuration."""
    fwhm_pixels = _finite_positive(
        gaussian_fwhm_pixels
    )

    if fwhm_pixels is None:
        seeing = _finite_positive(
            observation.metadata.seeing_fwhm
        )

        if (
            seeing is None
            or pixel_scale is None
        ):
            raise ValueError(
                "Gaussian PSF generation requires gaussian_fwhm_pixels or "
                "both seeing and pixel-scale metadata in the FITS file."
            )

        fwhm_pixels = (
            seeing
            / pixel_scale
        )

    return {
        "type": "gaussian",
        "gaussian": {
            "fwhm": float(
                fwhm_pixels
            ),
        },
    }


def make_config_from_fits(
    fits_path: str | Path,
    *,
    target_position: TargetPosition | None = None,
    output_path: str | Path | None = None,
    object_name: str | None = None,
    outputs_root: str | Path = "runs",
    overwrite: bool = True,
    cutout_mode: str = "auto",
    fixed_size: int | None = None,
    psf_type: str = "empirical",
    gaussian_fwhm_pixels: float | None = None,
) -> dict[str, Any]:
    """Create a runnable BayMorph configuration from one FITS image.

    Parameters
    ----------
    fits_path : str or pathlib.Path
        Input two-dimensional astronomical FITS image.
    target_position : mapping, sequence, or {"center"}
        Explicit target-identification seed. The value ``"center"`` requests
        the detector centre and does not imply automatic target discovery.
    output_path : str or pathlib.Path, optional
        YAML path to write. If omitted, no file is written.
    object_name : str, optional
        Analysis object name. The FITS filename stem is used by default.
    outputs_root : str or pathlib.Path, default="runs"
        BayMorph output root.
    overwrite : bool, default=True
        Whether BayMorph products may overwrite existing run products.
    cutout_mode : {"auto", "fixed"}, default="auto"
        Image-cutout strategy.
    fixed_size : int, optional
        Fixed cutout size. Required only for ``cutout_mode="fixed"``.
    psf_type : {"empirical", "gaussian"}, default="empirical"
        PSF strategy.
    gaussian_fwhm_pixels : float, optional
        Gaussian PSF FWHM in detector pixels. When Gaussian mode is requested
        and this value is omitted, BayMorph converts FITS seeing metadata to
        pixels when both seeing and pixel scale are available.

    Returns
    -------
    dict
        Minimal user-facing BayMorph configuration.

    Notes
    -----
    This function does not estimate morphology parameters or construct priors
    from image data. Model priors remain controlled by the normal BayMorph
    configuration layer.

    The target position is intentionally explicit because it identifies which
    detected source BayMorph should analyse. Using ``"center"`` is therefore
    an explicit assumption by the caller, not target detection.
    """
    fits_path = Path(
        fits_path
    )

    if not fits_path.exists():
        raise FileNotFoundError(
            f"FITS file not found: {fits_path}"
        )

    observation = Observation.from_fits(
        fits_path
    )

    resolved_name = (
        fits_path.stem
        if object_name is None
        else str(
            object_name
        ).strip()
    )

    if not resolved_name:
        raise ValueError(
            "object_name must be non-empty."
        )

    pixel_scale = _finite_positive(
        observation.metadata.pixel_scale
    )

    normalized_target = (
        _normalize_target_position(
            target_position
        )
    )

    cutout = _cutout_config(
        cutout_mode,
        fixed_size=fixed_size,
    )

    normalized_psf_type = str(
        psf_type
    ).strip().lower()

    if normalized_psf_type == "empirical":
        psf = _empirical_psf_config(
            observation,
            pixel_scale=pixel_scale,
        )

    elif normalized_psf_type == "gaussian":
        psf = _gaussian_psf_config(
            observation,
            pixel_scale=pixel_scale,
            gaussian_fwhm_pixels=(
                gaussian_fwhm_pixels
            ),
        )

    else:
        raise ValueError(
            "psf_type must be 'empirical' or 'gaussian'."
        )

    config: dict[
        str,
        Any,
    ] = {
        "object": {
            "name": resolved_name,
        },
        "data": {
            "image_path": str(
                fits_path
            ),
            "pixel_scale": pixel_scale,
            "target_position": (
                normalized_target
            ),
            "cutout": cutout,
        },
        "outputs": {
            "root": str(
                Path(
                    outputs_root
                )
            ),
            "overwrite": bool(
                overwrite
            ),
        },
        "psf": psf,
    }

    # Validate compatibility with the established single-observation
    # configuration contract without expanding the YAML with all defaults.
    resolve_config(
        config
    )

    if output_path is not None:
        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as stream:
            yaml.safe_dump(
                config,
                stream,
                sort_keys=False,
            )

    return config
