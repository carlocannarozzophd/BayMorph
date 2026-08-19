"""Persistence workflow for correlated Gaussian noise calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from baymorph.core.correlated_noise import (
    CorrelatedGaussianNoiseCalibration,
    calibrate_correlated_gaussian_noise,
)
from baymorph.pipeline.config import (
    load_resolved_config,
)
from baymorph.pipeline.image_preparation import (
    load_image_preparation_products,
)
from baymorph.pipeline.outputs import (
    pipeline_paths,
)


DEFAULT_CORRELATED_NOISE_FILENAME = (
    "correlated_gaussian_noise.npz"
)


def correlated_noise_calibration_path(
    config: str | Path | dict[str, Any],
) -> Path:
    """Return the standard persisted correlated-noise calibration path."""
    resolved = _resolve_config(
        config
    )

    preparation_paths = pipeline_paths(
        resolved
    )["image_preparation"]

    return (
        preparation_paths["dir"]
        / DEFAULT_CORRELATED_NOISE_FILENAME
    )


def build_correlated_noise_calibration(
    config: str | Path | dict[str, Any],
    *,
    marginal_tile_size: int,
    marginal_min_coverage: float,
    correlation_tile_size: int,
    correlation_stride: int,
    max_lag: int,
    spectrum_grid_size: int = 512,
    save: bool = True,
    overwrite: bool | None = None,
) -> CorrelatedGaussianNoiseCalibration:
    """Build correlated-noise calibration from persisted preparation products.

    Parameters
    ----------
    config : str, pathlib.Path, or dict
        BayMorph configuration or resolved configuration.
    marginal_tile_size : int
        Tile size used for the smooth marginal-scale calibration.
    marginal_min_coverage : float
        Minimum valid sky fraction for marginal-scale tiles.
    correlation_tile_size : int
        Fully valid tile size used for correlation calibration.
    correlation_stride : int
        Stride between correlation-calibration tiles.
    max_lag : int
        Maximum retained correlation lag.
    spectrum_grid_size : int, default=512
        Fourier verification grid for the finite correlation kernel.
    save : bool, default=True
        Persist the calibration in the image-preparation output directory.
    overwrite : bool, optional
        Override the configured output overwrite policy.

    Returns
    -------
    CorrelatedGaussianNoiseCalibration
        Calibrated fixed noise model.
    """
    resolved = _resolve_config(
        config
    )

    products = (
        load_image_preparation_products(
            resolved
        )
    )

    if products.sky_region is None:
        raise RuntimeError(
            "Correlated-noise calibration requires a persisted sky region."
        )

    calibration = (
        calibrate_correlated_gaussian_noise(
            products.cutout,
            products.sky_region.mask,
            marginal_tile_size=(
                marginal_tile_size
            ),
            marginal_min_coverage=(
                marginal_min_coverage
            ),
            correlation_tile_size=(
                correlation_tile_size
            ),
            correlation_stride=(
                correlation_stride
            ),
            max_lag=max_lag,
            spectrum_grid_size=(
                spectrum_grid_size
            ),
        )
    )

    metadata = dict(
        calibration.metadata
    )

    metadata.update(
        {
            "source_image_shape": list(
                products.cutout.shape
            ),
            "source_sky_pixels": int(
                np.count_nonzero(
                    products.sky_region.mask
                )
            ),
            "source_background": float(
                products.background
            ),
        }
    )

    calibration = (
        CorrelatedGaussianNoiseCalibration(
            marginal_sigma=(
                calibration.marginal_sigma
            ),
            correlation_kernel=(
                calibration.correlation_kernel
            ),
            metadata=metadata,
        )
    )

    if save:
        path = (
            correlated_noise_calibration_path(
                resolved
            )
        )

        if overwrite is None:
            overwrite = bool(
                resolved[
                    "outputs"
                ].get(
                    "overwrite",
                    False,
                )
            )

        save_correlated_noise_calibration(
            path,
            calibration,
            overwrite=bool(
                overwrite
            ),
        )

    return calibration


def save_correlated_noise_calibration(
    path: str | Path,
    calibration: CorrelatedGaussianNoiseCalibration,
    *,
    overwrite: bool = False,
) -> None:
    """Persist one correlated Gaussian noise calibration."""
    output_path = Path(
        path
    )

    if (
        output_path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"Correlated-noise calibration already exists: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_json = json.dumps(
        calibration.metadata,
        sort_keys=True,
    )

    np.savez_compressed(
        output_path,
        marginal_sigma=(
            calibration.marginal_sigma
        ),
        correlation_kernel=(
            calibration.correlation_kernel
        ),
        metadata_json=np.asarray(
            metadata_json
        ),
    )


def load_correlated_noise_calibration(
    path_or_config: (
        str
        | Path
        | dict[str, Any]
    ),
) -> CorrelatedGaussianNoiseCalibration:
    """Load a persisted correlated Gaussian noise calibration."""
    if isinstance(
        path_or_config,
        dict,
    ):
        path = (
            correlated_noise_calibration_path(
                path_or_config
            )
        )
    else:
        candidate = Path(
            path_or_config
        )

        if candidate.suffix == ".npz":
            path = candidate
        else:
            path = (
                correlated_noise_calibration_path(
                    candidate
                )
            )

    if not path.exists():
        raise FileNotFoundError(
            f"Correlated-noise calibration not found: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as payload:
        marginal_sigma = np.asarray(
            payload[
                "marginal_sigma"
            ],
            dtype=float,
        )

        correlation_kernel = np.asarray(
            payload[
                "correlation_kernel"
            ],
            dtype=float,
        )

        metadata_json = str(
            payload[
                "metadata_json"
            ].item()
        )

    metadata = json.loads(
        metadata_json
    )

    return CorrelatedGaussianNoiseCalibration(
        marginal_sigma=marginal_sigma,
        correlation_kernel=correlation_kernel,
        metadata=metadata,
    )


def _resolve_config(
    config: str | Path | dict[str, Any],
) -> dict[str, Any]:
    """Return one resolved BayMorph configuration."""
    if isinstance(
        config,
        dict,
    ):
        return config

    return load_resolved_config(
        config
    )
