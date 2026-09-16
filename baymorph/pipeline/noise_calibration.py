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
            "source_fingerprints": (
                correlated_noise_source_fingerprints(
                    products
                )
            ),
            "calibration_settings": {
                "marginal_tile_size": int(
                    marginal_tile_size
                ),
                "marginal_min_coverage": float(
                    marginal_min_coverage
                ),
                "correlation_tile_size": int(
                    correlation_tile_size
                ),
                "correlation_stride": int(
                    correlation_stride
                ),
                "max_lag": int(max_lag),
                "spectrum_grid_size": int(
                    spectrum_grid_size
                ),
            },
            "correlation_taper": (
                "separable_bartlett"
            ),
            "generator_identity": (
                _correlated_noise_calibration_generator_identity()
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

DEFAULT_CORRELATED_NOISE_SETTINGS = {
    "marginal_tile_size": 32,
    "marginal_min_coverage": 0.20,
    "correlation_tile_size": 48,
    "correlation_stride": 8,
    "max_lag": 20,
    "spectrum_grid_size": 512,
}


def array_sha256(array: np.ndarray) -> str:
    """Return a stable SHA-256 fingerprint including dtype and shape."""
    import hashlib

    values = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.view(np.uint8).tobytes())
    return digest.hexdigest()


def correlated_noise_source_fingerprints(products) -> dict[str, str]:
    """Fingerprint every prepared product that fixes the noise model."""
    if products.sky_region is None:
        raise RuntimeError(
            "Correlated-noise calibration requires a persisted sky region."
        )

    return {
        "cutout_sha256": array_sha256(products.cutout),
        "prepared_data_sha256": array_sha256(products.prepared_data),
        "sky_mask_sha256": array_sha256(
            np.asarray(products.sky_region.mask, dtype=bool)
        ),
        "fitting_mask_sha256": array_sha256(
            np.asarray(products.fitting_mask, dtype=bool)
        ),
    }


def correlated_noise_settings_from_config(
    config: dict[str, Any],
) -> dict[str, int | float]:
    """Resolve fixed correlated-noise calibration settings from config."""
    likelihood = config.get("fit", {}).get("likelihood", {})

    if not isinstance(likelihood, dict):
        raise TypeError("fit.likelihood must be a mapping.")

    calibration = likelihood.get("calibration", {})
    if calibration is None:
        calibration = {}

    if not isinstance(calibration, dict):
        raise TypeError(
            "fit.likelihood.calibration must be a mapping."
        )

    values = {
        key: calibration.get(key, default)
        for key, default in DEFAULT_CORRELATED_NOISE_SETTINGS.items()
    }

    settings: dict[str, int | float] = {
        "marginal_tile_size": int(values["marginal_tile_size"]),
        "marginal_min_coverage": float(
            values["marginal_min_coverage"]
        ),
        "correlation_tile_size": int(
            values["correlation_tile_size"]
        ),
        "correlation_stride": int(values["correlation_stride"]),
        "max_lag": int(values["max_lag"]),
        "spectrum_grid_size": int(values["spectrum_grid_size"]),
    }

    max_lag = int(settings["max_lag"])

    if int(settings["marginal_tile_size"]) < 2:
        raise ValueError(
            "marginal_tile_size must be at least two."
        )

    if not 0.0 < float(
        settings["marginal_min_coverage"]
    ) <= 1.0:
        raise ValueError(
            "marginal_min_coverage must lie in (0, 1]."
        )

    if max_lag < 1:
        raise ValueError("max_lag must be at least one.")

    if int(settings["correlation_tile_size"]) <= 2 * max_lag:
        raise ValueError(
            "correlation_tile_size must exceed twice max_lag."
        )

    if int(settings["correlation_stride"]) < 1:
        raise ValueError(
            "correlation_stride must be at least one."
        )

    return settings


def validate_correlated_noise_calibration_sources(
    calibration: CorrelatedGaussianNoiseCalibration,
    products,
    *,
    settings: dict[str, int | float] | None = None,
) -> None:
    """Reject calibration metadata stale for current prepared inputs."""
    expected_sources = correlated_noise_source_fingerprints(
        products
    )

    if calibration.metadata.get(
        "source_fingerprints"
    ) != expected_sources:
        raise ValueError(
            "Correlated-noise calibration does not match the current "
            "cutout, prepared image, sky mask, and fitting mask."
        )

    expected_generator = (
        _correlated_noise_calibration_generator_identity()
    )
    if calibration.metadata.get(
        "generator_identity"
    ) != expected_generator:
        raise ValueError(
            "Correlated-noise calibration generator identity does not "
            "match the current implementation and numerical runtime."
        )

    if settings is not None:
        recorded = calibration.metadata.get(
            "calibration_settings"
        )

        if recorded != dict(settings):
            raise ValueError(
                "Correlated-noise calibration settings do not match "
                "the current resolved configuration."
            )

def _correlated_noise_calibration_generator_identity(
) -> dict[str, Any]:
    """Return the calibration implementation and numerical identity."""
    import hashlib
    import importlib.metadata

    import scipy

    import baymorph.core.correlated_noise as core_module

    core_source = getattr(
        core_module,
        "__file__",
        None,
    )
    if core_source is None:
        raise RuntimeError(
            "Cannot locate the correlated-noise implementation."
        )

    try:
        version = importlib.metadata.version(
            "baymorph"
        )
    except importlib.metadata.PackageNotFoundError:
        version = None

    def file_sha256(
        file_path: str | Path,
    ) -> str:
        return hashlib.sha256(
            Path(file_path).read_bytes()
        ).hexdigest()

    return {
        "algorithm": (
            "empirical_stationary_srs_v1"
        ),
        "pipeline_source_sha256": file_sha256(
            __file__
        ),
        "core_source_sha256": file_sha256(
            core_source
        ),
        "baymorph_version": version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }
