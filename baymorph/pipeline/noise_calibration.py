"""Persistence workflow for the sky correlation of the noise model (AD-047)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from baymorph.core.correlated_noise import (
    CALIBRATION_SCHEMA_VERSION,
    FIELD_SEPARATED_GROWTH_PIXELS,
    SKY_CORRELATION_ESTIMATORS,
    CorrelatedGaussianNoiseCalibration,
    calibrate_correlated_gaussian_noise,
)
from baymorph.pipeline.config import (
    DEFAULT_CONFIG,
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
    save: bool = True,
    overwrite: bool | None = None,
) -> CorrelatedGaussianNoiseCalibration:
    """Estimate the sky correlation from persisted preparation products.

    The estimators and their constants are fixed by AD-047 and AD-051; the
    configuration only declares which one applies, with
    ``preparation.noise.sky_correlation``.

    Parameters
    ----------
    config : str, pathlib.Path, or dict
        BayMorph configuration or resolved configuration.
    save : bool, default=True
        Persist the calibration in the image-preparation output directory.
    overwrite : bool, optional
        Override the configured output overwrite policy.

    Returns
    -------
    CorrelatedGaussianNoiseCalibration
        The fixed sky correlation of the observation.
    """
    resolved = _resolve_config(
        config
    )
    reject_correlated_noise_calibration_settings(
        resolved
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

    estimator = configured_sky_correlation(
        resolved
    )

    if (
        estimator == "field_separated"
        and products.contaminant_growth_pixels != FIELD_SEPARATED_GROWTH_PIXELS
    ):
        raise ValueError(
            "The field-separated sky correlation was validated on sky regions "
            "whose contaminant masks are grown by "
            f"{FIELD_SEPARATED_GROWTH_PIXELS} px; this preparation grew them "
            f"by {products.contaminant_growth_pixels} px."
        )

    calibration = calibrate_correlated_gaussian_noise(
        products.cutout,
        products.sky_region.mask,
        estimator=estimator,
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
            "generator_identity": (
                _correlated_noise_calibration_generator_identity()
            ),
        }
    )

    calibration = CorrelatedGaussianNoiseCalibration(
        correlation_kernel=calibration.correlation_kernel,
        metadata=metadata,
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
    """Load a persisted calibration of the current schema.

    A calibration of an earlier schema is rejected, never converted or
    overwritten: schema 1 held the tapered tile estimator and the smooth
    marginal field that AD-047 retired, and schema 2 did not record which
    estimator made the kernel.
    """
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
        names = set(payload.files)
        metadata = json.loads(
            str(
                payload[
                    "metadata_json"
                ].item()
            )
        )
        schema_version = metadata.get(
            "schema_version"
        )

        if (
            schema_version != CALIBRATION_SCHEMA_VERSION
            or names != {"correlation_kernel", "metadata_json"}
        ):
            raise ValueError(
                f"Correlated-noise calibration {path} has schema "
                f"{schema_version!r}; BayMorph accepts only schema "
                f"{CALIBRATION_SCHEMA_VERSION}, whose sky correlation is "
                "estimated on the whole sky region without a taper by a "
                "declared estimator. Build a new calibration in a new output "
                "namespace."
            )

        if metadata.get("estimator") not in SKY_CORRELATION_ESTIMATORS.values():
            raise ValueError(
                f"Correlated-noise calibration {path} records the unknown "
                f"estimator {metadata.get('estimator')!r}."
            )

        correlation_kernel = np.asarray(
            payload[
                "correlation_kernel"
            ],
            dtype=float,
        )

    return CorrelatedGaussianNoiseCalibration(
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


def reject_correlated_noise_calibration_settings(
    config: dict[str, Any],
) -> None:
    """Reject estimator settings, which AD-047 fixes, in the configuration."""
    likelihood = config.get("fit", {}).get("likelihood", {})

    if isinstance(likelihood, dict) and "calibration" in likelihood:
        raise ValueError(
            "fit.likelihood.calibration is no longer accepted: AD-047 fixes "
            "the sky-correlation estimators (the whole sky region, one plane "
            "removed, no taper, lags up to 8), and "
            "preparation.noise.sky_correlation declares which one applies. "
            "Remove the block."
        )


def configured_sky_correlation(
    config: dict[str, Any],
) -> str:
    """Return the sky-correlation estimator that the configuration declares.

    A configuration passed as a mapping without the entry takes the default
    that :func:`baymorph.pipeline.config.resolve_config` would give it.
    """
    noise = config.get("preparation", {}).get("noise") or {}
    estimator = noise.get(
        "sky_correlation",
        DEFAULT_CONFIG["preparation"]["noise"]["sky_correlation"],
    )

    if estimator not in SKY_CORRELATION_ESTIMATORS:
        raise ValueError(
            "preparation.noise.sky_correlation must be one of "
            f"{list(SKY_CORRELATION_ESTIMATORS)}, not {estimator!r}."
        )

    return estimator


def validate_correlated_noise_calibration_estimator(
    calibration: CorrelatedGaussianNoiseCalibration,
    config: dict[str, Any],
) -> None:
    """Reject a calibration made by an estimator other than the declared one."""
    estimator = configured_sky_correlation(
        config
    )
    expected = SKY_CORRELATION_ESTIMATORS[estimator]

    if calibration.metadata.get("estimator") != expected:
        raise ValueError(
            "Correlated-noise calibration was made by the estimator "
            f"{calibration.metadata.get('estimator')!r}, but the "
            f"configuration declares preparation.noise.sky_correlation: "
            f"{estimator}. Build the calibration in a new output namespace."
        )


def validate_correlated_noise_calibration_sources(
    calibration: CorrelatedGaussianNoiseCalibration,
    products,
) -> None:
    """Reject calibration metadata stale for current prepared inputs."""
    if calibration.metadata.get(
        "schema_version"
    ) != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            "Correlated-noise calibration does not have the current schema "
            f"{CALIBRATION_SCHEMA_VERSION}."
        )

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
        "estimators": sorted(SKY_CORRELATION_ESTIMATORS.values()),
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
