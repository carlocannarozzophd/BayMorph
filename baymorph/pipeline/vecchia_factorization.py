"""Immutable persistence for the production Vecchia noise factorization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import errno
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
from typing import Any

import numpy as np
import scipy

import baymorph.likelihood.vecchia_gaussian as vecchia_module
from baymorph.likelihood.vecchia_gaussian import (
    VECCHIA_ALGORITHM as CORE_VECCHIA_ALGORITHM,
    VECCHIA_DISTANCE_METRIC as CORE_VECCHIA_DISTANCE_METRIC,
    VECCHIA_FACTORIZATION_SCHEMA,
    VECCHIA_FACTORIZATION_VERSION,
    VECCHIA_NEIGHBOR_POLICY as CORE_VECCHIA_NEIGHBOR_POLICY,
    VECCHIA_ORDERING as CORE_VECCHIA_ORDERING,
    VECCHIA_TIE_BREAK as CORE_VECCHIA_TIE_BREAK,
    VecchiaFactorization,
    build_vecchia_factorization,
)
from baymorph.core.correlated_noise import (
    CORRELATION_MAX_LAG,
    SKY_CORRELATION_ESTIMATORS,
)
from baymorph.pipeline.config import load_resolved_config
from baymorph.pipeline.image_preparation import load_image_preparation_products
from baymorph.pipeline.noise_calibration import (
    correlated_noise_calibration_path,
    load_correlated_noise_calibration,
    reject_correlated_noise_calibration_settings,
    validate_correlated_noise_calibration_estimator,
    validate_correlated_noise_calibration_sources,
)


DEFAULT_VECCHIA_FACTORIZATION_DIRECTORY = "correlated_gaussian_vecchia"
VECCHIA_MANIFEST_FILENAME = "manifest.json"
VECCHIA_COMPLETE_FILENAME = "COMPLETE.json"
VECCHIA_SCHEMA = VECCHIA_FACTORIZATION_SCHEMA
# Version of the artifact, which binds the factorization to its sources; the
# factorization object keeps VECCHIA_FACTORIZATION_VERSION. Version 2 binds
# the sky correlation of AD-047 and no marginal scale.
VECCHIA_SCHEMA_VERSION = 2
VECCHIA_ALGORITHM = CORE_VECCHIA_ALGORITHM
VECCHIA_ORDERING = CORE_VECCHIA_ORDERING
VECCHIA_NEIGHBOR_POLICY = CORE_VECCHIA_NEIGHBOR_POLICY
VECCHIA_DISTANCE_METRIC = CORE_VECCHIA_DISTANCE_METRIC
VECCHIA_TIE_BREAK = CORE_VECCHIA_TIE_BREAK
# AD-047 fixes 24 neighbours for fits; the higher orders may be built only to
# check the approximation on real data, outside any fit.
PRODUCTION_VECCHIA_NEIGHBORS = 24
VALIDATION_VECCHIA_NEIGHBORS = 96
SUPPORTED_VECCHIA_NEIGHBOR_ORDERS = (
    PRODUCTION_VECCHIA_NEIGHBORS,
    64,
    VALIDATION_VECCHIA_NEIGHBORS,
)
DEFAULT_MIN_CONDITIONAL_VARIANCE = 1.0e-12
VECCHIA_PUBLICATION_WAIT_SECONDS = 5.0
VECCHIA_PUBLICATION_POLL_SECONDS = 0.05

_ARRAY_FILES = {
    "valid_flat_indices": "valid_flat_indices.npy",
    "neighbor_indptr": "neighbor_indptr.npy",
    "neighbor_indices": "neighbor_indices.npy",
    "weights": "weights.npy",
    "conditional_variances": "conditional_variances.npy",
}

_ARRAY_DTYPES = {
    "valid_flat_indices": np.dtype(np.int64),
    "neighbor_indptr": np.dtype(np.int64),
    "neighbor_indices": np.dtype(np.int32),
    "weights": np.dtype(np.float64),
    "conditional_variances": np.dtype(np.float64),
}

_SOURCE_BINDING_KEYS = {
    "cutout_sha256",
    "prepared_image_sha256",
    "sky_mask_sha256",
    "fitting_mask_sha256",
    "correlation_kernel_sha256",
    "correlated_noise_calibration_sha256",
    "correlation_estimator",
    "calibration_generator_identity",
}


@dataclass(frozen=True)
class _SourceSnapshot:
    """Owned immutable inputs used to build one factorization."""

    fitting_mask: np.ndarray
    correlation_kernel: np.ndarray
    binding: dict[str, Any]


def vecchia_factorization_path(
    config: str | Path | dict[str, Any],
    *,
    max_neighbors: int = PRODUCTION_VECCHIA_NEIGHBORS,
) -> Path:
    """Return the immutable Vecchia artifact-directory path."""
    max_neighbors, _ = _validate_settings(
        max_neighbors,
        DEFAULT_MIN_CONDITIONAL_VARIANCE,
    )
    resolved = _resolve_config(config)
    return (
        correlated_noise_calibration_path(resolved).parent
        / DEFAULT_VECCHIA_FACTORIZATION_DIRECTORY
        / f"m{max_neighbors}"
    )


def vecchia_factorization_manifest_path(
    config: str | Path | dict[str, Any],
    *,
    max_neighbors: int = PRODUCTION_VECCHIA_NEIGHBORS,
) -> Path:
    """Return the Vecchia artifact manifest path."""
    return (
        vecchia_factorization_path(
            config,
            max_neighbors=max_neighbors,
        )
        / VECCHIA_MANIFEST_FILENAME
    )


def build_vecchia_factorization_artifact(
    config: str | Path | dict[str, Any],
    *,
    max_neighbors: int = PRODUCTION_VECCHIA_NEIGHBORS,
    min_conditional_variance: float = DEFAULT_MIN_CONDITIONAL_VARIANCE,
) -> VecchiaFactorization:
    """Build and atomically publish one immutable Vecchia factorization."""
    max_neighbors, min_conditional_variance = _validate_settings(
        max_neighbors,
        min_conditional_variance,
    )
    resolved = _resolve_config(config)
    snapshot = _capture_source_snapshot(resolved)

    factorization = build_vecchia_factorization(
        snapshot.fitting_mask,
        snapshot.correlation_kernel,
        max_neighbors=max_neighbors,
        min_conditional_variance=min_conditional_variance,
    )
    if factorization.valid_mask_sha256 != snapshot.binding[
        "fitting_mask_sha256"
    ]:
        raise RuntimeError("Vecchia builder returned a stale fitting-mask hash.")
    if factorization.correlation_kernel_sha256 != snapshot.binding[
        "correlation_kernel_sha256"
    ]:
        raise RuntimeError(
            "Vecchia builder returned a stale correlation-kernel hash."
        )
    _save_vecchia_bundle(
        vecchia_factorization_path(
            resolved,
            max_neighbors=max_neighbors,
        ),
        factorization,
        binding=snapshot.binding,
        min_conditional_variance=min_conditional_variance,
    )
    del factorization
    return load_vecchia_factorization_artifact(
        resolved,
        max_neighbors=max_neighbors,
        min_conditional_variance=min_conditional_variance,
    )


def ensure_vecchia_factorization_artifact(
    config: str | Path | dict[str, Any],
    *,
    max_neighbors: int = PRODUCTION_VECCHIA_NEIGHBORS,
    min_conditional_variance: float = DEFAULT_MIN_CONDITIONAL_VARIANCE,
) -> VecchiaFactorization:
    """Build a missing Vecchia artifact or validate the immutable existing one."""
    max_neighbors, min_conditional_variance = _validate_settings(
        max_neighbors,
        min_conditional_variance,
    )
    resolved = _resolve_config(config)
    path = vecchia_factorization_path(
        resolved,
        max_neighbors=max_neighbors,
    )

    if path.exists():
        _wait_for_completed_bundle(path)
        return load_vecchia_factorization_artifact(
            resolved,
            max_neighbors=max_neighbors,
            min_conditional_variance=min_conditional_variance,
        )

    try:
        return build_vecchia_factorization_artifact(
            resolved,
            max_neighbors=max_neighbors,
            min_conditional_variance=min_conditional_variance,
        )
    except FileExistsError:
        _wait_for_completed_bundle(path)
        return load_vecchia_factorization_artifact(
            resolved,
            max_neighbors=max_neighbors,
            min_conditional_variance=min_conditional_variance,
        )


def load_vecchia_factorization_manifest(
    path_or_config: str | Path | dict[str, Any],
    *,
    max_neighbors: int = PRODUCTION_VECCHIA_NEIGHBORS,
) -> dict[str, Any]:
    """Load and authenticate the completion marker and artifact manifest."""
    max_neighbors, _ = _validate_settings(
        max_neighbors,
        DEFAULT_MIN_CONDITIONAL_VARIANCE,
    )
    path, _ = _artifact_path_and_config(
        path_or_config,
        max_neighbors=max_neighbors,
        require_config=False,
    )
    directory_fd = _open_artifact_directory(path)
    try:
        manifest = _load_authenticated_manifest(directory_fd)
    finally:
        os.close(directory_fd)
    if manifest["max_neighbors"] != max_neighbors:
        raise ValueError(
            "Vecchia manifest does not match its neighbour-order namespace."
        )
    return manifest


def load_vecchia_factorization_artifact(
    path_or_config: str | Path | dict[str, Any],
    *,
    max_neighbors: int = PRODUCTION_VECCHIA_NEIGHBORS,
    min_conditional_variance: float = DEFAULT_MIN_CONDITIONAL_VARIANCE,
) -> VecchiaFactorization:
    """Load an authenticated factorization and reject stale source bindings."""
    max_neighbors, min_conditional_variance = _validate_settings(
        max_neighbors,
        min_conditional_variance,
    )
    path, resolved = _artifact_path_and_config(
        path_or_config,
        max_neighbors=max_neighbors,
        require_config=True,
    )
    if resolved is None:
        raise RuntimeError("Vecchia inference loading requires a resolved config.")
    directory_fd = _open_artifact_directory(path)
    try:
        manifest = _load_authenticated_manifest(directory_fd)
        arrays = _load_arrays(directory_fd, manifest)
    finally:
        os.close(directory_fd)
    factorization = _make_factorization(manifest, arrays)

    _validate_loaded_factorization(
        factorization,
        manifest,
        max_neighbors=max_neighbors,
        min_conditional_variance=min_conditional_variance,
    )

    expected_binding = _capture_source_snapshot(resolved).binding
    if manifest.get("source_binding") != expected_binding:
        raise ValueError(
            "Vecchia factorization does not match the current prepared "
            "image, fitting mask, noise field, correlation kernel, or "
            "calibration bytes. Build it in a new immutable run directory."
        )

    return factorization


def _save_vecchia_bundle(
    path: Path,
    factorization: VecchiaFactorization,
    *,
    binding: Mapping[str, Any],
    min_conditional_variance: float,
) -> dict[str, Any]:
    """Publish a complete bundle without ever replacing an existing path."""
    arrays = _factorization_arrays(factorization)
    _validate_array_contract(arrays, factorization)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent)
    )

    try:
        array_specs: dict[str, dict[str, Any]] = {}
        for name, filename in _ARRAY_FILES.items():
            array_path = temporary / filename
            _write_npy(array_path, arrays[name])
            array_specs[name] = _array_spec(arrays[name], array_path, filename)

        manifest = _build_manifest(
            factorization,
            binding=binding,
            array_specs=array_specs,
            min_conditional_variance=min_conditional_variance,
        )
        manifest["payload_sha256"] = _manifest_payload_sha256(manifest)
        manifest_path = temporary / VECCHIA_MANIFEST_FILENAME
        _write_json(manifest_path, manifest)
        completion = {
            "schema": VECCHIA_SCHEMA,
            "schema_version": VECCHIA_SCHEMA_VERSION,
            "manifest": VECCHIA_MANIFEST_FILENAME,
            "manifest_sha256": _file_sha256(manifest_path),
        }
        _write_json(temporary / VECCHIA_COMPLETE_FILENAME, completion)
        for filename in (
            *_ARRAY_FILES.values(),
            VECCHIA_MANIFEST_FILENAME,
            VECCHIA_COMPLETE_FILENAME,
        ):
            (temporary / filename).chmod(0o444)
        _fsync_directory(temporary)

        try:
            path.mkdir()
        except FileExistsError as exc:
            raise FileExistsError(
                f"Vecchia factorization already exists: {path}"
            ) from exc

        publication_order = [
            *_ARRAY_FILES.values(),
            VECCHIA_MANIFEST_FILENAME,
            VECCHIA_COMPLETE_FILENAME,
        ]
        for filename in publication_order:
            _publish_file_exclusive(
                temporary / filename,
                path / filename,
            )
            if filename == VECCHIA_MANIFEST_FILENAME:
                _fsync_directory(path)

        # COMPLETE is linked last. Readers therefore see either an invalid
        # incomplete bundle or a fully authenticated one, never a completed
        # partial bundle.
        _fsync_directory(path)
        _fsync_directory(path.parent)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _build_manifest(
    factorization: VecchiaFactorization,
    *,
    binding: Mapping[str, Any],
    array_specs: Mapping[str, Any],
    min_conditional_variance: float,
) -> dict[str, Any]:
    counts = np.diff(np.asarray(factorization.neighbor_indptr))
    variances = np.asarray(factorization.conditional_variances)
    arrays = _factorization_arrays(factorization)
    resident_bytes = int(sum(value.nbytes for value in arrays.values()))
    logdet = float(np.sum(np.log(variances), dtype=np.float64))

    return {
        "schema": VECCHIA_SCHEMA,
        "schema_version": VECCHIA_SCHEMA_VERSION,
        "algorithm": VECCHIA_ALGORITHM,
        "ordering": VECCHIA_ORDERING,
        "pixel_order": "C",
        "neighbor_policy": VECCHIA_NEIGHBOR_POLICY,
        "distance_metric": VECCHIA_DISTANCE_METRIC,
        "tie_break": VECCHIA_TIE_BREAK,
        "image_shape": [int(value) for value in factorization.image_shape],
        "n_valid_pixels": int(variances.size),
        "neighbor_order": f"m{int(factorization.max_neighbors)}",
        "max_neighbors": int(factorization.max_neighbors),
        "n_unique_patterns": int(factorization.n_unique_patterns),
        "min_conditional_variance": float(min_conditional_variance),
        "correlation_logdet": logdet,
        "source_binding": _json_value(binding),
        "arrays": _json_value(array_specs),
        "generator_identity": _generator_identity(),
        "diagnostics": {
            "minimum_conditional_variance": float(np.min(variances)),
            "maximum_conditional_variance": float(np.max(variances)),
            "minimum_neighbor_count": int(np.min(counts)),
            "maximum_neighbor_count": int(np.max(counts)),
            "neighbor_edge_count": int(np.sum(counts, dtype=np.int64)),
            "resident_array_bytes": resident_bytes,
        },
    }


def _validate_manifest_structure(manifest: Mapping[str, Any]) -> None:
    identity = {
        "schema": VECCHIA_SCHEMA,
        "schema_version": VECCHIA_SCHEMA_VERSION,
        "algorithm": VECCHIA_ALGORITHM,
        "ordering": VECCHIA_ORDERING,
        "pixel_order": "C",
        "neighbor_policy": VECCHIA_NEIGHBOR_POLICY,
        "distance_metric": VECCHIA_DISTANCE_METRIC,
        "tie_break": VECCHIA_TIE_BREAK,
        "generator_identity": _generator_identity(),
    }
    for key, expected in identity.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Vecchia manifest has stale or invalid {key}.")

    array_specs = manifest.get("arrays")
    if not isinstance(array_specs, Mapping) or set(array_specs) != set(_ARRAY_FILES):
        raise ValueError("Vecchia manifest array inventory is invalid.")
    binding = manifest.get("source_binding")
    if not isinstance(binding, Mapping) or set(binding) != _SOURCE_BINDING_KEYS:
        raise ValueError("Vecchia manifest source binding is missing.")

    for key in _SOURCE_BINDING_KEYS:
        if key.endswith("_sha256") and not _is_sha256(binding[key]):
            raise ValueError(f"Vecchia source binding {key} is invalid.")
    if not isinstance(binding["calibration_generator_identity"], Mapping):
        raise ValueError(
            "Vecchia source binding calibration_generator_identity is invalid."
        )
    if binding["correlation_estimator"] not in SKY_CORRELATION_ESTIMATORS.values():
        raise ValueError("Vecchia source binding correlation_estimator is invalid.")

    shape = manifest.get("image_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
        or any(value < 1 for value in shape)
    ):
        raise ValueError("Vecchia manifest image_shape is invalid.")

    for key in (
        "n_valid_pixels",
        "max_neighbors",
        "n_unique_patterns",
    ):
        value = manifest.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise ValueError(f"Vecchia manifest {key} is invalid.")

    if manifest.get("neighbor_order") != f"m{manifest['max_neighbors']}":
        raise ValueError("Vecchia manifest neighbor_order is invalid.")
    if manifest["max_neighbors"] not in SUPPORTED_VECCHIA_NEIGHBOR_ORDERS:
        raise ValueError("Vecchia manifest max_neighbors is unsupported.")

    for key in (
        "min_conditional_variance",
        "correlation_logdet",
    ):
        value = manifest.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Vecchia manifest {key} is invalid.")
        if not np.isfinite(float(value)):
            raise ValueError(f"Vecchia manifest {key} is not finite.")


def _load_arrays(
    directory_fd: int,
    manifest: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}

    for name, expected_filename in _ARRAY_FILES.items():
        spec = manifest["arrays"][name]
        if (
            not isinstance(spec, Mapping)
            or set(spec)
            != {"filename", "dtype", "shape", "nbytes", "array_sha256", "file_sha256"}
            or spec.get("filename") != expected_filename
        ):
            raise ValueError(f"Vecchia array specification is invalid for {name}.")
        if not _is_sha256(spec.get("file_sha256")) or not _is_sha256(
            spec.get("array_sha256")
        ):
            raise ValueError(f"Vecchia array checksums are invalid for {name}.")
        array = _load_authenticated_npy(
            directory_fd,
            expected_filename,
            expected_file_sha256=str(spec["file_sha256"]),
            context=f"Vecchia array {name}",
        )
        if array.dtype != _ARRAY_DTYPES[name]:
            raise ValueError(f"Vecchia array dtype is invalid for {name}.")
        if array.dtype.str != spec.get("dtype"):
            raise ValueError(f"Vecchia manifest dtype is invalid for {name}.")
        if list(array.shape) != spec.get("shape"):
            raise ValueError(f"Vecchia array shape is invalid for {name}.")
        if int(array.nbytes) != spec.get("nbytes"):
            raise ValueError(f"Vecchia array byte count is invalid for {name}.")
        if _array_sha256(array) != spec.get("array_sha256"):
            raise ValueError(f"Vecchia array content checksum failed for {name}.")
        arrays[name] = array

    return arrays


def _make_factorization(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> VecchiaFactorization:
    return VecchiaFactorization(
        image_shape=tuple(int(value) for value in manifest["image_shape"]),
        valid_flat_indices=arrays["valid_flat_indices"],
        neighbor_indptr=arrays["neighbor_indptr"],
        neighbor_indices=arrays["neighbor_indices"],
        weights=arrays["weights"],
        conditional_variances=arrays["conditional_variances"],
        max_neighbors=int(manifest["max_neighbors"]),
        correlation_logdet=float(manifest["correlation_logdet"]),
        correlation_kernel_sha256=str(
            manifest["source_binding"]["correlation_kernel_sha256"]
        ),
        valid_mask_sha256=str(
            manifest["source_binding"]["fitting_mask_sha256"]
        ),
        n_unique_patterns=int(manifest["n_unique_patterns"]),
        min_conditional_variance=float(manifest["min_conditional_variance"]),
        ordering=VECCHIA_ORDERING,
        approximation="nearest_neighbor_conditional_gaussian",
        schema=VECCHIA_SCHEMA,
        version=VECCHIA_FACTORIZATION_VERSION,
    )


def _validate_loaded_factorization(
    factorization: VecchiaFactorization,
    manifest: Mapping[str, Any],
    *,
    max_neighbors: int,
    min_conditional_variance: float,
) -> None:
    if int(manifest["max_neighbors"]) != max_neighbors:
        raise ValueError("Vecchia artifact does not match max_neighbors.")
    if float(manifest["min_conditional_variance"]) != min_conditional_variance:
        raise ValueError(
            "Vecchia artifact does not match min_conditional_variance."
        )
    if factorization.max_neighbors != max_neighbors:
        raise ValueError("Loaded Vecchia factorization has an invalid order.")
    if factorization.min_conditional_variance != min_conditional_variance:
        raise ValueError(
            "Loaded Vecchia factorization has an invalid conditional-variance "
            "threshold."
        )
    if factorization.correlation_kernel_sha256 != manifest[
        "source_binding"
    ]["correlation_kernel_sha256"]:
        raise ValueError("Loaded Vecchia factorization has an invalid kernel hash.")
    if factorization.valid_mask_sha256 != manifest["source_binding"][
        "fitting_mask_sha256"
    ]:
        raise ValueError("Loaded Vecchia factorization has an invalid mask hash.")
    computed_logdet = float(
        np.sum(np.log(factorization.conditional_variances), dtype=np.float64)
    )
    if computed_logdet != float(manifest["correlation_logdet"]):
        raise ValueError("Vecchia factorization log determinant is inconsistent.")


def _factorization_arrays(
    factorization: VecchiaFactorization,
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(getattr(factorization, name)) for name in _ARRAY_FILES
    }


def _validate_array_contract(
    arrays: Mapping[str, np.ndarray],
    factorization: VecchiaFactorization,
) -> None:
    n_valid = arrays["conditional_variances"].size
    n_edges = arrays["weights"].size
    if n_valid != factorization.n_valid_pixels:
        raise ValueError("Vecchia factorization valid-pixel count is inconsistent.")
    if n_edges != factorization.n_neighbor_links:
        raise ValueError("Vecchia factorization neighbor-link count is inconsistent.")
    expected_shapes = {
        "valid_flat_indices": (n_valid,),
        "neighbor_indptr": (n_valid + 1,),
        "neighbor_indices": (n_edges,),
        "weights": (n_edges,),
        "conditional_variances": (n_valid,),
    }
    for name, array in arrays.items():
        if array.dtype != _ARRAY_DTYPES[name]:
            raise ValueError(
                f"Vecchia factorization {name} must have dtype "
                f"{_ARRAY_DTYPES[name]}."
            )
        if array.shape != expected_shapes[name]:
            raise ValueError(f"Vecchia factorization {name} has invalid shape.")
        if not array.flags.c_contiguous:
            raise ValueError(f"Vecchia factorization {name} must be C-contiguous.")


def _capture_source_snapshot(
    resolved: dict[str, Any],
) -> _SourceSnapshot:
    """Capture owned source arrays and bind the build to those exact bytes."""
    products = load_image_preparation_products(resolved)
    calibration_path = correlated_noise_calibration_path(resolved)
    calibration_state = _file_state(calibration_path)
    calibration = load_correlated_noise_calibration(calibration_path)
    reject_correlated_noise_calibration_settings(resolved)
    _validate_calibration_kernel(calibration)
    validate_correlated_noise_calibration_estimator(calibration, resolved)
    validate_correlated_noise_calibration_sources(
        calibration,
        products,
    )
    _validate_source_shapes(products)

    cutout = _stable_array_snapshot(products.cutout, name="cutout")
    prepared_image = _stable_array_snapshot(
        products.prepared_data,
        name="prepared image",
    )
    sky_mask = _stable_array_snapshot(
        products.sky_region.mask,
        name="sky mask",
        dtype=bool,
    )
    fitting_mask = _stable_array_snapshot(
        products.fitting_mask,
        name="fitting mask",
        dtype=bool,
    )
    correlation_kernel = _stable_array_snapshot(
        calibration.correlation_kernel,
        name="correlation kernel",
        dtype=np.float64,
    )

    calibration_sha256 = _stable_file_sha256(calibration_path)
    if _file_state(calibration_path) != calibration_state:
        raise RuntimeError(
            "Correlated-noise calibration changed while its source inputs "
            "were captured."
        )

    binding = {
        "cutout_sha256": _array_sha256(cutout),
        "prepared_image_sha256": _array_sha256(prepared_image),
        "sky_mask_sha256": _array_sha256(sky_mask),
        "fitting_mask_sha256": _array_sha256(fitting_mask),
        "correlation_kernel_sha256": _array_sha256(correlation_kernel),
        "correlated_noise_calibration_sha256": calibration_sha256,
        "correlation_estimator": calibration.metadata["estimator"],
        "calibration_generator_identity": _json_value(
            calibration.metadata.get("generator_identity")
        ),
    }
    return _SourceSnapshot(
        fitting_mask=fitting_mask,
        correlation_kernel=correlation_kernel,
        binding=binding,
    )


def _stable_array_snapshot(
    value: Any,
    *,
    name: str,
    dtype: Any | None = None,
) -> np.ndarray:
    """Copy one array and reject mutation during the snapshot operation."""
    source = np.asarray(value)
    before = _array_sha256(source)
    snapshot = np.array(source, dtype=dtype, order="C", copy=True)
    after = _array_sha256(source)
    if before != after:
        raise RuntimeError(f"{name.capitalize()} changed while it was copied.")
    snapshot.setflags(write=False)
    return snapshot


def _validate_source_shapes(products: Any) -> None:
    if products.sky_region is None:
        raise RuntimeError("Vecchia factorization requires a persisted sky region.")
    shape = np.asarray(products.prepared_data).shape
    if len(shape) != 2:
        raise ValueError("Prepared image must be two-dimensional.")
    for name, value in (
        ("cutout", products.cutout),
        ("fitting_mask", products.fitting_mask),
        ("sky_mask", products.sky_region.mask),
    ):
        if np.asarray(value).shape != shape:
            raise ValueError(f"{name} must match the prepared image shape.")


def _validate_calibration_kernel(calibration: Any) -> None:
    """Require the sky correlation of a declared estimator and the lag of AD-047."""
    if calibration.metadata.get("estimator") not in SKY_CORRELATION_ESTIMATORS.values():
        raise ValueError(
            "Vecchia factorization requires the sky correlation estimated "
            f"by one of {sorted(SKY_CORRELATION_ESTIMATORS.values())} "
            "(AD-047, AD-051)."
        )
    if calibration.metadata.get("max_lag") != CORRELATION_MAX_LAG:
        raise ValueError(
            f"Vecchia factorization requires a correlation to lag "
            f"{CORRELATION_MAX_LAG} (AD-047)."
        )
    expected_shape = (2 * CORRELATION_MAX_LAG + 1, 2 * CORRELATION_MAX_LAG + 1)
    if np.asarray(calibration.correlation_kernel).shape != expected_shape:
        raise ValueError(
            "Correlation-kernel shape does not match the lag of AD-047."
        )


def _array_spec(
    array: np.ndarray,
    path: Path,
    filename: str,
) -> dict[str, Any]:
    return {
        "filename": filename,
        "dtype": array.dtype.str,
        "shape": [int(value) for value in array.shape],
        "nbytes": int(array.nbytes),
        "array_sha256": _array_sha256(array),
        "file_sha256": _file_sha256(path),
    }


def _array_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and C-order bytes without materializing a copy."""
    values = np.asarray(array)
    if not values.flags.c_contiguous:
        values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    view = memoryview(values).cast("B")
    block_size = 8 * 1024 * 1024
    for start in range(0, len(view), block_size):
        digest.update(view[start : start + block_size])
    return digest.hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_state(path: Path) -> tuple[int, int, int, int, int]:
    """Return stable fields used to detect concurrent artifact mutation."""
    state = path.stat()
    return (
        state.st_dev,
        state.st_ino,
        state.st_size,
        state.st_mtime_ns,
        state.st_ctime_ns,
    )


def _stable_file_sha256(path: Path) -> str:
    """Hash one source file and reject a concurrent byte change."""
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(
                f"Source file must not be a symbolic link: {path}"
            ) from exc
        raise
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Source file must be a regular file: {path}")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeError(f"Source file changed while it was hashed: {path}")
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _generator_identity() -> dict[str, Any]:
    core_path = getattr(vecchia_module, "__file__", None)
    if core_path is None:
        raise RuntimeError("Cannot locate Vecchia implementation source.")
    try:
        version = importlib.metadata.version("baymorph")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "pipeline_source_sha256": _stable_file_sha256(Path(__file__)),
        "core_source_sha256": _stable_file_sha256(Path(core_path)),
        "baymorph_version": version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }


def _manifest_payload_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("payload_sha256", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON-safe.")


def _write_npy(path: Path, array: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        _json_value(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_file_exclusive(source: Path, destination: Path) -> None:
    """Publish one staged file without replacing an existing directory entry."""
    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Vecchia artifact entry already exists: {destination}"
        ) from exc
    source.unlink()


def _wait_for_completed_bundle(path: Path) -> None:
    """Wait briefly for a competing atomic publication to become complete."""
    deadline = time.monotonic() + VECCHIA_PUBLICATION_WAIT_SECONDS
    marker = path / VECCHIA_COMPLETE_FILENAME
    while not marker.is_file() and time.monotonic() < deadline:
        time.sleep(VECCHIA_PUBLICATION_POLL_SECONDS)
    if not marker.is_file():
        raise RuntimeError(
            "Vecchia factorization directory exists but is incomplete. A "
            "publisher may have failed. Keep it untouched for diagnosis, "
            "then use a new immutable run or remove only this reviewed "
            f"incomplete artifact directory: {path}"
        )


def _open_artifact_directory(path: Path) -> int:
    """Open one real artifact directory and pin its inode for a load."""
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Vecchia factorization not found: {path}"
        ) from exc
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(
                f"Vecchia artifact directory is invalid: {path}"
            ) from exc
        raise
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"Vecchia artifact path is not a directory: {path}")
    return descriptor


def _open_regular_file_at(
    directory_fd: int,
    filename: str,
    *,
    context: str,
) -> int:
    """Open a non-symlink regular file relative to a pinned directory."""
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(filename, flags, dir_fd=directory_fd)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{context} is missing: {filename}") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"{context} must not be a symbolic link.") from exc
        raise
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"{context} must be a regular file.")
    return descriptor


def _read_regular_file_at(
    directory_fd: int,
    filename: str,
    *,
    context: str,
) -> bytes:
    """Read stable bytes from one file in a pinned artifact directory."""
    descriptor = _open_regular_file_at(
        directory_fd,
        filename,
        context=context,
    )
    try:
        before = os.fstat(descriptor)
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeError(f"{context} changed while it was read.")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise RuntimeError(f"{context} changed while it was read.")
    return payload


def _parse_json_bytes(payload: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is not valid UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object.")
    return value


def _load_authenticated_manifest(directory_fd: int) -> dict[str, Any]:
    """Authenticate exact manifest bytes from one pinned directory inode."""
    completion = _parse_json_bytes(
        _read_regular_file_at(
            directory_fd,
            VECCHIA_COMPLETE_FILENAME,
            context="Vecchia completion marker",
        ),
        "Vecchia completion marker",
    )
    manifest_bytes = _read_regular_file_at(
        directory_fd,
        VECCHIA_MANIFEST_FILENAME,
        context="Vecchia manifest",
    )
    expected_completion = {
        "schema": VECCHIA_SCHEMA,
        "schema_version": VECCHIA_SCHEMA_VERSION,
        "manifest": VECCHIA_MANIFEST_FILENAME,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    if completion != expected_completion:
        raise ValueError(
            "Vecchia completion marker does not authenticate the manifest."
        )
    manifest = _parse_json_bytes(manifest_bytes, "Vecchia manifest")
    if manifest.get("payload_sha256") != _manifest_payload_sha256(manifest):
        raise ValueError("Vecchia manifest payload checksum is invalid.")
    _validate_manifest_structure(manifest)
    return manifest


def _load_authenticated_npy(
    directory_fd: int,
    filename: str,
    *,
    expected_file_sha256: str,
    context: str,
) -> np.ndarray:
    """Authenticate and memory-map one NPY file through the same open inode."""
    descriptor = _open_regular_file_at(
        directory_fd,
        filename,
        context=context,
    )
    stream = os.fdopen(descriptor, "rb", closefd=True)
    try:
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        after_hash = os.fstat(stream.fileno())
        if _stat_identity(before) != _stat_identity(after_hash):
            raise RuntimeError(f"{context} changed while it was hashed.")
        if digest.hexdigest() != expected_file_sha256:
            name = context.removeprefix("Vecchia array ")
            raise ValueError(f"Vecchia array file checksum failed for {name}.")

        stream.seek(0)
        version = np.lib.format.read_magic(stream)
        readers = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }
        reader = readers.get(version)
        if reader is None:
            raise ValueError(f"{context} uses unsupported NPY version {version}.")
        shape, fortran_order, dtype = reader(stream)
        if dtype.hasobject:
            raise ValueError(f"{context} must not contain Python objects.")
        if fortran_order:
            raise ValueError(f"{context} must use C-order storage.")
        offset = stream.tell()
        expected_size = offset + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if expected_size != before.st_size:
            raise ValueError(f"{context} has an invalid NPY payload size.")
        array = np.memmap(
            stream,
            dtype=dtype,
            mode="r",
            offset=offset,
            shape=shape,
            order="C",
        )
        after_map = os.fstat(stream.fileno())
        if _stat_identity(before) != _stat_identity(after_map):
            raise RuntimeError(f"{context} changed while it was mapped.")
        return array
    finally:
        stream.close()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            unsupported = {
                errno.EINVAL,
                getattr(errno, "ENOTSUP", errno.EINVAL),
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            }
            if exc.errno not in unsupported:
                raise
    finally:
        os.close(descriptor)


def _validate_settings(
    max_neighbors: int,
    min_conditional_variance: float,
) -> tuple[int, float]:
    if isinstance(max_neighbors, (bool, np.bool_)) or not isinstance(
        max_neighbors,
        (int, np.integer),
    ):
        raise TypeError("max_neighbors must be an integer.")
    max_neighbors = int(max_neighbors)
    if max_neighbors not in SUPPORTED_VECCHIA_NEIGHBOR_ORDERS:
        raise ValueError(
            "Vecchia max_neighbors must be one of the frozen orders "
            f"{SUPPORTED_VECCHIA_NEIGHBOR_ORDERS}."
        )
    if isinstance(min_conditional_variance, bool):
        raise ValueError("min_conditional_variance must be numeric.")
    min_conditional_variance = float(min_conditional_variance)
    if (
        not np.isfinite(min_conditional_variance)
        or min_conditional_variance <= 0
        or min_conditional_variance >= 1
    ):
        raise ValueError(
            "min_conditional_variance must be finite and lie between 0 and 1."
        )
    return max_neighbors, min_conditional_variance


def _artifact_path_and_config(
    path_or_config: str | Path | dict[str, Any],
    *,
    max_neighbors: int,
    require_config: bool,
) -> tuple[Path, dict[str, Any] | None]:
    if isinstance(path_or_config, dict):
        resolved = _resolve_config(path_or_config)
        return (
            vecchia_factorization_path(
                resolved,
                max_neighbors=max_neighbors,
            ),
            resolved,
        )

    candidate = Path(path_or_config).expanduser()
    is_artifact_path = (
        candidate.name == f"m{max_neighbors}"
        and candidate.parent.name == DEFAULT_VECCHIA_FACTORIZATION_DIRECTORY
    )
    if is_artifact_path:
        if require_config:
            raise ValueError(
                "Loading a Vecchia factorization by artifact path is disabled "
                "for inference. Pass the observation config so all prepared "
                "and calibrated source bindings are revalidated."
            )
        return candidate.resolve(strict=False), None

    resolved = _resolve_config(candidate)
    return (
        vecchia_factorization_path(
            resolved,
            max_neighbors=max_neighbors,
        ),
        resolved,
    )


def _resolve_config(
    config: str | Path | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    return load_resolved_config(config)


__all__ = [
    "DEFAULT_MIN_CONDITIONAL_VARIANCE",
    "DEFAULT_VECCHIA_FACTORIZATION_DIRECTORY",
    "PRODUCTION_VECCHIA_NEIGHBORS",
    "SUPPORTED_VECCHIA_NEIGHBOR_ORDERS",
    "VALIDATION_VECCHIA_NEIGHBORS",
    "VECCHIA_ALGORITHM",
    "VECCHIA_COMPLETE_FILENAME",
    "VECCHIA_DISTANCE_METRIC",
    "VECCHIA_MANIFEST_FILENAME",
    "VECCHIA_NEIGHBOR_POLICY",
    "VECCHIA_SCHEMA",
    "VECCHIA_SCHEMA_VERSION",
    "VECCHIA_TIE_BREAK",
    "build_vecchia_factorization_artifact",
    "ensure_vecchia_factorization_artifact",
    "load_vecchia_factorization_artifact",
    "load_vecchia_factorization_manifest",
    "vecchia_factorization_manifest_path",
    "vecchia_factorization_path",
]
