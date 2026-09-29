"""Image-preparation workflows for real BayMorph FITS images."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from baymorph.core.background import (
    estimate_background_constant,
    estimate_background_photutils,
    subtract_background,
)
from baymorph.core.detection import (
    DetectionResult,
    detect_sources_segmentation,
)
from baymorph.core.noise import (
    estimate_noise_map,
    make_constant_noise_map,
)
from baymorph.core import (
    build_fitting_mask,
    load_external_mask,
    negative_outlier_mask,
    union,
)
from baymorph.core.observation import Observation
from baymorph.core.sky import (
    PlateauResult,
    RadialProfile,
    SkyRegion,
    build_sky_region,
)
from baymorph.pipeline.checkpoint_signing import (
    write_json_atomic as _write_json_atomic,
)
from baymorph.pipeline.config import (
    load_resolved_config,
    resolve_runtime_config,
)
from baymorph.pipeline.cutout import (
    CutoutDefinition,
    resolve_cutout,
    resolve_plateau_cutout,
    resolve_preliminary_auto_cutout,
    validate_plateau_cutout_containment,
)
from baymorph.pipeline.observation_metadata import observation_metadata_dict
from baymorph.pipeline.outputs import (
    make_output_tree,
    pipeline_paths,
    save_resolved_config,
)
from baymorph.pipeline.preparation_steps import (
    AutoCutoutResult,
    TargetGeometry,
    estimate_target_sky,
)
from baymorph.pipeline.prepared_identity import (
    build_prepared_input_identity,
    validate_prepared_input_identity,
)
from baymorph.pipeline.runtime_report import (
    build_image_preparation_runtime,
    build_runtime_report,
    save_runtime_report,
)
from baymorph.pipeline.segmentation import (
    select_segment_near_position,
)


_PREPARATION_BUNDLE_SCHEMA_VERSION = 1
_PREPARATION_MANIFEST_NAME = "image_preparation_artifacts.json"
_PREPARATION_COMPLETE_NAME = "IMAGE_PREPARATION_COMPLETE.json"
_PREPARATION_METADATA_NAME = "image_preparation_metadata.json"

_REQUIRED_PREPARATION_ARTIFACTS = frozenset(
    {
        "prepared_image.fits",
        "noise_map.fits",
        "mask.fits",
        "fitting_mask.fits",
        "large_cutout.npy",
        "target_segment.npy",
        "segmentation.npy",
        "automatic_mask.npy",
        "fitting_mask.npy",
        _PREPARATION_METADATA_NAME,
    }
)


@dataclass
class RealFITSPreparation:
    """Prepared real-image products before target-specific masking.

    Parameters
    ----------
    data : ndarray
        Background-subtracted image data.
    background : float
        Estimated scalar background level.
    mask : ndarray of bool
        Automatic source-detection mask.
    noise : ndarray
        Estimated noise map.
    detection : DetectionResult
        Source-detection products.
    """

    data: np.ndarray
    background: float
    mask: np.ndarray
    noise: np.ndarray
    detection: DetectionResult


def _as_finite_image(data, *, context: str) -> np.ndarray:
    """Return a finite two-dimensional floating-point science image."""
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError(
            f"Science data for {context} must be two-dimensional; "
            f"received shape {array.shape}."
        )
    invalid = ~np.isfinite(array)
    if np.any(invalid):
        raise ValueError(
            f"Science data for {context} contain "
            f"{int(np.count_nonzero(invalid))} non-finite pixels. "
            "Resolve non-finite input pixels before image preparation."
        )
    return array


def _resolve_fitting_mask_mode(data_cfg: dict[str, Any]) -> str:
    """Return the configured fitting-mask composition mode."""
    mode = str(data_cfg.get("mask_mode", "union")).strip().lower()
    if mode not in {"union", "exact", "replace"}:
        raise ValueError(
            "Config data.mask_mode must be one of 'union', 'exact', or "
            "'replace'."
        )
    return mode


def _configured_external_mask_path(
    data_cfg: Mapping[str, Any],
) -> str | Path | None:
    """Return the configured curated-mask path, if any."""
    if "external_mask_path" in data_cfg:
        raise ValueError(
            "Config data.external_mask_path is no longer supported. "
            "Declare the curated mask under data.mask_path."
        )

    return data_cfg.get("mask_path")


def _validate_prepared_arrays(
    *,
    prepared_data: np.ndarray,
    noise: np.ndarray,
) -> None:
    """Fail before persistence when final inference arrays are invalid."""
    if not np.all(np.isfinite(prepared_data)):
        raise RuntimeError(
            "Background-subtracted fitting data contain non-finite pixels."
        )
    if noise.shape != prepared_data.shape:
        raise RuntimeError(
            "The final noise map shape does not match the prepared image."
        )
    if not np.all(np.isfinite(noise)) or np.any(noise <= 0.0):
        raise RuntimeError(
            "The final noise map must contain only finite positive values."
        )


def _build_geometry_target_segment(
    target_segment: np.ndarray,
    fitting_mask: np.ndarray,
    *,
    contaminant_mask: np.ndarray | None = None,
    fitting_mask_mode: str = "union",
) -> np.ndarray:
    """Build target support after applying the scientific fitting mask.

    The fitting mask controls which pixels enter inference. It does not
    establish semantic ownership of unmasked automatic contaminants.
    Geometry therefore remains anchored to the detected target segment.
    """
    target = np.asarray(
        target_segment,
        dtype=bool,
    ).copy()

    exclusions = np.asarray(
        fitting_mask,
        dtype=bool,
    )

    if target.shape != exclusions.shape:
        raise ValueError(
            "Target and fitting masks must have the same shape."
        )

    mode = str(
        fitting_mask_mode
    ).strip().lower()

    if mode not in {
        "union",
        "exact",
        "replace",
    }:
        raise ValueError(
            "Fitting-mask mode must be one of 'union', 'exact', or "
            "'replace'."
        )

    if mode in {
        "exact",
        "replace",
    }:
        if contaminant_mask is None:
            raise ValueError(
                "Automatic contaminant support is required to construct "
                "geometry in exact fitting-mask mode."
            )

        contaminants = np.asarray(
            contaminant_mask,
            dtype=bool,
        )

        if contaminants.shape != target.shape:
            raise ValueError(
                "Target and automatic contaminant masks must have the same "
                "shape."
            )

    target &= ~exclusions

    return np.asarray(
        target,
        dtype=bool,
    )

def _validate_products_for_persistence(products: PreparedObservation) -> None:
    """Validate persisted preparation products before the first write."""
    prepared_data = _as_finite_image(
        products.prepared_data,
        context="persisted prepared image",
    )
    _validate_prepared_arrays(
        prepared_data=prepared_data,
        noise=np.asarray(products.noise, dtype=float),
    )

    cutout = _as_finite_image(
        products.cutout,
        context="persisted science cutout",
    )
    if cutout.shape != prepared_data.shape:
        raise ValueError(
            "The science cutout shape must match the prepared image."
        )

    for name in (
        "automatic_mask",
        "fitting_mask",
        "target_segment",
        "contaminant_mask",
    ):
        mask = build_fitting_mask(getattr(products, name))
        if mask.shape != prepared_data.shape:
            raise ValueError(
                f"'{name}' shape must match the prepared image."
            )

    if products.external_mask is not None:
        external_mask = build_fitting_mask(products.external_mask)
        if external_mask.shape != prepared_data.shape:
            raise ValueError(
                "'external_mask' shape must match the prepared image."
            )

    segmentation = _as_finite_image(
        products.segmentation,
        context="persisted segmentation",
    )
    if segmentation.shape != prepared_data.shape:
        raise ValueError(
            "The segmentation shape must match the prepared image."
        )
    if not np.isfinite(float(products.background)):
        raise ValueError("The persisted background level must be finite.")

    expected_prepared = cutout - float(products.background)
    if not np.array_equal(prepared_data, expected_prepared):
        raise RuntimeError(
            "The persisted prepared image is not the science cutout minus "
            "the recorded constant background."
        )
    if not np.any(np.asarray(products.target_segment, dtype=bool)):
        raise RuntimeError("The persisted target support contains no pixels.")


def _normalized_cutout_bounds(
    bounds: Mapping[str, Any] | None,
    *,
    source_shape: tuple[int, int],
    cutout_shape: tuple[int, int],
) -> dict[str, int]:
    """Validate persisted cutout bounds against source and cutout shapes."""
    source_ny, source_nx = source_shape

    if bounds is None:
        if cutout_shape != source_shape:
            raise RuntimeError(
                "Saved image-preparation products have no cutout bounds and "
                "do not cover the complete source FITS image."
            )
        return {
            "xmin": 0,
            "xmax": source_nx,
            "ymin": 0,
            "ymax": source_ny,
        }

    if not isinstance(bounds, Mapping):
        raise RuntimeError("Saved cutout bounds must be a mapping.")

    normalized: dict[str, int] = {}
    for name in ("xmin", "xmax", "ymin", "ymax"):
        value = bounds.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise RuntimeError(
                f"Saved cutout bound {name!r} must be an integer."
            )
        normalized[name] = int(value)

    if not (
        0 <= normalized["xmin"] < normalized["xmax"] <= source_nx
        and 0 <= normalized["ymin"] < normalized["ymax"] <= source_ny
    ):
        raise RuntimeError(
            "Saved cutout bounds lie outside the current source FITS image."
        )

    expected_shape = (
        normalized["ymax"] - normalized["ymin"],
        normalized["xmax"] - normalized["xmin"],
    )
    if expected_shape != cutout_shape:
        raise RuntimeError(
            "Saved cutout bounds do not match the persisted cutout shape."
        )
    return normalized


def _validate_products_against_current_sources(
    config: Mapping[str, Any],
    products: PreparedObservation,
    *,
    observation: Observation | None = None,
) -> None:
    """Verify cutout and mask semantics against immutable current sources."""
    if observation is None:
        observation = Observation.from_fits(config["data"]["image_path"])

    source = _as_finite_image(
        observation.image.data,
        context="current source FITS image",
    )
    cutout = _as_finite_image(
        products.cutout,
        context="persisted source cutout",
    )
    bounds = _normalized_cutout_bounds(
        products.cutout_bounds,
        source_shape=source.shape,
        cutout_shape=cutout.shape,
    )
    expected_cutout = source[
        bounds["ymin"]:bounds["ymax"],
        bounds["xmin"]:bounds["xmax"],
    ]
    if not np.array_equal(cutout, expected_cutout):
        raise RuntimeError(
            "Persisted science cutout does not match the current source "
            "FITS pixels at the saved bounds."
        )

    if products.original_image is not None:
        original_image = _as_finite_image(
            products.original_image,
            context="persisted original image",
        )
        if not np.array_equal(original_image, source):
            raise RuntimeError(
                "Persisted original image does not match the current source "
                "FITS pixels."
            )

    data_config = config["data"]
    mode = _resolve_fitting_mask_mode(dict(data_config))
    saved_mode = str(products.fitting_mask_mode).strip().lower()
    if saved_mode != mode:
        raise RuntimeError(
            "Persisted fitting-mask mode does not match the current "
            "configuration."
        )

    configured_mask_path = _configured_external_mask_path(data_config)
    current_mask_crop = None
    if configured_mask_path is not None:
        full_mask = load_external_mask(
            configured_mask_path,
            expected_shape=source.shape,
        )
        current_mask_crop = full_mask[
            bounds["ymin"]:bounds["ymax"],
            bounds["xmin"]:bounds["xmax"],
        ].astype(bool, copy=True)

    if current_mask_crop is None:
        if products.external_mask is not None:
            raise RuntimeError(
                "Persisted external mask has no current configured source."
            )
    elif products.external_mask is None or not np.array_equal(
        np.asarray(products.external_mask, dtype=bool),
        current_mask_crop,
    ):
        raise RuntimeError(
            "Persisted external mask does not match the current curated "
            "mask crop."
        )

    segmentation = np.asarray(products.segmentation)
    expected_contaminants = (segmentation > 0) & (
        segmentation != int(products.target_label)
    )
    contaminants = np.asarray(products.contaminant_mask, dtype=bool)
    if not np.array_equal(contaminants, expected_contaminants):
        raise RuntimeError(
            "Persisted contaminant mask does not match the segmentation and "
            "target label."
        )

    composition_mask = build_fitting_mask(
        expected_contaminants,
        external_mask=current_mask_crop,
        mode=mode,
    )
    expected_fitting_mask, _ = _apply_negative_outlier_mask(
        composition_mask,
        prepared_data=products.prepared_data,
        noise=products.noise,
        threshold_sigma=_negative_threshold_sigma(config),
    )
    fitting_mask = np.asarray(products.fitting_mask, dtype=bool)
    if not np.array_equal(fitting_mask, expected_fitting_mask):
        raise RuntimeError(
            "Persisted fitting mask does not match the current curated-mask "
            "composition."
        )

    raw_target_segment = segmentation == int(products.target_label)
    expected_target_segment = _build_geometry_target_segment(
        raw_target_segment,
        composition_mask,
        contaminant_mask=expected_contaminants,
        fitting_mask_mode=mode,
    )
    if not np.array_equal(
        np.asarray(products.target_segment, dtype=bool),
        expected_target_segment,
    ):
        raise RuntimeError(
            "Persisted target support does not match the current fitting-mask "
            "geometry semantics."
        )


def _sha256_file(path: Path) -> str:
    """Return a stable SHA-256 digest for one regular file."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"Artifact changed while it was hashed: {path}")
    return digest.hexdigest()


def _artifact_manifest(
    directory: Path,
    artifact_paths: list[Path],
) -> dict[str, Any]:
    """Build a content manifest for a complete preparation bundle."""
    artifacts = {}
    for artifact_path in artifact_paths:
        resolved = artifact_path.resolve(strict=True)
        try:
            relative = resolved.relative_to(directory.resolve(strict=True))
        except ValueError as exc:
            raise RuntimeError(
                f"Preparation artifact lies outside its bundle: {artifact_path}"
            ) from exc
        name = relative.as_posix()
        if name in artifacts:
            raise RuntimeError(f"Duplicate preparation artifact: {name}")
        artifacts[name] = {
            "size_bytes": int(resolved.stat().st_size),
            "sha256": _sha256_file(resolved),
        }
    return {
        "schema_version": _PREPARATION_BUNDLE_SCHEMA_VERSION,
        "artifacts": artifacts,
    }


def _load_and_verify_preparation_bundle(directory: Path) -> frozenset[str]:
    """Verify completion, manifest, and every artifact before loading."""
    if not directory.is_dir() or not any(directory.iterdir()):
        raise FileNotFoundError(
            "Image-preparation products do not exist."
        )

    marker_path = directory / _PREPARATION_COMPLETE_NAME
    manifest_path = directory / _PREPARATION_MANIFEST_NAME

    if not marker_path.is_file():
        raise RuntimeError(
            "Image-preparation bundle is incomplete: completion marker is "
            "missing. Re-run image preparation in a new output namespace."
        )

    try:
        with marker_path.open("r", encoding="utf-8") as stream:
            marker = json.load(stream)
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(
            "Image-preparation completion marker is unreadable."
        ) from exc

    if not isinstance(marker, Mapping) or marker.get("schema_version") != 1:
        raise RuntimeError(
            "Image-preparation completion marker has an unsupported schema."
        )
    if marker.get("manifest") != _PREPARATION_MANIFEST_NAME:
        raise RuntimeError(
            "Image-preparation completion marker names an unexpected "
            "artifact manifest."
        )
    if not manifest_path.is_file():
        raise RuntimeError("Image-preparation artifact manifest is missing.")
    if marker.get("manifest_sha256") != _sha256_file(manifest_path):
        raise RuntimeError(
            "Image-preparation completion marker does not match the artifact "
            "manifest."
        )

    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(
            "Image-preparation artifact manifest is unreadable."
        ) from exc

    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != 1
        or not isinstance(manifest.get("artifacts"), Mapping)
    ):
        raise RuntimeError(
            "Image-preparation artifact manifest has an unsupported schema."
        )

    artifact_names = frozenset(str(name) for name in manifest["artifacts"])
    missing_required = _REQUIRED_PREPARATION_ARTIFACTS - artifact_names
    if missing_required:
        detail = ", ".join(sorted(missing_required))
        raise RuntimeError(
            "Image-preparation artifact manifest omits required artifacts: "
            f"{detail}."
        )

    root = directory.resolve(strict=True)
    for name, expected in manifest["artifacts"].items():
        if not isinstance(name, str) or not isinstance(expected, Mapping):
            raise RuntimeError(
                "Image-preparation artifact manifest contains an invalid "
                "entry."
            )
        candidate = (root / name).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                "Image-preparation artifact manifest contains an unsafe path."
            ) from exc

        if (
            not candidate.is_file()
            or expected.get("size_bytes") != candidate.stat().st_size
            or expected.get("sha256") != _sha256_file(candidate)
        ):
            raise RuntimeError(
                "Image-preparation artifact integrity check failed for "
                f"{name!r}."
            )

    return artifact_names


def prepare_real_fits_image(
    data,
    background_box_size: int = 20,
    background_filter_size: int = 3,
    detection_sigma: float = 3.0,
    detection_n_pixels: int = 5,
    detection_n_levels: int = 32,
):
    """Prepare a real image array for fitting.

    Parameters
    ----------
    data : array-like
        Input image.
    background_box_size : int, default=20
        Background mesh size used by Photutils.
    background_filter_size : int, default=3
        Median-filter size applied to the background mesh.
    detection_sigma : float, default=3.0
        Source-detection threshold in units of noise.
    detection_n_pixels : int, default=5
        Minimum connected pixels required for source detection.
    detection_n_levels : int, default=32
        Number of detection levels forwarded to the detection interface.

    Returns
    -------
    RealFITSPreparation
        Prepared image, background estimate, detection mask, noise map, and
        detection products.
    """
    data = _as_finite_image(
        data,
        context="source detection",
    )

    detection = detect_sources_segmentation(
        data,
        n_sigma=detection_sigma,
        n_pixels=detection_n_pixels,
        n_levels=detection_n_levels,
    )

    mask = detection.mask

    try:
        background_result = estimate_background_photutils(
            data,
            box_size=background_box_size,
            filter_size=background_filter_size,
            mask=mask,
        )
    except ValueError:
        background_result = estimate_background_constant(
            data,
        )

    background_subtracted = subtract_background(
        data,
        background_result.background,
    )

    noise = estimate_noise_map(
        background_subtracted,
        mask=mask,
    )

    return RealFITSPreparation(
        data=background_subtracted,
        background=background_result.background,
        mask=mask,
        noise=noise,
        detection=detection,
    )


@dataclass(slots=True)
class _TargetDetection:
    """Target-selection products for one image coordinate system."""

    prepared: RealFITSPreparation
    target_label: int
    target_segment: np.ndarray
    contaminant_mask: np.ndarray


def _negative_threshold_sigma(
    config: Mapping[str, Any],
) -> float | None:
    """Return the configured negative-artifact threshold, or None."""
    threshold = (
        config.get("preparation", {})
        .get("masking", {})
        .get("negative_threshold_sigma")
    )

    return None if threshold is None else float(threshold)


def _apply_negative_outlier_mask(
    fitting_mask: np.ndarray,
    *,
    prepared_data: np.ndarray,
    noise: np.ndarray,
    threshold_sigma: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exclude impossibly negative pixels from one fitting mask.

    The automatic mask is built from positive source detections, so no
    setting of the detection threshold can exclude a negative detector
    artifact such as a saturation trail. This criterion is applied in every
    fitting-mask mode: ``exact`` declares which curated source mask is
    authoritative, not that physically impossible pixels may enter the
    likelihood.

    Returns
    -------
    tuple of ndarray
        The fitting mask including the negative outliers, and the negative
        outlier mask itself.
    """
    if threshold_sigma is None:
        return (
            fitting_mask,
            np.zeros(
                fitting_mask.shape,
                dtype=bool,
            ),
        )

    negative_outliers = negative_outlier_mask(
        prepared_data,
        noise,
        threshold_sigma=threshold_sigma,
    )

    return (
        union(
            fitting_mask,
            negative_outliers,
        ),
        negative_outliers,
    )


def _detect_target_in_image(
    data: np.ndarray,
    *,
    target_x: float,
    target_y: float,
    detection_cfg: dict[str, Any],
    background_cfg: dict[str, Any],
    context: str,
) -> _TargetDetection:
    """Detect sources and select the target near a requested position."""
    detection_n_pixels = detection_cfg["n_pixels"]

    prepared = prepare_real_fits_image(
        data,
        background_box_size=background_cfg["box_size"],
        background_filter_size=background_cfg["filter_size"],
        detection_sigma=detection_cfg["n_sigma"],
        detection_n_pixels=detection_n_pixels,
        detection_n_levels=detection_cfg["n_levels"],
    )

    segmentation = prepared.detection.segmentation
    if segmentation is None:
        raise RuntimeError(f"No segmentation map was produced during {context}.")

    target_label = select_segment_near_position(
        segmentation,
        x=target_x,
        y=target_y,
    )

    target_segment = segmentation.data == target_label

    contaminant_mask = (segmentation.data > 0) & (segmentation.data != target_label)

    return _TargetDetection(
        prepared=prepared,
        target_label=int(target_label),
        target_segment=target_segment,
        contaminant_mask=contaminant_mask,
    )


def _resolve_auto_cutout(
    image: np.ndarray,
    *,
    target_x: float,
    target_y: float,
    cutout_cfg: dict[str, Any],
    detection_cfg: dict[str, Any],
    background_cfg: dict[str, Any],
    external_mask: np.ndarray | None = None,
    fitting_mask_mode: str = "union",
) -> AutoCutoutResult:
    """Resolve a plateau-based automatic cutout for a target source."""
    full_detection = _detect_target_in_image(
        image,
        target_x=target_x,
        target_y=target_y,
        detection_cfg=detection_cfg,
        background_cfg=background_cfg,
        context="full-image detection",
    )

    full_fitting_mask = build_fitting_mask(
        full_detection.contaminant_mask,
        external_mask=external_mask,
        mode=fitting_mask_mode,
    )
    full_target_segment = _build_geometry_target_segment(
        full_detection.target_segment,
        full_fitting_mask,
        contaminant_mask=full_detection.contaminant_mask,
        fitting_mask_mode=fitting_mask_mode,
    )

    preliminary_definition = resolve_preliminary_auto_cutout(
        image_shape=image.shape,
        target_segment=full_target_segment,
        center_x=target_x,
        center_y=target_y,
        initial_factor=float(cutout_cfg.get("initial_factor", 4.0)),
        min_size=int(cutout_cfg.get("min_size", 100)),
        max_size=cutout_cfg.get("max_size"),
    )

    preliminary_cutout = make_target_cutout(
        image,
        preliminary_definition,
    )

    preliminary_detection = _detect_target_in_image(
        preliminary_cutout,
        target_x=target_x - preliminary_definition.xmin,
        target_y=target_y - preliminary_definition.ymin,
        detection_cfg=detection_cfg,
        background_cfg=background_cfg,
        context="preliminary-cutout detection",
    )

    preliminary_external_mask = (
        make_target_cutout(
            external_mask,
            preliminary_definition,
        ).astype(bool)
        if external_mask is not None
        else None
    )

    preliminary_fitting_mask = build_fitting_mask(
        preliminary_detection.contaminant_mask,
        external_mask=preliminary_external_mask,
        mode=fitting_mask_mode,
    )
    preliminary_target_segment = _build_geometry_target_segment(
        preliminary_detection.target_segment,
        preliminary_fitting_mask,
        contaminant_mask=preliminary_detection.contaminant_mask,
        fitting_mask_mode=fitting_mask_mode,
    )

    geometry, radial_profile, plateau = estimate_target_sky(
        preliminary_cutout,
        preliminary_target_segment,
        mask=preliminary_fitting_mask,
        min_radius=0.0,
        window_size=5,
        sigma_threshold=1.0,
        outer_fraction=0.25,
        min_coverage=0.5,
    )

    global_center_x = preliminary_definition.xmin + geometry.x0
    global_center_y = preliminary_definition.ymin + geometry.y0

    final_cutout = resolve_plateau_cutout(
        image_shape=image.shape,
        center_x=global_center_x,
        center_y=global_center_y,
        plateau_radius=plateau.radius,
        sky_margin=float(cutout_cfg.get("sky_margin", 20.0)),
        min_size=cutout_cfg.get("min_size"),
        max_size=cutout_cfg.get("max_size"),
    )

    validate_plateau_cutout_containment(
        image_shape=(final_cutout.height, final_cutout.width),
        center_x=global_center_x - final_cutout.xmin,
        center_y=global_center_y - final_cutout.ymin,
        plateau_radius=plateau.radius,
        sky_margin=float(cutout_cfg.get("sky_margin", 20.0)),
    )

    return AutoCutoutResult(
        cutout=final_cutout,
        preliminary_cutout=preliminary_cutout,
        preliminary_segmentation=(
            preliminary_detection.prepared.detection.segmentation.data.copy()
        ),
        geometry=geometry,
        radial_profile=radial_profile,
        plateau=plateau,
    )


@dataclass
class PreparedObservation:
    """Target-specific image-preparation products.

    Parameters
    ----------
    cutout : ndarray
        Original target cutout before background subtraction.
    original_image : ndarray, optional
        Full original input image before cutout extraction.
    cutout_bounds : dict, optional
        Pixel bounds of the cutout in the original image, with keys
        ``xmin``, ``xmax``, ``ymin``, and ``ymax``.
    prepared_data : ndarray
        Background-subtracted cutout used for fitting.
    noise : ndarray
        Noise map associated with ``prepared_data``.
    automatic_mask : ndarray of bool
        Automatic source-detection mask.

    fitting_mask : ndarray of bool
        Canonical mask used during likelihood evaluation.

    target_segment : ndarray of bool
        Effective target support used for geometry and plateau estimation.
        Curated exclusions are removed without inferring target membership
        from unmasked pixels.
    contaminant_mask : ndarray of bool
        Automatic exclusion mask containing detected non-target sources.
    segmentation : ndarray
        Integer segmentation map.
    background : float
        Estimated scalar background level.
    target_x, target_y : int
        Measured target centre in cutout pixel coordinates.
    target_label : int
        Segmentation label assigned to the target source.
    cutout_mode : {"fixed", "auto"}, default="fixed"
        Cutout strategy used during image preparation.
    original_image : ndarray, optional
        Full original input image before cutout extraction.
    cutout_bounds : dict of str, int, optional
        Pixel bounds of the final cutout in the original image, with keys
        ``xmin``, ``xmax``, ``ymin``, and ``ymax``.
    external_mask : ndarray of bool, optional
        User-provided exclusion mask extracted onto the final cutout.
    fitting_mask_mode : {"union", "exact", "replace"}, default="union"
        Composition rule used to construct ``fitting_mask``.
    geometry : TargetGeometry, optional
        Target geometry measured on the final cutout.
    radial_profile : RadialProfile, optional
        Elliptical radial profile measured on the final cutout.
    plateau : PlateauResult, optional
        Plateau detected in the final target radial profile.
    sky_region : SkyRegion, optional
        Final sky region used for background and noise estimation.
    preliminary_cutout : ndarray, optional
        Preliminary adaptive cutout used while resolving an automatic cutout.
        It is None when ``cutout_mode`` is ``"fixed"``.
    preliminary_segmentation : ndarray, optional
        Segmentation map measured on the preliminary adaptive cutout. It is
        None when ``cutout_mode`` is ``"fixed"``.
    sky_gradient : dict of str to float, optional
        Plane fitted to the sky region while measuring the noise scale, with
        the gradient per axis, its standard error, and its significance. It is
        a diagnostic only: it never selects a background model. A gradient
        significant at several sigma indicates that
        ``fit.background.model: plane`` is warranted, which remains an
        explicit user decision.
    """

    cutout: np.ndarray
    prepared_data: np.ndarray
    noise: np.ndarray
    automatic_mask: np.ndarray
    fitting_mask: np.ndarray
    target_segment: np.ndarray
    contaminant_mask: np.ndarray
    segmentation: np.ndarray
    background: float
    target_x: int
    target_y: int
    target_label: int

    cutout_mode: str = "fixed"
    original_image: np.ndarray | None = None
    cutout_bounds: dict[str, int] | None = None
    external_mask: np.ndarray | None = None
    fitting_mask_mode: str = "union"

    geometry: TargetGeometry | None = None
    radial_profile: RadialProfile | None = None
    plateau: PlateauResult | None = None
    sky_region: SkyRegion | None = None

    preliminary_cutout: np.ndarray | None = None
    preliminary_segmentation: np.ndarray | None = None

    sky_gradient: dict[str, float] | None = None


def make_target_cutout(
    data: np.ndarray,
    cutout: CutoutDefinition,
) -> np.ndarray:
    """Extract a target cutout from an image.

    Parameters
    ----------
    data : ndarray
        Parent image.
    cutout : CutoutDefinition
        Resolved cutout bounds.

    Returns
    -------
    ndarray
        Extracted image cutout.
    """
    return data[
        cutout.ymin : cutout.ymax,
        cutout.xmin : cutout.xmax,
    ]


def prepare_real_image_from_config(
    config: str | Path | dict[str, Any],
    *,
    save: bool = True,
) -> PreparedObservation:
    """Prepare a real galaxy image from a BayMorph configuration.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Resolved configuration dictionary or path to a YAML configuration file.
    save : bool, default=True
        If True, save image-preparation products to the configured output tree.

    Returns
    -------
    PreparedObservation
        Prepared target cutout, masks, segmentation products, and metadata.

    Raises
    ------
    ValueError
        If ``data.target_position`` is missing.
    RuntimeError
        If no segmentation map is produced.
    """
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = config

    make_output_tree(resolved)

    observation = Observation.from_fits(resolved["data"]["image_path"])

    resolved = resolve_runtime_config(
        resolved,
        observation,
    )

    input_identity = build_prepared_input_identity(
        resolved,
        observation=observation,
    )

    image = observation.image

    _as_finite_image(
        image.data,
        context="input FITS image",
    )

    target_position = resolved["data"]["target_position"]
    if target_position is None:
        raise ValueError(
            "Config data.target_position is required for image preparation."
        )

    cutout_cfg = resolved["data"]["cutout"]
    detection_cfg = resolved["preparation"]["detection"]
    background_cfg = resolved["preparation"]["background"]

    fitting_mask_mode = _resolve_fitting_mask_mode(resolved["data"])

    external_mask = None
    external_mask_path = _configured_external_mask_path(resolved["data"])
    if external_mask_path is not None:
        external_mask = load_external_mask(
            external_mask_path,
            expected_shape=image.data.shape,
        )
    elif fitting_mask_mode in {"exact", "replace"}:
        raise ValueError(
            "Config data.mask_path is required when data.mask_mode is "
            f"{fitting_mask_mode!r}."
        )

    cutout_mode = cutout_cfg.get("mode", "fixed")

    auto_cutout_result = None

    if cutout_mode == "fixed":
        cutout_definition = resolve_cutout(
            image_shape=image.data.shape,
            center_x=target_position["x"],
            center_y=target_position["y"],
            config=cutout_cfg,
        )

    elif cutout_mode == "auto":
        auto_cutout_result = _resolve_auto_cutout(
            image.data,
            target_x=target_position["x"],
            target_y=target_position["y"],
            cutout_cfg=cutout_cfg,
            detection_cfg=detection_cfg,
            background_cfg=background_cfg,
            external_mask=external_mask,
            fitting_mask_mode=fitting_mask_mode,
        )
        cutout_definition = auto_cutout_result.cutout

    else:
        raise ValueError(f"Unsupported cutout mode: {cutout_mode!r}.")

    cutout = make_target_cutout(
        image.data,
        cutout_definition,
    )

    full_external_mask = external_mask
    if full_external_mask is not None:
        external_mask = make_target_cutout(
            full_external_mask,
            cutout_definition,
        ).astype(bool)

    seed_target_x = float(target_position["x"]) - cutout_definition.xmin
    seed_target_y = float(target_position["y"]) - cutout_definition.ymin

    final_detection = _detect_target_in_image(
        cutout,
        target_x=seed_target_x,
        target_y=seed_target_y,
        detection_cfg=detection_cfg,
        background_cfg=background_cfg,
        context="final-cutout detection",
    )

    prepared = final_detection.prepared
    segmentation = prepared.detection.segmentation
    target_label = final_detection.target_label
    target_segment = final_detection.target_segment
    contaminant_mask = final_detection.contaminant_mask

    fitting_mask = build_fitting_mask(
        contaminant_mask,
        external_mask=external_mask,
        mode=fitting_mask_mode,
    )

    target_segment = _build_geometry_target_segment(
        target_segment,
        fitting_mask,
        contaminant_mask=contaminant_mask,
        fitting_mask_mode=fitting_mask_mode,
    )

    geometry, radial_profile, plateau = estimate_target_sky(
        cutout,
        target_segment,
        mask=fitting_mask,
        min_radius=0.0,
        window_size=5,
        sigma_threshold=1.0,
        outer_fraction=0.25,
        min_coverage=0.5,
    )

    target_x = int(round(geometry.x0))
    target_y = int(round(geometry.y0))

    validate_plateau_cutout_containment(
        image_shape=cutout.shape,
        center_x=geometry.x0,
        center_y=geometry.y0,
        plateau_radius=plateau.radius,
        sky_margin=float(cutout_cfg.get("sky_margin", 20.0)),
    )

    sky_region = build_sky_region(
        shape=cutout.shape,
        geometry=geometry,
        plateau=plateau,
        margin=5.0,
        mask=fitting_mask,
    )

    background_result = estimate_background_constant(
        cutout,
        mask=~sky_region.mask,
        noise_method="corrected_std",
        detrend=True,
    )

    prepared_data = subtract_background(
        cutout,
        background_result.background_level,
    )

    noise = make_constant_noise_map(
        prepared_data.shape,
        background_result.sigma,
    )

    _validate_prepared_arrays(
        prepared_data=prepared_data,
        noise=noise,
    )

    # Applied to the final likelihood support, after the sky region, the
    # background level and the noise scale have been estimated. Those
    # estimators are robust and clipped, so the few pixels excluded here do
    # not materially change them, and deferring keeps the target support and
    # the sky region exactly reconstructible from persisted products.
    fitting_mask, _negative_outliers = _apply_negative_outlier_mask(
        fitting_mask,
        prepared_data=prepared_data,
        noise=noise,
        threshold_sigma=_negative_threshold_sigma(resolved),
    )

    products = PreparedObservation(
        cutout=cutout,
        prepared_data=prepared_data,
        noise=noise,
        automatic_mask=prepared.mask,
        fitting_mask=fitting_mask,
        target_segment=target_segment,
        contaminant_mask=contaminant_mask,
        segmentation=segmentation.data,
        background=float(background_result.background_level),
        sky_gradient=(
            None
            if background_result.parameters is None
            else {
                key: float(value)
                for key, value in background_result.parameters.items()
            }
        ),
        target_x=target_x,
        target_y=target_y,
        target_label=int(target_label),
        cutout_mode=cutout_mode,
        geometry=geometry,
        radial_profile=radial_profile,
        plateau=plateau,
        sky_region=sky_region,
        preliminary_cutout=(
            auto_cutout_result.preliminary_cutout
            if auto_cutout_result is not None
            else None
        ),
        preliminary_segmentation=(
            auto_cutout_result.preliminary_segmentation
            if auto_cutout_result is not None
            else None
        ),
        original_image=image.data,
        external_mask=external_mask,
        fitting_mask_mode=fitting_mask_mode,
        cutout_bounds={
            "xmin": int(cutout_definition.xmin),
            "xmax": int(cutout_definition.xmax),
            "ymin": int(cutout_definition.ymin),
            "ymax": int(cutout_definition.ymax),
        },
    )

    if save:
        save_image_preparation_products(
            resolved,
            products,
            observation=observation,
            input_identity=input_identity,
        )
        save_resolved_config(resolved)

        runtime = build_image_preparation_runtime(
            products,
            observation=observation_metadata_dict(
                observation,
            ),
        )

        report = build_runtime_report(
            resolved,
            runtime=runtime,
        )

        save_runtime_report(
            resolved,
            report,
        )

    return products


def save_image_preparation_products(
    config: dict[str, Any],
    products: PreparedObservation,
    *,
    observation: Observation | None = None,
    input_identity: dict[str, Any] | None = None,
) -> None:
    """Save image-preparation products and diagnostics to disk.

    Parameters
    ----------
    config : dict
        Resolved BayMorph configuration.
    products : PreparedObservation
        Image-preparation products to save.
    observation : Observation, optional
        Original observation providing normalized observational metadata.
    input_identity : dict, optional
        Source identity captured when the FITS observation was acquired.

    Notes
    -----
    The saved products include the prepared image, noise map, masks,
    segmentation maps, cutouts, target geometry, radial profile, plateau,
    sky region, and scalar metadata when available.

    Returns
    -------
    None
        Products are written to the configured image-preparation directory.
    """
    _validate_products_for_persistence(products)

    if input_identity is None:
        input_identity = build_prepared_input_identity(
            config,
            observation=observation,
        )

    validate_prepared_input_identity(
        input_identity,
        config,
        observation=observation,
    )
    _validate_products_against_current_sources(
        config,
        products,
        observation=observation,
    )

    paths = pipeline_paths(config)["image_preparation"]
    overwrite = bool(config["outputs"].get("overwrite", False))
    directory = paths["dir"]
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / _PREPARATION_METADATA_NAME
    manifest_path = directory / _PREPARATION_MANIFEST_NAME
    completion_path = directory / _PREPARATION_COMPLETE_NAME

    artifact_paths = [
        paths["prepared_image"],
        paths["noise_map"],
        paths["mask"],
        paths["fitting_mask"],
        directory / "large_cutout.npy",
        directory / "target_segment.npy",
        directory / "segmentation.npy",
        directory / "automatic_mask.npy",
        directory / "fitting_mask.npy",
        metadata_path,
    ]
    optional_artifacts = (
        (products.external_mask, directory / "external_mask.npy"),
        (products.preliminary_cutout, directory / "preliminary_cutout.npy"),
        (
            products.preliminary_segmentation,
            directory / "preliminary_segmentation.npy",
        ),
        (products.radial_profile, directory / "radial_profile.npz"),
        (products.sky_region, directory / "sky_region.npz"),
        (products.original_image, directory / "original_image.npy"),
    )
    artifact_paths.extend(
        artifact_path
        for value, artifact_path in optional_artifacts
        if value is not None
    )

    if not overwrite:
        existing = [
            artifact_path
            for artifact_path in (
                *artifact_paths,
                manifest_path,
                completion_path,
            )
            if artifact_path.exists()
        ]
        if existing:
            raise FileExistsError(
                "Image-preparation output already exists and overwrite is "
                f"disabled: {existing[0]}"
            )
    else:
        completion_path.unlink(missing_ok=True)

    fits.writeto(
        paths["prepared_image"],
        products.prepared_data,
        overwrite=overwrite,
    )
    fits.writeto(
        paths["noise_map"],
        products.noise,
        overwrite=overwrite,
    )
    fits.writeto(
        paths["mask"],
        products.contaminant_mask.astype(np.uint8),
        overwrite=overwrite,
    )
    fits.writeto(
        paths["fitting_mask"],
        products.fitting_mask.astype(np.uint8),
        overwrite=overwrite,
    )

    np.save(paths["dir"] / "large_cutout.npy", products.cutout)
    np.save(paths["dir"] / "target_segment.npy", products.target_segment)
    np.save(paths["dir"] / "segmentation.npy", products.segmentation)

    np.save(
        paths["dir"] / "automatic_mask.npy",
        products.automatic_mask,
    )

    np.save(
        paths["dir"] / "fitting_mask.npy",
        products.fitting_mask,
    )

    if products.external_mask is not None:
        np.save(
            paths["dir"] / "external_mask.npy",
            products.external_mask,
        )

    if products.preliminary_cutout is not None:
        np.save(
            paths["dir"] / "preliminary_cutout.npy",
            products.preliminary_cutout,
        )

    if products.preliminary_segmentation is not None:
        np.save(
            paths["dir"] / "preliminary_segmentation.npy",
            products.preliminary_segmentation,
        )

    if products.radial_profile is not None:
        np.savez(
            paths["dir"] / "radial_profile.npz",
            radius=products.radial_profile.radius,
            profile=products.radial_profile.profile,
            n_pixels=products.radial_profile.n_pixels,
            scatter=products.radial_profile.scatter,
            area=products.radial_profile.area,
        )

    if products.sky_region is not None:
        np.savez(
            paths["dir"] / "sky_region.npz",
            mask=products.sky_region.mask,
            radius_map=products.sky_region.radius_map,
            inner_radius=products.sky_region.inner_radius,
            outer_radius=products.sky_region.outer_radius,
            n_pixels=products.sky_region.n_pixels,
            coverage=products.sky_region.coverage,
        )

    if products.original_image is not None:
        np.save(paths["dir"] / "original_image.npy", products.original_image)

    negative_threshold = _negative_threshold_sigma(config)
    _, persisted_negative_outliers = _apply_negative_outlier_mask(
        np.zeros(
            np.asarray(products.fitting_mask).shape,
            dtype=bool,
        ),
        prepared_data=products.prepared_data,
        noise=products.noise,
        threshold_sigma=negative_threshold,
    )

    metadata = {
        "input_identity": input_identity,
        "negative_outliers": {
            "threshold_sigma": negative_threshold,
            "excluded_pixels": int(
                np.count_nonzero(persisted_negative_outliers)
            ),
        },
        "observation": (
            observation_metadata_dict(observation) if observation is not None else None
        ),
        "background": products.background,
        "target_x": products.target_x,
        "target_y": products.target_y,
        "target_label": products.target_label,
        "cutout_mode": products.cutout_mode,
        "cutout_bounds": products.cutout_bounds,
        "fitting_mask_mode": products.fitting_mask_mode,
        "geometry": (
            {
                "x0": products.geometry.x0,
                "y0": products.geometry.y0,
                "q": products.geometry.q,
                "theta": products.geometry.theta,
            }
            if products.geometry is not None
            else None
        ),
        "plateau": (
            {
                "radius": products.plateau.radius,
                "index": products.plateau.index,
                "background_level": products.plateau.background_level,
                "noise_sigma": products.plateau.noise_sigma,
                "window_start": products.plateau.window_start,
                "window_stop": products.plateau.window_stop,
            }
            if products.plateau is not None
            else None
        ),
        "sky_gradient": products.sky_gradient,
    }

    validate_prepared_input_identity(
        input_identity,
        config,
        observation=observation,
    )
    _validate_products_against_current_sources(
        config,
        products,
        observation=observation,
    )
    _write_json_atomic(metadata_path, metadata)

    manifest = _artifact_manifest(directory, artifact_paths)
    _write_json_atomic(manifest_path, manifest)
    manifest_sha256 = _sha256_file(manifest_path)

    validate_prepared_input_identity(
        input_identity,
        config,
        observation=observation,
    )
    _write_json_atomic(
        completion_path,
        {
            "schema_version": _PREPARATION_BUNDLE_SCHEMA_VERSION,
            "manifest": _PREPARATION_MANIFEST_NAME,
            "manifest_sha256": manifest_sha256,
        },
    )


def image_preparation_is_complete(
    config: str | Path | dict[str, Any],
) -> bool:
    """Return whether a completed image preparation exists on disk.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Resolved configuration dictionary or path to a YAML configuration
        file.

    Returns
    -------
    bool
        True when the completion marker is present.

    Notes
    -----
    The marker is written last, after every required product has been
    validated, so its presence identifies a finished preparation. It does not
    assert that those products match the current inputs: that identity is
    validated, fail-closed, by :func:`load_image_preparation_products`.
    """
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = config

    directory = pipeline_paths(resolved)["image_preparation"]["dir"]

    return (directory / _PREPARATION_COMPLETE_NAME).is_file()


def load_image_preparation_products(
    config: str | Path | dict[str, Any],
) -> PreparedObservation:
    """Load saved image-preparation products and diagnostics.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Resolved configuration dictionary or path to a YAML configuration
        file.

    Returns
    -------
    PreparedObservation
        Reconstructed image-preparation products, including optional target
        geometry, radial profile, plateau, sky region, preliminary cutout,
        and preliminary segmentation when these were saved.

    Notes
    -----
    Persisted products are accepted only when their immutable FITS, curated
    mask, WCS, and preparation-configuration identity matches the current
    inputs. Products without this identity must be regenerated. The canonical
    fitting mask is mandatory.
    """
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = config

    paths = pipeline_paths(resolved)["image_preparation"]
    artifact_names = _load_and_verify_preparation_bundle(paths["dir"])

    with open(paths["dir"] / _PREPARATION_METADATA_NAME, "r") as f:
        metadata = json.load(f)

    validate_prepared_input_identity(
        metadata.get("input_identity"),
        resolved,
    )

    prepared_data = np.asarray(
        fits.getdata(paths["prepared_image"]),
        dtype=float,
    )
    noise = np.asarray(fits.getdata(paths["noise_map"]), dtype=float)
    _validate_prepared_arrays(
        prepared_data=prepared_data,
        noise=noise,
    )
    contaminant_mask = load_external_mask(
        paths["mask"],
        expected_shape=prepared_data.shape,
    )

    cutout = np.load(paths["dir"] / "large_cutout.npy")
    target_segment = np.load(paths["dir"] / "target_segment.npy")
    segmentation = np.load(paths["dir"] / "segmentation.npy")

    automatic_mask_path = paths["dir"] / "automatic_mask.npy"
    automatic_mask = np.load(automatic_mask_path).astype(bool)

    fitting_mask_fits_path = paths.get("fitting_mask")
    fitting_mask_npy_path = paths["dir"] / "fitting_mask.npy"
    if fitting_mask_fits_path is None or not fitting_mask_fits_path.is_file():
        raise RuntimeError(
            "Identity-era image-preparation bundle is incomplete: the "
            "canonical fitting mask is missing."
        )
    fitting_mask = load_external_mask(
        fitting_mask_fits_path,
        expected_shape=prepared_data.shape,
    )
    fitting_mask_npy = build_fitting_mask(
        np.load(fitting_mask_npy_path)
    )
    if not np.array_equal(fitting_mask, fitting_mask_npy):
        raise RuntimeError(
            "The persisted fitting-mask FITS and NPY artifacts disagree."
        )

    if fitting_mask.shape != prepared_data.shape:
        raise RuntimeError(
            "The saved fitting mask shape does not match the prepared image."
        )

    preliminary_cutout_path = paths["dir"] / "preliminary_cutout.npy"
    preliminary_cutout = (
        np.load(preliminary_cutout_path)
        if "preliminary_cutout.npy" in artifact_names
        else None
    )

    preliminary_segmentation_path = paths["dir"] / "preliminary_segmentation.npy"
    preliminary_segmentation = (
        np.load(preliminary_segmentation_path)
        if "preliminary_segmentation.npy" in artifact_names
        else None
    )

    original_image_path = paths["dir"] / "original_image.npy"
    original_image = (
        np.load(original_image_path)
        if "original_image.npy" in artifact_names
        else None
    )

    external_mask_path = paths["dir"] / "external_mask.npy"
    external_mask = (
        np.load(external_mask_path).astype(bool)
        if "external_mask.npy" in artifact_names
        else None
    )

    geometry_data = metadata.get("geometry")
    geometry = (
        TargetGeometry(
            x0=float(geometry_data["x0"]),
            y0=float(geometry_data["y0"]),
            q=float(geometry_data["q"]),
            theta=float(geometry_data["theta"]),
        )
        if geometry_data is not None
        else None
    )

    plateau_data = metadata.get("plateau")
    plateau = (
        PlateauResult(
            radius=float(plateau_data["radius"]),
            index=int(plateau_data["index"]),
            background_level=float(plateau_data["background_level"]),
            noise_sigma=float(plateau_data["noise_sigma"]),
            window_start=int(plateau_data["window_start"]),
            window_stop=int(plateau_data["window_stop"]),
        )
        if plateau_data is not None
        else None
    )

    radial_profile_path = paths["dir"] / "radial_profile.npz"
    if "radial_profile.npz" in artifact_names:
        with np.load(radial_profile_path) as profile_data:
            radial_profile = RadialProfile(
                radius=profile_data["radius"],
                profile=profile_data["profile"],
                n_pixels=profile_data["n_pixels"],
                scatter=profile_data["scatter"],
                area=profile_data["area"],
            )
    else:
        radial_profile = None

    sky_region_path = paths["dir"] / "sky_region.npz"
    if "sky_region.npz" in artifact_names and plateau is not None:
        with np.load(sky_region_path) as sky_data:
            sky_region = SkyRegion(
                mask=sky_data["mask"].astype(bool),
                radius_map=sky_data["radius_map"],
                plateau=plateau,
                inner_radius=float(sky_data["inner_radius"]),
                outer_radius=float(sky_data["outer_radius"]),
                n_pixels=int(sky_data["n_pixels"]),
                coverage=float(sky_data["coverage"]),
            )
    else:
        sky_region = None

    products = PreparedObservation(
        cutout=cutout,
        prepared_data=prepared_data,
        noise=noise,
        automatic_mask=automatic_mask,
        fitting_mask=fitting_mask,
        target_segment=target_segment,
        contaminant_mask=contaminant_mask,
        segmentation=segmentation,
        background=float(metadata["background"]),
        target_x=int(metadata["target_x"]),
        target_y=int(metadata["target_y"]),
        target_label=int(metadata["target_label"]),
        cutout_mode=metadata.get("cutout_mode", "fixed"),
        original_image=original_image,
        external_mask=external_mask,
        fitting_mask_mode=metadata.get(
            "fitting_mask_mode",
            "union",
        ),
        cutout_bounds=metadata.get("cutout_bounds"),
        geometry=geometry,
        radial_profile=radial_profile,
        plateau=plateau,
        sky_region=sky_region,
        preliminary_cutout=preliminary_cutout,
        preliminary_segmentation=preliminary_segmentation,
        sky_gradient=metadata.get("sky_gradient"),
    )
    _validate_products_for_persistence(products)
    _validate_products_against_current_sources(
        resolved,
        products,
    )
    return products
