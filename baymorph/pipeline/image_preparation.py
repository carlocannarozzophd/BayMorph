"""Image-preparation workflows for real BayMorph FITS images."""

from __future__ import annotations

import json
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
)
from baymorph.core.observation import Observation
from baymorph.core.sky import (
    PlateauResult,
    RadialProfile,
    SkyRegion,
    build_sky_region,
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
from baymorph.pipeline.runtime_report import (
    build_image_preparation_runtime,
    build_runtime_report,
    save_runtime_report,
)
from baymorph.pipeline.segmentation import (
    select_segment_near_position,
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
    data = np.asarray(
        data,
        dtype=float,
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

    preliminary_definition = resolve_preliminary_auto_cutout(
        image_shape=image.shape,
        target_segment=full_detection.target_segment,
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

    geometry, radial_profile, plateau = estimate_target_sky(
        preliminary_cutout,
        preliminary_detection.target_segment,
        mask=preliminary_detection.contaminant_mask,
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
        Boolean mask identifying the target segment.
    contaminant_mask : ndarray of bool
        Final exclusion mask combining non-target detected sources with the
        optional external mask.
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

    geometry: TargetGeometry | None = None
    radial_profile: RadialProfile | None = None
    plateau: PlateauResult | None = None
    sky_region: SkyRegion | None = None

    preliminary_cutout: np.ndarray | None = None
    preliminary_segmentation: np.ndarray | None = None


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

    image = observation.image

    target_position = resolved["data"]["target_position"]
    if target_position is None:
        raise ValueError(
            "Config data.target_position is required for image preparation."
        )

    cutout_cfg = resolved["data"]["cutout"]
    detection_cfg = resolved["preparation"]["detection"]
    background_cfg = resolved["preparation"]["background"]

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
        )
        cutout_definition = auto_cutout_result.cutout

    else:
        raise ValueError(f"Unsupported cutout mode: {cutout_mode!r}.")

    cutout = make_target_cutout(
        image.data,
        cutout_definition,
    )

    external_mask = None
    external_mask_path = (
        resolved["data"].get("mask_path")
        or resolved["data"].get("external_mask_path")
    )

    if external_mask_path is not None:
        full_external_mask = load_external_mask(
            external_mask_path,
            expected_shape=image.data.shape,
        )
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
        noise_method="mad",
    )

    prepared_data = subtract_background(
        cutout,
        background_result.background_level,
    )

    noise = make_constant_noise_map(
        prepared_data.shape,
        background_result.sigma,
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
    paths = pipeline_paths(config)["image_preparation"]
    overwrite = bool(config["outputs"].get("overwrite", False))

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

    metadata = {
        "observation": (
            observation_metadata_dict(observation) if observation is not None else None
        ),
        "background": products.background,
        "target_x": products.target_x,
        "target_y": products.target_y,
        "target_label": products.target_label,
        "cutout_mode": products.cutout_mode,
        "cutout_bounds": products.cutout_bounds,
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
    }

    with open(paths["dir"] / "image_preparation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


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
    Outputs produced by older BayMorph versions remain supported. Optional
    diagnostic products that are not present on disk are returned as None.
    If the separately saved automatic mask is unavailable, the contaminant
    mask is used as a backward-compatible fallback.
    """
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = config

    paths = pipeline_paths(resolved)["image_preparation"]

    prepared_data = fits.getdata(paths["prepared_image"])
    noise = fits.getdata(paths["noise_map"])
    contaminant_mask = fits.getdata(paths["mask"]).astype(bool)

    cutout = np.load(paths["dir"] / "large_cutout.npy")
    target_segment = np.load(paths["dir"] / "target_segment.npy")
    segmentation = np.load(paths["dir"] / "segmentation.npy")

    automatic_mask_path = paths["dir"] / "automatic_mask.npy"
    automatic_mask = (
        np.load(automatic_mask_path).astype(bool)
        if automatic_mask_path.exists()
        else contaminant_mask.copy()
    )

    fitting_mask_path = paths["dir"] / "fitting_mask.npy"

    fitting_mask = (
        np.load(fitting_mask_path).astype(bool)
        if fitting_mask_path.exists()
        else contaminant_mask.copy()
    )

    preliminary_cutout_path = paths["dir"] / "preliminary_cutout.npy"
    preliminary_cutout = (
        np.load(preliminary_cutout_path) if preliminary_cutout_path.exists() else None
    )

    preliminary_segmentation_path = paths["dir"] / "preliminary_segmentation.npy"
    preliminary_segmentation = (
        np.load(preliminary_segmentation_path)
        if preliminary_segmentation_path.exists()
        else None
    )

    original_image_path = paths["dir"] / "original_image.npy"
    original_image = (
        np.load(original_image_path) if original_image_path.exists() else None
    )

    external_mask_path = paths["dir"] / "external_mask.npy"
    external_mask = (
        np.load(external_mask_path).astype(bool)
        if external_mask_path.exists()
        else None
    )

    with open(paths["dir"] / "image_preparation_metadata.json", "r") as f:
        metadata = json.load(f)

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
    if radial_profile_path.exists():
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
    if sky_region_path.exists() and plateau is not None:
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

    return PreparedObservation(
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
        cutout_bounds=metadata.get("cutout_bounds"),
        geometry=geometry,
        radial_profile=radial_profile,
        plateau=plateau,
        sky_region=sky_region,
        preliminary_cutout=preliminary_cutout,
        preliminary_segmentation=preliminary_segmentation,
    )
