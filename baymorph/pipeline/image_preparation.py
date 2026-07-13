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
    estimate_background_sigma,
    subtract_background,
)
from baymorph.core.detection import (
    DetectionResult,
    detect_sources_segmentation,
)
from baymorph.core.image import ImageData
from baymorph.core.noise import (
    estimate_noise_map,
    make_constant_noise_map,
)
from baymorph.core.sky import (
    PlateauResult,
    RadialProfile,
    SkyRegion,
    build_sky_region,
)
from baymorph.pipeline.segmentation import (
    select_segment_near_position,
)

from .preparation_steps import (
    TargetGeometry,
)

from .config import load_resolved_config
from .cutout import (
    CutoutDefinition,
    resolve_cutout,
    resolve_plateau_cutout,
    resolve_preliminary_auto_cutout,
)
from .outputs import (
    make_output_tree,
    pipeline_paths,
    save_resolved_config,
)
from .preparation_steps import estimate_target_sky


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
        background_result = estimate_background_sigma(
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
    detection_n_pixels = detection_cfg.get(
        "n_pixels",
        detection_cfg.get("npixels"),
    )

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
        raise RuntimeError(
            f"No segmentation map was produced during {context}."
        )

    target_label = select_segment_near_position(
        segmentation,
        x=target_x,
        y=target_y,
    )

    target_segment = segmentation.data == target_label

    contaminant_mask = (
        (segmentation.data > 0)
        & (segmentation.data != target_label)
    )

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
) -> CutoutDefinition:
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
        initial_factor=float(
            cutout_cfg.get("initial_factor", 4.0)
        ),
        min_size=int(
            cutout_cfg.get("min_size", 100)
        ),
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

    global_center_x = (
        preliminary_definition.xmin
        + geometry.x0
    )
    global_center_y = (
        preliminary_definition.ymin
        + geometry.y0
    )

    return resolve_plateau_cutout(
        image_shape=image.shape,
        center_x=global_center_x,
        center_y=global_center_y,
        plateau_radius=plateau.radius,
        sky_margin=float(
            cutout_cfg.get("sky_margin", 20.0)
        ),
        min_size=cutout_cfg.get("min_size"),
        max_size=cutout_cfg.get("max_size"),
    )


@dataclass
class ImagePreparationProducts:
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
    target_segment : ndarray of bool
        Boolean mask identifying the target segment.
    contaminant_mask : ndarray of bool
        Boolean mask identifying non-target detected sources.
    segmentation : ndarray
        Integer segmentation map.
    background : float
        Estimated scalar background level.
    target_x, target_y : int
        Measured target centre in cutout pixel coordinates.
    target_label : int
        Segmentation label assigned to the target source.
    geometry : TargetGeometry, optional
        Measured target geometry.
    radial_profile : RadialProfile, optional
        Elliptical radial profile.
    plateau : PlateauResult, optional
        Plateau detection result.
    sky_region : SkyRegion, optional
        Final sky region used for background estimation.
    preliminary_cutout : ndarray, optional
        Preliminary adaptive cutout.
    preliminary_segmentation : ndarray, optional
        Segmentation map measured on the preliminary cutout.
    """

    cutout: np.ndarray
    prepared_data: np.ndarray
    noise: np.ndarray
    automatic_mask: np.ndarray
    target_segment: np.ndarray
    contaminant_mask: np.ndarray
    segmentation: np.ndarray
    background: float
    target_x: int
    target_y: int
    target_label: int

    original_image: np.ndarray | None = None
    cutout_bounds: dict[str, int] | None = None

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
        cutout.ymin:cutout.ymax,
        cutout.xmin:cutout.xmax,
    ]


def prepare_real_image_from_config(
    config: str | Path | dict[str, Any],
    *,
    save: bool = True,
) -> ImagePreparationProducts:
    """Prepare a real galaxy image from a BayMorph configuration.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Resolved configuration dictionary or path to a YAML configuration file.
    save : bool, default=True
        If True, save image-preparation products to the configured output tree.

    Returns
    -------
    ImagePreparationProducts
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

    image = ImageData.from_fits(resolved["data"]["image_path"])

    target_position = resolved["data"]["target_position"]
    if target_position is None:
        raise ValueError(
            "Config data.target_position is required for image preparation."
        )

    cutout_cfg = resolved["data"]["cutout"]
    detection_cfg = resolved["preparation"]["detection"]
    background_cfg = resolved["preparation"]["background"]

    cutout_mode = cutout_cfg.get("mode", "fixed")

    if cutout_mode == "fixed":
        cutout_definition = resolve_cutout(
            image_shape=image.data.shape,
            center_x=target_position["x"],
            center_y=target_position["y"],
            config=cutout_cfg,
        )

    elif cutout_mode == "auto":
        cutout_definition = _resolve_auto_cutout(
            image.data,
            target_x=target_position["x"],
            target_y=target_position["y"],
            cutout_cfg=cutout_cfg,
            detection_cfg=detection_cfg,
            background_cfg=background_cfg,
        )

    else:
        raise ValueError(
            f"Unsupported cutout mode: {cutout_mode!r}."
        )

    cutout = make_target_cutout(
        image.data,
        cutout_definition,
    )

    seed_target_x = (
        float(target_position["x"])
        - cutout_definition.xmin
    )
    seed_target_y = (
        float(target_position["y"])
        - cutout_definition.ymin
    )

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

    geometry, radial_profile, plateau = estimate_target_sky(
        cutout,
        target_segment,
        mask=contaminant_mask,
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
        mask=contaminant_mask,
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

    products = ImagePreparationProducts(
        cutout=cutout,
        prepared_data=prepared_data,
        noise=noise,
        automatic_mask=prepared.mask,
        target_segment=target_segment,
        contaminant_mask=contaminant_mask,
        segmentation=segmentation.data,
        background=float(background_result.background_level),
        target_x=target_x,
        target_y=target_y,
        target_label=int(target_label),

        geometry=geometry,
        radial_profile=radial_profile,
        plateau=plateau,
        sky_region=sky_region,

        preliminary_cutout=None,
        preliminary_segmentation=None,

        original_image=image.data,
        cutout_bounds={
            "xmin": int(cutout_definition.xmin),
            "xmax": int(cutout_definition.xmax),
            "ymin": int(cutout_definition.ymin),
            "ymax": int(cutout_definition.ymax),
        },
    )

    if save:
        save_image_preparation_products(resolved, products)
        save_resolved_config(resolved)

    return products


def save_image_preparation_products(
    config: dict[str, Any],
    products: ImagePreparationProducts,
) -> None:
    """Save image-preparation products to disk.

    Parameters
    ----------
    config : dict
        Resolved pipeline configuration.
    products : ImagePreparationProducts
        Image-preparation products to save.

    Returns
    -------
    None
        Products are written to the configured output directory.
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

    if products.original_image is not None:
        np.save(paths["dir"] / "original_image.npy", products.original_image)

    metadata = {
        "background": products.background,
        "target_x": products.target_x,
        "target_y": products.target_y,
        "target_label": products.target_label,
        "cutout_bounds": products.cutout_bounds,
    }

    with open(paths["dir"] / "image_preparation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_image_preparation_products(
    config: str | Path | dict[str, Any],
) -> ImagePreparationProducts:
    """Load saved image-preparation products from disk.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Resolved configuration dictionary or path to a YAML configuration file.

    Returns
    -------
    ImagePreparationProducts
        Image-preparation products reconstructed from saved files.
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

    original_image_path = paths["dir"] / "original_image.npy"
    original_image = (
        np.load(original_image_path)
        if original_image_path.exists()
        else None
    )

    with open(paths["dir"] / "image_preparation_metadata.json", "r") as f:
        metadata = json.load(f)

    return ImagePreparationProducts(
        cutout=cutout,
        prepared_data=prepared_data,
        noise=noise,
        automatic_mask=contaminant_mask,
        target_segment=target_segment,
        contaminant_mask=contaminant_mask,
        segmentation=segmentation,
        background=float(metadata["background"]),
        target_x=int(metadata["target_x"]),
        target_y=int(metadata["target_y"]),
        target_label=int(metadata["target_label"]),
        original_image=original_image,
        cutout_bounds=metadata.get("cutout_bounds"),
    )