"""Image-preparation workflows for real BayMorph FITS images."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from baymorph.core.background import (
    estimate_background_photutils,
    estimate_background_sigma,
    subtract_background,
)
from baymorph.core.detection import (
    DetectionResult,
    detect_sources_segmentation,
)
from baymorph.core.image import ImageData
from baymorph.core.noise import estimate_noise_map
from baymorph.core.sky import (
    PlateauResult,
    RadialProfile,
    compute_elliptical_radial_profile,
    detect_profile_plateau,
)
from baymorph.pipeline.segmentation import select_segment_near_position

from .config import load_resolved_config
from .cutout import CutoutDefinition, resolve_cutout
from .outputs import (
    make_output_tree,
    pipeline_paths,
    save_resolved_config,
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


@dataclass(slots=True)
class TargetGeometry:
    """Initial geometric estimate of the target source.

    Parameters
    ----------
    x0, y0 : float
        Estimated target centre in image coordinates.
    q : float
        Estimated minor-to-major axis ratio.
    theta : float
        Position angle in radians, measured counter-clockwise from the
        positive x-axis.
    """

    x0: float
    y0: float
    q: float
    theta: float


def estimate_target_geometry_from_segment(
    target_segment: np.ndarray,
) -> TargetGeometry:
    """Estimate target geometry from a boolean segmentation mask.

    The source centre and shape are estimated from the unweighted second
    moments of the target-segment pixels.

    Parameters
    ----------
    target_segment : ndarray of bool
        Boolean mask identifying the target source.

    Returns
    -------
    TargetGeometry
        Initial centre, axis ratio, and position-angle estimates.

    Raises
    ------
    ValueError
        If the segment is not two-dimensional or contains fewer than three
        pixels.
    """
    segment = np.asarray(target_segment, dtype=bool)

    if segment.ndim != 2:
        raise ValueError("'target_segment' must be a two-dimensional array.")

    y, x = np.nonzero(segment)

    if x.size < 3:
        raise ValueError(
            "At least three target-segment pixels are required "
            "to estimate the source geometry."
        )

    x0 = float(np.mean(x))
    y0 = float(np.mean(y))

    dx = x.astype(float) - x0
    dy = y.astype(float) - y0

    covariance = np.array(
        [
            [np.mean(dx * dx), np.mean(dx * dy)],
            [np.mean(dx * dy), np.mean(dy * dy)],
        ],
        dtype=float,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    major_variance = float(eigenvalues[0])
    minor_variance = float(eigenvalues[1])

    if major_variance <= 0.0:
        q = 1.0
        theta = 0.0
    else:
        q = float(
            np.sqrt(
                max(minor_variance, 0.0) / major_variance
            )
        )
        q = float(np.clip(q, np.finfo(float).eps, 1.0))

        major_axis = eigenvectors[:, 0]
        theta = float(
            np.arctan2(
                major_axis[1],
                major_axis[0],
            )
        )

    return TargetGeometry(
        x0=x0,
        y0=y0,
        q=q,
        theta=theta,
    )


def build_target_radial_profile(
    image: np.ndarray,
    geometry: TargetGeometry,
    *,
    mask: np.ndarray | None = None,
    statistic: str = "median",
    min_pixels: int = 1,
) -> RadialProfile:
    """Build an elliptical radial profile from target geometry.

    Parameters
    ----------
    image : ndarray
        Input image.
    geometry : TargetGeometry
        Initial target geometry.
    mask : ndarray of bool, optional
        Pixels excluded from the radial profile.
    statistic : {"median", "mean", "sum"}, default="median"
        Statistic computed within each elliptical annulus.
    min_pixels : int, default=1
        Minimum number of valid pixels required in an annulus.

    Returns
    -------
    RadialProfile
        Elliptical radial profile of the target.
    """
    return compute_elliptical_radial_profile(
        image,
        x0=geometry.x0,
        y0=geometry.y0,
        q=geometry.q,
        theta=geometry.theta,
        mask=mask,
        statistic=statistic,
        min_pixels=min_pixels,
    )


def estimate_target_sky(
    image: np.ndarray,
    target_segment: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    statistic: str = "median",
    min_pixels: int = 1,
    min_radius: float = 0.0,
    window_size: int = 5,
    sigma_threshold: float = 1.0,
    outer_fraction: float = 0.25,
    min_coverage: float = 0.5,
) -> tuple[
    TargetGeometry,
    RadialProfile,
    PlateauResult,
]:
    """Estimate target geometry, radial profile, and sky plateau.

    Parameters
    ----------
    image : ndarray
        Input image.
    target_segment : ndarray of bool
        Boolean mask identifying the target source.
    mask : ndarray of bool, optional
        Pixels excluded from the radial profile.
    statistic : {"median", "mean", "sum"}, default="median"
        Statistic computed within each elliptical annulus.
    min_pixels : int, default=1
        Minimum number of valid pixels required in each annulus.
    min_radius : float, default=0.0
        Minimum radius allowed for plateau detection.
    window_size : int, default=5
        Number of consecutive radial bins required for a plateau.
    sigma_threshold : float, default=1.0
        Plateau tolerance in units of the outer-profile scatter.
    outer_fraction : float, default=0.25
        Fraction of outer valid bins used to estimate the sky level.
    min_coverage : float, default=0.5
        Minimum accepted annular pixel coverage.

    Returns
    -------
    geometry : TargetGeometry
        Initial target geometry.
    radial_profile : RadialProfile
        Elliptical radial profile.
    plateau : PlateauResult
        Detected sky plateau.
    """
    geometry = estimate_target_geometry_from_segment(
        target_segment,
    )

    radial_profile = build_target_radial_profile(
        image,
        geometry,
        mask=mask,
        statistic=statistic,
        min_pixels=min_pixels,
    )

    plateau = detect_profile_plateau(
        radial_profile,
        min_radius=min_radius,
        window_size=window_size,
        sigma_threshold=sigma_threshold,
        outer_fraction=outer_fraction,
        min_coverage=min_coverage,
    )

    return geometry, radial_profile, plateau


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
        Target position in cutout pixel coordinates.
    target_label : int
        Segmentation label assigned to the target source.
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

    cutout_definition = resolve_cutout(
        image_shape=image.data.shape,
        center_x=target_position["x"],
        center_y=target_position["y"],
        config=cutout_cfg,
    )

    cutout = make_target_cutout(
        image.data,
        cutout_definition,
    )

    detection_cfg = resolved["preparation"]["detection"]
    background_cfg = resolved["preparation"]["background"]

    detection_n_pixels = detection_cfg.get(
        "n_pixels",
        detection_cfg.get("npixels"),
    )

    prepared = prepare_real_fits_image(
        cutout,
        background_box_size=background_cfg["box_size"],
        background_filter_size=background_cfg["filter_size"],
        detection_sigma=detection_cfg["n_sigma"],
        detection_n_pixels=detection_n_pixels,
        detection_n_levels=detection_cfg["n_levels"],
    )

    segmentation = prepared.detection.segmentation
    if segmentation is None:
        raise RuntimeError("No segmentation map was produced.")

    target_x = int(target_position["x"]) - cutout_definition.xmin
    target_y = int(target_position["y"]) - cutout_definition.ymin

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

    products = ImagePreparationProducts(
        cutout=cutout,
        prepared_data=prepared.data,
        noise=prepared.noise,
        automatic_mask=prepared.mask,
        target_segment=target_segment,
        contaminant_mask=contaminant_mask,
        segmentation=segmentation.data,
        background=float(prepared.background),
        target_x=target_x,
        target_y=target_y,
        target_label=int(target_label),
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