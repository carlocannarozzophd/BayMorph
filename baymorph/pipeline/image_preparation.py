from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from baymorph.core.image import ImageData
from baymorph.core.background import (
    estimate_background_photutils,
    estimate_background_sigma,
    subtract_background,
)
from baymorph.core.detection import (
    DetectionResult,
    detect_sources_segmentation,
)
from baymorph.core.noise import estimate_noise_map
from baymorph.pipeline.segmentation import select_segment_near_position
from .config import load_resolved_config
from .outputs import make_output_tree, pipeline_paths, save_resolved_config

@dataclass
class RealFITSPreparation:
    """Prepared real-image products before target-specific masking."""

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
):
    """Prepare a real image for fitting."""
    data = np.asarray(
        data,
        dtype=float,
    )

    detection = detect_sources_segmentation(
        data,
        n_sigma=detection_sigma,
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


def make_target_cutout(
    data: np.ndarray,
    target_x: int,
    target_y: int,
    cutout_size: int,
) -> tuple[np.ndarray, dict[str, int]]:
    half_size = cutout_size // 2

    x_min = int(target_x) - half_size
    x_max = int(target_x) + half_size
    y_min = int(target_y) - half_size
    y_max = int(target_y) + half_size

    if x_min < 0 or y_min < 0 or x_max > data.shape[1] or y_max > data.shape[0]:
        raise ValueError(
            "Requested cutout extends outside the image boundaries."
        )

    cutout = data[y_min:y_max, x_min:x_max]

    bounds = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }

    return cutout, bounds


def prepare_real_image_from_config(
    config: str | Path | dict[str, Any],
    *,
    save: bool = True,
) -> ImagePreparationProducts:
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = config

    make_output_tree(resolved)

    image = ImageData.from_fits(resolved["data"]["image_path"])

    target_position = resolved["data"]["target_position"]
    if target_position is None:
        raise ValueError("Config data.target_position is required for image preparation.")

    cutout_size = resolved["data"]["cutout_size"]
    if cutout_size is None:
        raise ValueError("Config data.cutout_size is required for image preparation.")

    cutout, bounds = make_target_cutout(
        image.data,
        target_x=target_position["x"],
        target_y=target_position["y"],
        cutout_size=cutout_size,
    )

    detection_cfg = resolved["preparation"]["detection"]
    background_cfg = resolved["preparation"]["background"]

    prepared = prepare_real_fits_image(
        cutout,
        background_box_size=background_cfg["box_size"],
        background_filter_size=background_cfg["filter_size"],
        detection_sigma=detection_cfg["n_sigma"],
    )

    segmentation = prepared.detection.segmentation
    if segmentation is None:
        raise RuntimeError("No segmentation map was produced.")

    target_x = int(target_position["x"]) - bounds["x_min"]
    target_y = int(target_position["y"]) - bounds["y_min"]

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
    )

    if save:
        save_image_preparation_products(resolved, products)
        save_resolved_config(resolved)

    return products


def save_image_preparation_products(
    config: dict[str, Any],
    products: ImagePreparationProducts,
) -> None:
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

    metadata = {
        "background": products.background,
        "target_x": products.target_x,
        "target_y": products.target_y,
        "target_label": products.target_label,
    }

    import json

    with open(paths["dir"] / "image_preparation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_image_preparation_products(
    config: str | Path | dict[str, Any],
) -> ImagePreparationProducts:
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

    import json

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
    )