"""Mask-construction utilities for astronomical images."""

from functools import reduce
from pathlib import Path

import numpy as np
from astropy.io import fits
from photutils.segmentation import (
    detect_sources,
    detect_threshold,
)
from scipy.ndimage import binary_dilation


def load_external_mask(
    filename: str | Path,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load and validate an external exclusion mask from a FITS file.

    Non-zero input pixels are interpreted as masked pixels. The returned
    array always has boolean dtype.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the FITS mask file.
    expected_shape : tuple of int, optional
        Required mask shape as ``(ny, nx)``. If provided, a shape mismatch
        raises ``ValueError``.

    Returns
    -------
    ndarray of bool
        Two-dimensional boolean exclusion mask.

    Raises
    ------
    FileNotFoundError
        If the mask file does not exist.
    ValueError
        If the FITS primary HDU does not contain a finite two-dimensional
        array or if its shape differs from ``expected_shape``.
    """
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(f"External mask file not found: {path}")

    data = np.asarray(
        fits.getdata(path),
    )

    if data.ndim != 2:
        raise ValueError(
            "External mask must be a two-dimensional FITS image; "
            f"received shape {data.shape}."
        )

    if not np.all(np.isfinite(data)):
        raise ValueError("External mask contains non-finite pixel values.")

    if expected_shape is not None and data.shape != expected_shape:
        raise ValueError(
            "External mask shape does not match the science image: "
            f"mask shape {data.shape}, image shape {expected_shape}."
        )

    return data != 0


def build_automatic_mask(
    segmentation,
    target_label: int,
) -> np.ndarray:
    """Build an exclusion mask from a segmentation image.

    Pixels belonging to detected sources other than the selected target are
    masked. Background pixels and pixels belonging to the target remain
    unmasked.

    Parameters
    ----------
    segmentation : photutils.segmentation.SegmentationImage or array-like
        Segmentation image whose background pixels have label zero.
    target_label : int
        Positive segmentation label identifying the target source.

    Returns
    -------
    ndarray of bool
        Boolean exclusion mask for all detected non-target sources.

    Raises
    ------
    ValueError
        If the segmentation is not two-dimensional, if ``target_label`` is
        not positive, or if the requested target label is absent.
    """
    segmentation_data = np.asarray(
        getattr(segmentation, "data", segmentation),
    )

    if segmentation_data.ndim != 2:
        raise ValueError("Segmentation data must be a two-dimensional array.")

    target_label = int(target_label)

    if target_label <= 0:
        raise ValueError("'target_label' must be a positive segmentation label.")

    if not np.any(segmentation_data == target_label):
        raise ValueError(
            f"Target label {target_label} is absent from the segmentation."
        )

    return (segmentation_data > 0) & (segmentation_data != target_label)


def _as_boolean_mask(
    mask,
    *,
    name: str = "mask",
) -> np.ndarray:
    """Return a validated two-dimensional boolean mask."""

    array = np.asarray(mask)

    if array.ndim != 2:
        raise ValueError(f"'{name}' must be a two-dimensional array.")

    return np.array(
        array,
        dtype=bool,
        copy=True,
    )


def _validate_mask_collection(*masks) -> list[np.ndarray]:
    """Validate a collection of masks."""

    if len(masks) < 2:
        raise ValueError("At least two masks are required.")

    validated = [
        _as_boolean_mask(
            mask,
            name=f"masks[{index}]",
        )
        for index, mask in enumerate(masks)
    ]

    reference_shape = validated[0].shape

    for mask in validated[1:]:
        if mask.shape != reference_shape:
            raise ValueError("All masks must have the same shape.")

    return validated


def union(*masks) -> np.ndarray:
    """Return the logical union of two or more masks."""
    validated = _validate_mask_collection(
        *masks,
    )

    return reduce(
        np.logical_or,
        validated,
    )


def intersection(*masks) -> np.ndarray:
    """Return the logical intersection of two or more masks."""
    if len(masks) < 2:
        raise ValueError("At least two masks are required.")

    validated = [
        _as_boolean_mask(
            mask,
            name=f"masks[{index}]",
        )
        for index, mask in enumerate(masks)
    ]

    shape = validated[0].shape

    for mask in validated[1:]:
        if mask.shape != shape:
            raise ValueError("All masks must have the same shape.")

    result = validated[0]

    for mask in validated[1:]:
        result &= mask

    return result


def connected_source_mask(
    image,
    x,
    y,
    threshold_sigma=2.0,
    n_pixels=5,
    dilation_iterations=0,
    box_size=101,
):
    """Create a mask for the connected source containing a seed position.

    Parameters
    ----------
    image : array-like
        Input image.
    x : float
        X-coordinate of the seed position, in pixel coordinates.
    y : float
        Y-coordinate of the seed position, in pixel coordinates.
    threshold_sigma : float, default=2.0
        Detection threshold in units of the estimated local noise.
    n_pixels : int, default=5
        Minimum number of connected pixels required for a detection.
    dilation_iterations : int, default=0
        Number of binary-dilation iterations applied to the detected source
        mask. If zero, no dilation is applied.
    box_size : int, default=101
        Size of the local square window used for source detection.

    Returns
    -------
    ndarray of bool
        Boolean mask for the connected detected structure containing the seed
        position. If no valid connected source is found, the returned mask is
        empty.
    """
    ny, nx = image.shape

    half = box_size // 2

    x0 = max(0, int(round(x)) - half)
    x1 = min(nx, int(round(x)) + half + 1)

    y0 = max(0, int(round(y)) - half)
    y1 = min(ny, int(round(y)) + half + 1)

    local = image[y0:y1, x0:x1]

    threshold = detect_threshold(
        local,
        n_sigma=threshold_sigma,
    )

    segmentation = detect_sources(
        local,
        threshold,
        n_pixels=n_pixels,
    )

    if segmentation is None:
        return np.zeros_like(image, dtype=bool)

    x_local = int(round(x)) - x0
    y_local = int(round(y)) - y0

    label = segmentation.data[
        y_local,
        x_local,
    ]

    if label == 0:
        return np.zeros_like(image, dtype=bool)

    local_mask = segmentation.data == label

    if dilation_iterations > 0:
        local_mask = binary_dilation(
            local_mask,
            iterations=dilation_iterations,
        )

    mask = np.zeros_like(
        image,
        dtype=bool,
    )

    mask[
        y0:y1,
        x0:x1,
    ] = local_mask

    return mask


def star_halo_mask(
    image,
    x,
    y,
    radius=50,
    threshold_sigma=2.0,
    n_pixels=5,
    dilation_iterations=10,
):
    """Create an aggressive mask around a bright star.

    The final mask combines the connected source containing the seed position,
    a circular halo mask centered on the same position, and optional binary
    dilation.

    Parameters
    ----------
    image : array-like
        Input image.
    x : float
        X-coordinate of the star center, in pixel coordinates.
    y : float
        Y-coordinate of the star center, in pixel coordinates.
    radius : float, default=50
        Radius of the circular halo mask, in pixels.
    threshold_sigma : float, default=2.0
        Detection threshold in units of the estimated local noise.
    n_pixels : int, default=5
        Minimum number of connected pixels required for a detection.
    dilation_iterations : int, default=10
        Number of binary-dilation iterations applied to the combined mask. If
        zero, no final dilation is applied.

    Returns
    -------
    ndarray of bool
        Boolean mask covering the connected stellar source and its surrounding
        halo.
    """
    source_mask = connected_source_mask(
        image=image,
        x=x,
        y=y,
        threshold_sigma=threshold_sigma,
        n_pixels=n_pixels,
        dilation_iterations=0,
    )

    yy, xx = np.indices(
        image.shape,
    )

    circular_mask = ((xx - x) ** 2 + (yy - y) ** 2) <= radius**2

    mask = source_mask | circular_mask

    if dilation_iterations > 0:
        mask = binary_dilation(
            mask,
            iterations=dilation_iterations,
        )

    return mask
