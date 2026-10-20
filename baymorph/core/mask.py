"""Mask-construction utilities for astronomical images."""

from functools import reduce
from pathlib import Path

import numpy as np
from astropy.io import fits
from photutils.segmentation import detect_sources, detect_threshold
from scipy.ndimage import binary_dilation

from baymorph.core.image import WorkingRegion


def load_external_mask(
    filename: str | Path,
    *,
    expected_shape: tuple[int, int] | None = None,
    region: WorkingRegion | None = None,
) -> np.ndarray:
    """Load and validate an external exclusion mask from a FITS file.

    Non-zero input pixels are interpreted as masked pixels. The returned
    array always has boolean dtype. The mask is read from the primary HDU,
    or from the first extension when the primary HDU holds no data.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the FITS mask file.
    expected_shape : tuple of int, optional
        Required mask shape as ``(ny, nx)``. If provided, a shape mismatch
        raises ``ValueError``.
    region : WorkingRegion, optional
        Working region of the science image (AD-053). Only that part of the
        mask is read and returned, after the shape of the whole mask is
        checked against ``expected_shape``.

    Returns
    -------
    ndarray of bool
        Two-dimensional boolean exclusion mask, of the working region when
        one is given.

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

    if region is None:
        data = np.asarray(fits.getdata(path))
        shape = data.shape
    else:
        with fits.open(path) as hdul:
            hdu = hdul[0] if hdul[0].header.get("NAXIS", 0) else hdul[1]
            header = hdu.header
            shape = tuple(
                int(header.get(f"NAXIS{axis}", 0))
                for axis in range(int(header.get("NAXIS", 0)), 0, -1)
            )
            data = None

            if len(shape) == 2 and shape == region.image_shape:
                data = np.asarray(
                    hdu.section[region.ymin : region.ymax, region.xmin : region.xmax]
                )

    if len(shape) != 2:
        raise ValueError(
            "External mask must be a two-dimensional FITS image; "
            f"received shape {shape}."
        )

    if data is not None and not np.all(np.isfinite(data)):
        raise ValueError("External mask contains non-finite pixel values.")

    if expected_shape is not None and shape != tuple(expected_shape):
        raise ValueError(
            "External mask shape does not match the science image: "
            f"mask shape {shape}, image shape {tuple(expected_shape)}."
        )

    if data is None:
        raise ValueError(
            f"External mask of shape {shape} does not cover the working "
            f"region of an image of shape {region.image_shape}."
        )

    return data != 0


def _as_boolean_mask(mask, *, name: str = "mask") -> np.ndarray:
    """Return a validated two-dimensional boolean mask."""

    array = np.asarray(mask)

    if array.ndim != 2:
        raise ValueError(f"'{name}' must be a two-dimensional array.")

    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise ValueError(
            f"'{name}' must contain finite numeric or boolean values."
        ) from exc

    if not np.all(finite):
        raise ValueError(f"'{name}' contains non-finite pixel values.")

    return np.array(array, dtype=bool, copy=True)


def _validate_mask_collection(*masks) -> list[np.ndarray]:
    """Validate a collection of masks."""

    if len(masks) < 2:
        raise ValueError("At least two masks are required.")

    validated = [
        _as_boolean_mask(mask, name=f"masks[{index}]")
        for index, mask in enumerate(masks)
    ]

    reference_shape = validated[0].shape

    for mask in validated[1:]:
        if mask.shape != reference_shape:
            raise ValueError("All masks must have the same shape.")

    return validated


def union(*masks) -> np.ndarray:
    """Return the logical union of two or more masks."""
    validated = _validate_mask_collection(*masks)

    return reduce(np.logical_or, validated)


def intersection(*masks) -> np.ndarray:
    """Return the logical intersection of two or more masks."""
    validated = _validate_mask_collection(*masks)

    return reduce(np.logical_and, validated)


def build_fitting_mask(automatic_mask, *, external_mask=None, mode: str = "union"):
    """Construct a union mask or use a curated mask verbatim."""

    mask = _as_boolean_mask(automatic_mask, name="automatic_mask")

    resolved_mode = str(mode).strip().lower()

    if resolved_mode not in {"union", "exact", "replace"}:
        raise ValueError("'mode' must be one of 'union', 'exact', or 'replace'.")

    if external_mask is None:
        if resolved_mode in {"exact", "replace"}:
            raise ValueError(
                "An external mask is required when fitting-mask mode is "
                f"{resolved_mode!r}."
            )
        return mask

    curated = _as_boolean_mask(external_mask, name="external_mask")

    if curated.shape != mask.shape:
        raise ValueError(
            "The external mask must have the same shape as the automatic " "mask."
        )

    if resolved_mode in {"exact", "replace"}:
        return curated

    return union(mask, curated)


def negative_outlier_mask(data, noise, *, threshold_sigma: float) -> np.ndarray:
    """Flag pixels that are impossibly negative for background-subtracted data.

    Source detection thresholds positive excursions, so a one-sided criterion
    cannot exclude a negative detector artifact such as a saturation trail.
    After background subtraction no physical surface brightness is negative,
    so a pixel far below zero is an artifact by construction.

    The criterion is deliberately not symmetric. A large positive pixel may be
    a bright source; a large negative one cannot be.

    It reads the data alone and never a fitted model, so it can be declared
    before any inference. Excluding pixels by their residual under a model
    would instead make the mask data-adaptive, and evidence obtained after
    such a selection would be conditional on the selection procedure.

    Parameters
    ----------
    data : array-like
        Background-subtracted science image.
    noise : array-like
        Marginal noise scale, either a scalar or a map matching ``data``.
    threshold_sigma : float
        Number of noise scales below zero at which a pixel is excluded.
        Must be finite and positive.

    Returns
    -------
    ndarray of bool
        Exclusion mask following the ``True = excluded`` convention.
    """
    values = np.asarray(data, dtype=float)

    if values.ndim != 2:
        raise ValueError("'data' must be a two-dimensional array.")

    if not np.all(np.isfinite(values)):
        raise ValueError("'data' contains non-finite pixel values.")

    scale = np.asarray(noise, dtype=float)

    if scale.ndim not in (0, 2):
        raise ValueError("'noise' must be a scalar or a two-dimensional array.")

    if scale.ndim == 2 and scale.shape != values.shape:
        raise ValueError("'noise' must have the same shape as 'data'.")

    if not np.all(np.isfinite(scale)):
        raise ValueError("'noise' contains non-finite values.")

    if np.any(scale <= 0.0):
        raise ValueError("'noise' must be strictly positive.")

    threshold = float(threshold_sigma)

    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("'threshold_sigma' must be finite and positive.")

    return values < -threshold * scale


def connected_source_mask(
    image, x, y, threshold_sigma=2.0, n_pixels=5, dilation_iterations=0, box_size=101
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

    threshold = detect_threshold(local, n_sigma=threshold_sigma)

    segmentation = detect_sources(local, threshold, n_pixels=n_pixels)

    if segmentation is None:
        return np.zeros_like(image, dtype=bool)

    x_local = int(round(x)) - x0
    y_local = int(round(y)) - y0

    label = segmentation.data[y_local, x_local]

    if label == 0:
        return np.zeros_like(image, dtype=bool)

    local_mask = segmentation.data == label

    if dilation_iterations > 0:
        local_mask = binary_dilation(local_mask, iterations=dilation_iterations)

    mask = np.zeros_like(image, dtype=bool)

    mask[y0:y1, x0:x1] = local_mask

    return mask


def star_halo_mask(
    image, x, y, radius=50, threshold_sigma=2.0, n_pixels=5, dilation_iterations=10
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

    yy, xx = np.indices(image.shape)

    circular_mask = ((xx - x) ** 2 + (yy - y) ** 2) <= radius**2

    mask = source_mask | circular_mask

    if dilation_iterations > 0:
        mask = binary_dilation(mask, iterations=dilation_iterations)

    return mask
