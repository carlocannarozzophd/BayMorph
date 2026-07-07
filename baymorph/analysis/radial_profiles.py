from __future__ import annotations

import numpy as np


def radial_profile_from_image(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radii: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a median radial profile from a two-dimensional image.

    Parameters
    ----------
    image : ndarray
        Input image.
    center_x : float
        X-coordinate of the profile center, in pixel coordinates.
    center_y : float
        Y-coordinate of the profile center, in pixel coordinates.
    radii : ndarray
        Radial bin edges. Must be one-dimensional and contain at least two
        values.
    mask : ndarray of bool, optional
        Boolean mask with the same shape as ``image``. Masked pixels are
        excluded from the radial-profile calculation.

    Returns
    -------
    ndarray
        Median image value in each radial bin. Empty bins are returned as
        ``NaN``.

    Raises
    ------
    ValueError
        If ``image`` is not two-dimensional, ``radii`` is not one-dimensional,
        ``radii`` contains fewer than two edges, or ``mask`` has an invalid
        shape.
    """
    image = np.asarray(image, dtype=float)
    radii = np.asarray(radii, dtype=float)

    if image.ndim != 2:
        raise ValueError("image must be 2D.")

    if radii.ndim != 1:
        raise ValueError("radii must be 1D.")

    if radii.size < 2:
        raise ValueError("radii must contain at least two edges.")

    yy, xx = np.indices(image.shape)
    radius = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

    valid = np.isfinite(image)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

        if mask.shape != image.shape:
            raise ValueError("mask must have the same shape as image.")

        valid &= ~mask

    profile = np.full(radii.size - 1, np.nan)

    for i in range(radii.size - 1):
        annulus = (
            (radius >= radii[i])
            & (radius < radii[i + 1])
            & valid
        )

        if np.any(annulus):
            profile[i] = np.nanmedian(image[annulus])

    return profile