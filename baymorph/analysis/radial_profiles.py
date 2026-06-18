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