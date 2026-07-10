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


def radial_bins(
    max_radius,
    *,
    mode="fixed",
    dr=1.0,
    min_pixels=50,
    log_n_bins=40,
):
    """Construct radial-bin edges for radial-profile measurements.

    Parameters
    ----------
    max_radius : float
        Maximum radius of the profile, in pixels.

    mode : {"fixed", "adaptive", "logarithmic"}, default="fixed"
        Strategy used to construct the radial bins.

        ``"fixed"``
            Uniform radial bins with constant width ``dr``.

        ``"adaptive"``
            Variable-width bins chosen to contain at least approximately
            ``min_pixels`` pixels per annulus while never becoming narrower
            than ``dr``.

        ``"logarithmic"``
            Logarithmically spaced bins between ``dr`` and
            ``max_radius``.

    dr : float, default=1.0
        Minimum radial-bin width, in pixels. For fixed binning this is the
        constant annulus width. For adaptive and logarithmic binning it
        defines the smallest allowed radial spacing.

    min_pixels : int, default=50
        Approximate minimum number of pixels per annulus when using
        adaptive binning.

    log_n_bins : int, default=40
        Number of logarithmically spaced bins when ``mode="logarithmic"``.

    Returns
    -------
    ndarray
        One-dimensional array containing the radial-bin edges, in pixels.
        The returned array always starts at zero and ends at
        ``max_radius``.

    Raises
    ------
    ValueError
        If an unsupported binning mode is requested.

    Notes
    -----
    The returned bin edges are intended for use with
    :func:`radial_profile_from_image` and the posterior-profile analysis
    routines. Bin centres can be obtained with

    ``0.5 * (radii[:-1] + radii[1:])``.
    """
    max_radius = float(max_radius)

    if mode == "fixed":
        edges = np.arange(
            0.0,
            max_radius,
            float(dr),
            dtype=float,
        )

        return np.unique(
            np.concatenate(
                (
                    edges,
                    [max_radius],
                )
            )
        )

    if mode == "logarithmic":
        inner = max(float(dr), 1.0)

        bins = np.geomspace(
            inner,
            max_radius,
            int(log_n_bins),
        )

        return np.unique(
            np.concatenate(
                ([0.0], bins)
            )
        )

    if mode == "adaptive":
        edges = [0.0]
        r = 0.0

        while r < max_radius:
            dr_current = max(
                float(dr),
                float(min_pixels) / max(2.0 * np.pi * max(r, 1.0), 1.0),
            )

            r = min(
                r + dr_current,
                max_radius,
            )

            edges.append(r)

        return np.asarray(
            edges,
            dtype=float,
        )

    raise ValueError(
        "Unknown radial binning mode "
        f"'{mode}'. Expected one of "
        "'fixed', 'adaptive', or 'logarithmic'."
    )