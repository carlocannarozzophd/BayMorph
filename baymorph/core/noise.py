"""Noise-map utilities."""

import numpy as np


def make_constant_noise_map(
    shape,
    sigma: float,
) -> np.ndarray:
    """Create a constant noise map."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    return np.full(
        shape,
        sigma,
        dtype=float,
    )


def estimate_noise_map(
    data,
    sigma=None,
    mask=None,
) -> np.ndarray:
    """Estimate a constant noise map.

    If sigma is provided, return a constant map with that value.
    Otherwise estimate sigma empirically from unmasked pixels using MAD.
    """
    data = np.asarray(
        data,
        dtype=float,
    )

    if sigma is not None:
        if sigma <= 0:
            raise ValueError("sigma must be positive.")

        return make_constant_noise_map(
            data.shape,
            sigma=sigma,
        )

    if mask is None:
        values = data[np.isfinite(data)]
    else:
        mask = np.asarray(
            mask,
            dtype=bool,
        )

        values = data[
            (~mask)
            & np.isfinite(data)
        ]

    if values.size == 0:
        raise ValueError(
            "No valid pixels available to estimate noise."
        )

    median = np.nanmedian(values)

    mad = np.nanmedian(
        np.abs(values - median)
    )

    estimated_sigma = 1.4826 * mad

    if (
        not np.isfinite(estimated_sigma)
        or estimated_sigma <= 0
    ):
        raise ValueError(
            "Estimated noise sigma must be positive."
        )

    return make_constant_noise_map(
        data.shape,
        sigma=estimated_sigma,
    )