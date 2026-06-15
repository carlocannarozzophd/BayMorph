"""Real FITS preparation pipeline."""

from dataclasses import dataclass

import numpy as np

from baymorph.core.background import (
    estimate_background_photutils,
    estimate_background_sigma,
    subtract_background,
)
from baymorph.core.masking import detect_sources_mask
from baymorph.core.noise import estimate_noise_map


@dataclass
class RealFITSPreparation:
    """Prepared image products for fitting."""

    data: np.ndarray
    background: float
    mask: np.ndarray
    noise: np.ndarray


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

    mask = detect_sources_mask(
        data,
        n_sigma=detection_sigma,
    )

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
    )