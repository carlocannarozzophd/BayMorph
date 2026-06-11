"""Core data structures and utilities for BayMorph."""

from .masking import (
    detect_sources_mask,
)
from .background import (
    BackgroundEstimate,
    estimate_background_sigma,
    estimate_background_photutils,
    subtract_background,
)
from .image import ImageData
from .noise import (
    estimate_noise_map,
    make_constant_noise_map,
)

__all__ = [
    "ImageData",
    "BackgroundEstimate",
    "estimate_background_sigma",
    "estimate_background_photutils",
    "subtract_background",
    "make_constant_noise_map",
    "estimate_noise_map",
    "detect_sources_mask",
]