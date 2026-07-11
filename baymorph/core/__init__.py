"""Core data structures and utilities for BayMorph."""

from .background import (
    BackgroundResult,
    estimate_background_photutils,
    estimate_background_sigma,
    subtract_background,
)
from .detection import (
    DetectionResult,
    detect_sources_mask,
    detect_sources_segmentation,
)
from .image import ImageData
from .mask import (
    connected_source_mask,
    star_halo_mask,
)
from .noise import (
    estimate_noise_map,
    make_constant_noise_map,
)
from .sky import build_elliptical_radius_map

__all__ = [
    "ImageData",
    "BackgroundResult",
    "DetectionResult",
    "estimate_background_photutils",
    "estimate_background_sigma",
    "subtract_background",
    "make_constant_noise_map",
    "estimate_noise_map",
    "estimate_noise",
    "detect_sources_mask",
    "detect_sources_segmentation",
    "connected_source_mask",
    "star_halo_mask",
    "build_elliptical_radius_map",
]