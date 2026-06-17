"""Core data structures and utilities for BayMorph."""

from .background import (
    BackgroundEstimate,
    estimate_background_sigma,
    estimate_background_photutils,
    subtract_background,
)

from .detection import (
    DetectionResult,
    detect_sources_mask,
    detect_sources_segmentation,
)

from .image import ImageData

from .noise import (
    estimate_noise_map,
    make_constant_noise_map,
)


from .mask import (
    connected_source_mask,
    star_halo_mask,
)

__all__ = [
    "ImageData",
    "BackgroundEstimate",
    "DetectionResult",
    "estimate_background_sigma",
    "estimate_background_photutils",
    "subtract_background",
    "make_constant_noise_map",
    "estimate_noise_map",
    "detect_sources_mask",
    "detect_sources_segmentation",
    "connected_source_mask",
    "star_halo_mask",
]