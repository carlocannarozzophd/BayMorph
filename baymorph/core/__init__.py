"""Core data structures and utilities for BayMorph."""

from .background import (
    BackgroundResult,
    estimate_background_photutils,
    estimate_background_sigma,
    estimate_noise,
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
from .sky import (
    PlateauResult,
    RadialProfile,
    build_elliptical_radius_map,
    compute_elliptical_radial_profile,
    detect_profile_plateau,
    SkyRegion,
    build_sky_region,
)
from .observation import (
    Observation,
    ObservationMetadata,
)

__all__ = [
    "BackgroundResult",
    "DetectionResult",
    "ImageData",
    "PlateauResult",
    "RadialProfile",
    "build_elliptical_radius_map",
    "compute_elliptical_radial_profile",
    "connected_source_mask",
    "detect_profile_plateau",
    "detect_sources_mask",
    "detect_sources_segmentation",
    "estimate_background_photutils",
    "estimate_background_sigma",
    "estimate_noise",
    "estimate_noise_map",
    "make_constant_noise_map",
    "star_halo_mask",
    "subtract_background",
    "SkyRegion",
    "build_sky_region",
    "Observation",
    "ObservationMetadata",
]
