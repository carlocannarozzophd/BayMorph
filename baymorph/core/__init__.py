"""Core data structures and utilities for BayMorph."""

from baymorph.core.background import (
    BackgroundResult,
    estimate_background_photutils,
    estimate_background_sigma,
    estimate_noise,
    subtract_background,
)
from baymorph.core.detection import (
    DetectionResult,
    detect_sources_mask,
    detect_sources_segmentation,
)
from baymorph.core.image import ImageData
from baymorph.core.mask import (
    connected_source_mask,
    star_halo_mask,
)
from baymorph.core.noise import (
    estimate_noise_map,
    make_constant_noise_map,
)
from baymorph.core.observation import (
    Observation,
    ObservationMetadata,
)
from baymorph.core.sky import (
    PlateauResult,
    RadialProfile,
    SkyRegion,
    build_elliptical_radius_map,
    build_sky_region,
    compute_elliptical_radial_profile,
    detect_profile_plateau,
)

__all__ = [
    "BackgroundResult",
    "DetectionResult",
    "ImageData",
    "Observation",
    "ObservationMetadata",
    "PlateauResult",
    "RadialProfile",
    "SkyRegion",
    "build_elliptical_radius_map",
    "build_sky_region",
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
]