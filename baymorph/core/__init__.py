"""Core data structures and utilities for BayMorph."""

from baymorph.core.background import (
    BackgroundResult,
    estimate_background_constant,
    estimate_background_photutils,
    estimate_noise,
    subtract_background,
)
from baymorph.core.detection import (
    DetectionResult,
    detect_sources_segmentation,
)
from baymorph.core.image import ImageData
from baymorph.core.mask import (
    build_fitting_mask,
    connected_source_mask,
    intersection,
    negative_outlier_mask,
    union,
    load_external_mask,
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
    "estimate_background_constant",
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
    "build_fitting_mask",
    "connected_source_mask",
    "negative_outlier_mask",
    "intersection",
    "union",
    "detect_profile_plateau",
    "detect_sources_segmentation",
    "estimate_background_photutils",
    "estimate_noise",
    "estimate_noise_map",
    "make_constant_noise_map",
    "load_external_mask",
    "star_halo_mask",
    "subtract_background",
]
