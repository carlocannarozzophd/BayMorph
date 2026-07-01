"""Pipeline utilities and workflows for BayMorph."""

from ..results import RealFitProducts
from .config import (
    load_config,
    load_resolved_config,
    resolve_config,
    validate_config,
)
from .corner import run_corner_diagnostics
from .cutout import (
    CutoutDefinition,
    resolve_cutout,
    resolve_fixed_cutout,
)
from .image_preparation import (
    ImagePreparationProducts,
    RealFITSPreparation,
    load_image_preparation_products,
    make_target_cutout,
    prepare_real_fits_image,
    prepare_real_image_from_config,
    save_image_preparation_products,
)
from .mask_refinement import refine_mask
from .outputs import (
    make_output_tree,
    object_output_dir,
    pipeline_paths,
    save_resolved_config,
)
from .psf_candidates import (
    PSFCandidate,
    candidate_row_passes_quality_cuts,
    estimate_candidate_half_size,
    estimate_candidate_half_size_from_sigmas,
    is_inside_exclusion_box,
    is_isolated_candidate,
    load_psf_candidates,
    run_psf_candidate_selection_from_config,
    save_psf_candidates,
    select_psf_candidates_from_rows,
    source_catalog_to_candidate_rows,
)
from .psf_construction import (
    centroid_from_flux,
    crop_center,
    estimate_psf_half_size,
    extract_centered_cutout,
    load_empirical_psf,
    normalize_psf,
    recenter_cutout,
    run_psf_construction_from_config,
    save_empirical_psf,
    save_empirical_psf_metadata,
    stack_psf_cutouts,
    subtract_outer_background,
)
from .radial_profile import run_radial_profile_diagnostics
from .real_fit import (
    load_real_fit_products,
    run_real_fit_from_config,
    save_real_fit_products,
)
from .residual_maps import run_residual_map_diagnostics
from .segmentation import select_segment_near_position
from .target import estimate_target_position
from .workflows import (
    initialize_pipeline_run,
    run_image_preparation,
    run_psf_candidate_selection,
    run_psf_construction,
    run_real_fit,
)

__all__ = [
    "RealFITSPreparation",
    "prepare_real_fits_image",
    "estimate_target_position",
    "select_segment_near_position",
    "refine_mask",
    "load_config",
    "load_resolved_config",
    "resolve_config",
    "validate_config",
    "make_output_tree",
    "object_output_dir",
    "save_resolved_config",
    "initialize_pipeline_run",
    "run_image_preparation",
    "run_psf_candidate_selection",
    "run_psf_construction",
    "run_real_fit",
    "pipeline_paths",
    "ImagePreparationProducts",
    "make_target_cutout",
    "prepare_real_image_from_config",
    "save_image_preparation_products",
    "load_image_preparation_products",
    "PSFCandidate",
    "load_psf_candidates",
    "run_psf_candidate_selection_from_config",
    "save_psf_candidates",
    "estimate_candidate_half_size",
    "is_isolated_candidate",
    "select_psf_candidates_from_rows",
    "candidate_row_passes_quality_cuts",
    "estimate_candidate_half_size_from_sigmas",
    "is_inside_exclusion_box",
    "source_catalog_to_candidate_rows",
    "load_empirical_psf",
    "run_psf_construction_from_config",
    "save_empirical_psf",
    "extract_centered_cutout",
    "normalize_psf",
    "stack_psf_cutouts",
    "subtract_outer_background",
    "centroid_from_flux",
    "recenter_cutout",
    "save_empirical_psf_metadata",
    "estimate_psf_half_size",
    "crop_center",
    "RealFitProducts",
    "run_real_fit_from_config",
    "save_real_fit_products",
    "load_real_fit_products",
    "CutoutDefinition",
    "resolve_cutout",
    "resolve_fixed_cutout",
    "run_corner_diagnostics",
    "run_residual_map_diagnostics",
    "run_radial_profile_diagnostics",
]