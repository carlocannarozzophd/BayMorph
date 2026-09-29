"""Radial-profile diagnostic workflow."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from baymorph.analysis import (
    compute_posterior_profile_bands,
    radial_bins,
    radial_profile_from_image,
)
from baymorph.analysis.radial_profiles import (
    radial_profile_statistics_from_image,
)
from baymorph.pipeline.diagnostics_common import (
    diagnostic_title as _diagnostic_title,
    object_label as _object_label,
    posterior_samples as _posterior_samples,
    resolve_diagnostic_config,
)
from baymorph.pipeline.image_preparation import (
    load_image_preparation_products,
)
from baymorph.pipeline.multi_observation_outputs import (
    multi_observation_paths,
)
from baymorph.pipeline.multi_observation_fit import (
    _build_multi_observation_context,
)
from baymorph.pipeline.outputs import (
    pipeline_paths,
)
from baymorph.pipeline.psf_construction import (
    estimate_psf_core_fwhm,
)
from baymorph.pipeline.real_fit import (
    _load_psf_from_config as _load_fit_psf_from_config,
    _make_model_factory,
)
from baymorph.plotting.radial import (
    plot_radial_profile_diagnostics,
)


def _load_psf_from_config(config):
    """Load the configured PSF for radial-profile diagnostics.

    Unlike the fit-time PSF loader, a missing empirical PSF file degrades
    to no PSF rather than raising, since this workflow only displays an
    already-completed fit.
    """
    try:
        return _load_fit_psf_from_config(config)
    except FileNotFoundError:
        return None


def _psf_fwhm_pixels(
    psf,
) -> float | None:
    """Return the spatial-resolution FWHM represented by a PSF.

    Gaussian PSFs use their configured FWHM directly. Empirical PSFs use
    the measured circularized core FWHM of the actual kernel used by the
    fit. ``None`` is returned when no reliable positive finite estimate is
    available.
    """
    if psf is None:
        return None

    configured_fwhm = getattr(
        psf,
        "fwhm",
        None,
    )

    if configured_fwhm is not None:
        configured_fwhm = float(
            configured_fwhm
        )

        if (
            np.isfinite(
                configured_fwhm
            )
            and configured_fwhm > 0.0
        ):
            return configured_fwhm

    kernel = getattr(
        psf,
        "kernel",
        None,
    )

    if callable(
        kernel
    ):
        kernel = kernel()
    elif kernel is None:
        kernel = psf

    try:
        fwhm = estimate_psf_core_fwhm(
            np.asarray(
                kernel,
                dtype=float,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if (
        fwhm is None
        or not np.isfinite(
            fwhm
        )
        or fwhm <= 0.0
    ):
        return None

    return float(
        fwhm
    )


def _default_radial_dr(
    psf,
) -> float:
    """Return the default PSF-aware radial-bin width in pixels."""
    fwhm = _psf_fwhm_pixels(
        psf
    )

    if fwhm is None:
        return 1.0

    return max(
        1.0,
        0.5 * fwhm,
    )


def _make_sample_model_factory(
    model_factory,
    parameter_names,
):
    """Return a model factory accepting one posterior sample."""

    def sample_model_factory(sample):
        parameters = {
            name: float(
                value
            )
            for name, value in zip(
                parameter_names,
                sample,
                strict=True,
            )
        }

        return model_factory(
            **parameters,
        )

    return sample_model_factory


def _make_multi_observation_sample_model_factory(
    problem,
    observation_name: str,
):
    """Build one observation representation from a global posterior sample."""

    def sample_model_factory(sample):
        routed = problem.routed_parameters(
            np.asarray(
                sample,
                dtype=float,
            )
        )

        return problem.representation_factory.build(
            observation_name,
            routed[
                observation_name
            ],
        )

    return sample_model_factory


def _compute_posterior_profiles(
    fit,
    model_factory,
    image_shape,
    radii,
    *,
    center_x,
    center_y,
    axis_ratio,
    position_angle,
    mask,
):
    """Compute single-observation posterior profiles on fixed annuli.

    The caller supplies the observed model factory it already built from the
    configuration. Rebuilding one here would silently fall back to the
    default residual-background model, and a posterior sampled with a plane
    background would then be rejected by a constant-background constructor.
    """
    if fit.posterior is None:
        return None

    parameter_names = list(
        fit.posterior.parameter_names
    )

    sample_model_factory = _make_sample_model_factory(
        model_factory,
        parameter_names,
    )

    return compute_posterior_profile_bands(
        posterior=_posterior_samples(
            fit.posterior,
        ),
        model_factory=sample_model_factory,
        image_shape=image_shape,
        radii=radii,
        center_x=center_x,
        center_y=center_y,
        axis_ratio=axis_ratio,
        position_angle=position_angle,
        mask=mask,
    )


_resolve_radial_config = resolve_diagnostic_config


def _radial_configuration(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the enabled radial-profile configuration."""
    diagnostics_cfg = config.get(
        "diagnostics",
        {},
    )

    radial_cfg = diagnostics_cfg.get(
        "radial_profile",
        {},
    )

    if radial_cfg is True:
        return {}

    if radial_cfg in (
        False,
        None,
    ):
        return None

    if not isinstance(
        radial_cfg,
        Mapping,
    ):
        raise TypeError(
            "diagnostics.radial_profile must be a boolean or mapping."
        )

    return dict(
        radial_cfg
    )


def _profile_geometry(
    parameters: Mapping[str, float],
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """Return detector-space centre, axis ratio, and position angle."""
    try:
        center_x = float(
            parameters[
                "x0"
            ]
        )
        center_y = float(
            parameters[
                "y0"
            ]
        )
    except KeyError as exc:
        raise ValueError(
            "Radial-profile diagnostics require detector-space x0 and y0."
        ) from exc

    axis_ratio = float(
        parameters.get(
            "q",
            1.0,
        )
    )

    position_angle = float(
        parameters.get(
            "theta",
            0.0,
        )
    )

    return (
        center_x,
        center_y,
        axis_ratio,
        position_angle,
    )


def _radial_edges(
    image_shape: tuple[int, int],
    radial_cfg: Mapping[str, Any],
    *,
    axis_ratio: float,
    psf=None,
) -> np.ndarray:
    """Build radial-bin edges for one observation.

    An explicitly configured ``dr`` always takes precedence. Otherwise the
    diagnostic defaults to half the spatial-resolution FWHM of the fitted
    PSF, with a one-pixel lower limit and a one-pixel fallback when the PSF
    resolution cannot be estimated.
    """
    max_radius = 0.5 * min(
        image_shape
    )

    configured_dr = radial_cfg.get(
        "dr"
    )

    if configured_dr is None:
        dr = _default_radial_dr(
            psf
        )
    else:
        dr = float(
            configured_dr
        )

    return radial_bins(
        max_radius,
        mode=radial_cfg.get(
            "binning",
            "fixed",
        ),
        dr=dr,
        min_pixels=int(
            radial_cfg.get(
                "min_pixels",
                50,
            )
        ),
        log_n_bins=int(
            radial_cfg.get(
                "log_n_bins",
                40,
            )
        ),
        axis_ratio=axis_ratio,
    )

def _component_images(
    model,
    image_shape: tuple[int, int],
    *,
    model_name: str,
) -> dict[str, np.ndarray]:
    """Render galaxy components while excluding additive background terms."""
    components = model.render_components(
        image_shape
    )

    if not components:
        raise RuntimeError(
            f"Model '{model_name}' returned no rendered components."
        )

    return {
        name: np.asarray(
            image,
            dtype=float,
        )
        for name, image in components.items()
    }


def _total_component_image(
    component_images: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Return the sum of all rendered galaxy components."""
    return np.sum(
        np.stack(
            list(
                component_images.values()
            ),
            axis=0,
        ),
        axis=0,
    )


def _residual_background_image(
    model,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Render the fitted residual background on one observation grid.

    Every production observed model applies its residual background as an
    observation effect and exposes ``render_background()`` (AD-031). A model
    without it cannot be diagnosed here, so the function fails closed.
    """
    if not hasattr(
        model,
        "render_background",
    ):
        raise TypeError(
            "Radial diagnostics require an observed model exposing "
            "render_background()."
        )

    background = np.asarray(
        model.render_background(
            image_shape
        ),
        dtype=float,
    )

    if background.shape != image_shape:
        raise ValueError(
            "Rendered background shape does not match the fitted image."
        )

    if not np.all(
        np.isfinite(
            background
        )
    ):
        raise ValueError(
            "Rendered background contains non-finite values."
        )

    return background


def _run_single_observation_radial_profiles(
    config: Mapping[str, Any],
    fit_results,
    *,
    radial_cfg: Mapping[str, Any],
):
    """Generate elliptical radial profiles for a single observation."""
    single_input = not isinstance(
        fit_results,
        dict,
    )

    image_products = load_image_preparation_products(
        config,
    )

    paths = pipeline_paths(
        config,
    )[
        "fits"
    ]

    diagnostics_cfg = config.get(
        "diagnostics",
        {},
    )

    psf = _load_psf_from_config(
        config,
    )

    if single_input:
        fit_results = {
            fit_results.model_name: fit_results,
        }

    profile_results = {}

    for model_name, fit in fit_results.items():
        (
            center_x,
            center_y,
            axis_ratio,
            position_angle,
        ) = _profile_geometry(
            fit.best_parameters
        )

        radii = _radial_edges(
            image_products.prepared_data.shape,
            radial_cfg,
            axis_ratio=axis_ratio,
            psf=psf,
        )

        model_factory = _make_model_factory(
            model_name,
            psf,
            background_cfg=(
                config
                .get("fit", {})
                .get("background")
            ),
        )

        best_model = model_factory(
            **fit.best_parameters,
        )

        component_images = _component_images(
            best_model,
            image_products.prepared_data.shape,
            model_name=model_name,
        )

        galaxy_model_image = _total_component_image(
            component_images
        )

        residual_background = _residual_background_image(
            best_model,
            image_products.prepared_data.shape,
        )

        background_subtracted_data = (
            np.asarray(
                image_products.prepared_data,
                dtype=float,
            )
            - residual_background
        )

        data_statistics = radial_profile_statistics_from_image(
            background_subtracted_data,
            center_x=center_x,
            center_y=center_y,
            radii=radii,
            mask=image_products.fitting_mask,
            axis_ratio=axis_ratio,
            position_angle=position_angle,
        )

        radius = data_statistics.radius
        profile_data = data_statistics.median

        profile_model = radial_profile_from_image(
            galaxy_model_image,
            center_x=center_x,
            center_y=center_y,
            radii=radii,
            mask=image_products.fitting_mask,
            axis_ratio=axis_ratio,
            position_angle=position_angle,
        )

        component_profiles = None

        if len(
            component_images
        ) > 1:
            component_profiles = {
                name: radial_profile_from_image(
                    image,
                    center_x=center_x,
                    center_y=center_y,
                    radii=radii,
                    mask=image_products.fitting_mask,
                    axis_ratio=axis_ratio,
                    position_angle=position_angle,
                )
                for name, image in component_images.items()
            }

        fit.radius = radius
        fit.profile_data = profile_data
        fit.profile_model = profile_model

        profile_bands = None

        if diagnostics_cfg.get(
            "posterior_profile_bands",
            True,
        ):
            profile_bands = _compute_posterior_profiles(
                fit=fit,
                model_factory=model_factory,
                image_shape=image_products.prepared_data.shape,
                radii=radii,
                center_x=center_x,
                center_y=center_y,
                axis_ratio=axis_ratio,
                position_angle=position_angle,
                mask=image_products.fitting_mask,
            )

        profile_results[
            model_name
        ] = profile_bands

        fig, _ = plot_radial_profile_diagnostics(
            radius=radius,
            profile_data=profile_data,
            profile_model=profile_model,
            profile_data_p16=data_statistics.p16,
            profile_data_p84=data_statistics.p84,
            profile_data_mean=data_statistics.mean,
            posterior_profiles=profile_bands,
            component_profiles=component_profiles,
            title=_diagnostic_title(
                config,
                model_name,
            ),
            legend_title=None,
        )

        output_path = paths[
            model_name
        ][
            "radial_profile"
        ]

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            bbox_inches="tight",
        )

        plt.close(
            fig,
        )

    if single_input:
        return next(
            iter(
                profile_results.values()
            )
        )

    return profile_results


def _run_multi_observation_radial_profiles(
    config: Mapping[str, Any],
    fit_results,
    *,
    base_dir: Path,
    radial_cfg: Mapping[str, Any],
):
    """Generate elliptical radial profiles on every native observation grid."""
    single_input = not isinstance(
        fit_results,
        dict,
    )

    if single_input:
        fits = {
            fit_results.model_name: fit_results,
        }
    else:
        fits = fit_results

    configured_observations = tuple(
        config[
            "observations"
        ]
    )

    for fit in fits.values():
        fit_observations = getattr(
            fit,
            "observations",
            None,
        )

        if not isinstance(
            fit_observations,
            Mapping,
        ):
            raise TypeError(
                "Multi-observation radial diagnostics require fit products "
                "exposing an 'observations' mapping."
            )

        missing = [
            observation_name
            for observation_name in configured_observations
            if observation_name not in fit_observations
        ]

        if missing:
            missing_text = ", ".join(
                repr(
                    name
                )
                for name in missing
            )

            raise ValueError(
                "Fit products are missing observation products for "
                f"{missing_text}."
            )

    context = _build_multi_observation_context(
        config,
        base_dir=base_dir,
    )

    problem = context.problem

    psfs = getattr(
        context,
        "psfs",
        {},
    )

    paths = multi_observation_paths(
        config
    )[
        "fits"
    ]

    diagnostics_cfg = config.get(
        "diagnostics",
        {},
    )

    results_by_model = {}

    for model_name, fit in fits.items():
        best_sample = np.asarray(
            [
                fit.best_parameters[
                    name
                ]
                for name in problem.parameter_names
            ],
            dtype=float,
        )

        routed_best = problem.routed_parameters(
            best_sample
        )

        observation_results = {}

        for observation_name in configured_observations:
            image_products = context.image_products[
                observation_name
            ]

            observation_configs = getattr(
                context,
                "observation_configs",
                {},
            )

            observation_config = observation_configs.get(
                observation_name,
                {},
            )

            object_title = _object_label(
                observation_config
            )

            if object_title is None:
                object_title = _object_label(
                    config
                )

            observation_psf = psfs.get(
                observation_name
            )

            observation_fit = fit.observations[
                observation_name
            ]

            (
                center_x,
                center_y,
                axis_ratio,
                position_angle,
            ) = _profile_geometry(
                observation_fit.parameters
            )

            radii = _radial_edges(
                image_products.prepared_data.shape,
                radial_cfg,
                axis_ratio=axis_ratio,
                psf=observation_psf,
            )

            best_model = (
                problem
                .representation_factory
                .build(
                    observation_name,
                    routed_best[
                        observation_name
                    ],
                )
            )

            component_images = _component_images(
                best_model,
                image_products.prepared_data.shape,
                model_name=model_name,
            )

            galaxy_model_image = _total_component_image(
                component_images
            )

            residual_background = _residual_background_image(
                best_model,
                image_products.prepared_data.shape,
            )

            background_subtracted_data = (
                np.asarray(
                    image_products.prepared_data,
                    dtype=float,
                )
                - residual_background
            )

            data_statistics = radial_profile_statistics_from_image(
                background_subtracted_data,
                center_x=center_x,
                center_y=center_y,
                radii=radii,
                mask=image_products.fitting_mask,
                axis_ratio=axis_ratio,
                position_angle=position_angle,
            )

            profile_model = radial_profile_from_image(
                galaxy_model_image,
                center_x=center_x,
                center_y=center_y,
                radii=radii,
                mask=image_products.fitting_mask,
                axis_ratio=axis_ratio,
                position_angle=position_angle,
            )

            component_profiles = None

            if len(
                component_images
            ) > 1:
                component_profiles = {
                    name: radial_profile_from_image(
                        image,
                        center_x=center_x,
                        center_y=center_y,
                        radii=radii,
                        mask=image_products.fitting_mask,
                        axis_ratio=axis_ratio,
                        position_angle=position_angle,
                    )
                    for name, image in component_images.items()
                }

            profile_bands = None

            if diagnostics_cfg.get(
                "posterior_profile_bands",
                True,
            ):
                sample_model_factory = (
                    _make_multi_observation_sample_model_factory(
                        problem,
                        observation_name,
                    )
                )

                profile_bands = compute_posterior_profile_bands(
                    posterior=_posterior_samples(
                        fit.posterior
                    ),
                    model_factory=sample_model_factory,
                    image_shape=image_products.prepared_data.shape,
                    radii=radii,
                    center_x=center_x,
                    center_y=center_y,
                    axis_ratio=axis_ratio,
                    position_angle=position_angle,
                    mask=image_products.fitting_mask,
                )

            observation_results[
                observation_name
            ] = profile_bands

            fig, _ = plot_radial_profile_diagnostics(
                radius=data_statistics.radius,
                profile_data=data_statistics.median,
                profile_model=profile_model,
                profile_data_p16=data_statistics.p16,
                profile_data_p84=data_statistics.p84,
                profile_data_mean=data_statistics.mean,
                posterior_profiles=profile_bands,
                component_profiles=component_profiles,
                title=object_title,
                legend_title=(
                    f"{model_name} | observation: "
                    f"{observation_name}"
                ),
            )

            output_path = paths[
                model_name
            ][
                "observations"
            ][
                observation_name
            ][
                "radial_profile"
            ]

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            fig.savefig(
                output_path,
                bbox_inches="tight",
            )

            plt.close(
                fig
            )

        results_by_model[
            model_name
        ] = observation_results

    if single_input:
        return next(
            iter(
                results_by_model.values()
            )
        )

    return results_by_model


def run_radial_profile_diagnostics(
    config,
    fit_results,
):
    """Generate and save elliptical radial-profile diagnostics.

    Single-observation analyses use one detector-space ellipse geometry.
    Multi-observation analyses use the detector-space best-fit geometry of
    each observation while retaining one shared joint posterior.

    Data, maximum-likelihood model components, and posterior predictions are
    measured on fixed best-fit elliptical annuli for each observation.

    Returns
    -------
    PosteriorProfileResult, dict, or None
        Single-observation posterior profile products, or observation-indexed
        posterior profile products for a joint multi-observation fit.
    """
    (
        resolved_config,
        is_multi_observation,
        base_dir,
    ) = _resolve_radial_config(
        config
    )

    radial_cfg = _radial_configuration(
        resolved_config
    )

    if radial_cfg is None:
        return {}

    if is_multi_observation:
        return _run_multi_observation_radial_profiles(
            resolved_config,
            fit_results,
            base_dir=base_dir,
            radial_cfg=radial_cfg,
        )

    return _run_single_observation_radial_profiles(
        resolved_config,
        fit_results,
        radial_cfg=radial_cfg,
    )
