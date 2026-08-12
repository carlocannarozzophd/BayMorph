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
from baymorph.pipeline.config import (
    load_config,
    load_resolved_config,
    resolve_config,
)
from baymorph.pipeline.image_preparation import (
    load_image_preparation_products,
)
from baymorph.pipeline.multi_observation_outputs import (
    multi_observation_paths,
)
from baymorph.pipeline.multiband_fit import (
    _build_multiband_context,
)
from baymorph.pipeline.outputs import (
    pipeline_paths,
)
from baymorph.pipeline.psf_construction import (
    load_empirical_psf,
)
from baymorph.pipeline.real_fit import (
    _diagnostic_title,
    _make_model_factory,
)
from baymorph.plotting.radial import (
    plot_radial_profile_diagnostics,
)
from baymorph.simulation.psf import GaussianPSF


def _load_psf_from_config(config):
    """Load the configured PSF for radial-profile diagnostics."""
    psf_cfg = config.get(
        "psf",
        {},
    )

    psf_type = psf_cfg.get(
        "type",
        "empirical",
    )

    if psf_type == "gaussian":
        gaussian_cfg = psf_cfg.get(
            "gaussian",
            {},
        )

        return GaussianPSF(
            fwhm=float(
                gaussian_cfg[
                    "fwhm"
                ]
            )
        )

    if psf_type == "empirical":
        try:
            return load_empirical_psf(
                config,
            )
        except FileNotFoundError:
            return None

    raise ValueError(
        f"Unsupported PSF type: {psf_type}"
    )


def _posterior_samples(posterior):
    """Return equal-weight posterior samples when available."""
    if hasattr(
        posterior,
        "posterior_samples",
    ):
        return posterior.posterior_samples

    if hasattr(
        posterior,
        "samples",
    ):
        return posterior.samples

    return posterior


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
    model_name,
    psf,
    image_shape,
    radii,
    *,
    center_x,
    center_y,
    axis_ratio,
    position_angle,
    mask,
):
    """Compute single-observation posterior profiles on fixed annuli."""
    if fit.posterior is None:
        return None

    parameter_names = list(
        fit.posterior.parameter_names
    )

    model_factory = _make_model_factory(
        model_name,
        psf,
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


def _resolve_radial_config(
    config: str | Path | Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    bool,
    Path,
]:
    """Resolve configuration and identify single or multi-observation mode."""
    if isinstance(
        config,
        (str, Path),
    ):
        path = Path(
            config
        )

        raw_config = load_config(
            path
        )

        if "observations" in raw_config:
            return (
                raw_config,
                True,
                path.parent,
            )

        return (
            load_resolved_config(
                path
            ),
            False,
            path.parent,
        )

    if isinstance(
        config,
        Mapping,
    ):
        raw_config = dict(
            config
        )

        if "observations" in raw_config:
            return (
                raw_config,
                True,
                Path.cwd(),
            )

        return (
            resolve_config(
                raw_config
            ),
            False,
            Path.cwd(),
        )

    raise TypeError(
        "config must be a path or a mapping."
    )


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
) -> np.ndarray:
    """Build radial-bin edges for one observation."""
    max_radius = 0.5 * min(
        image_shape
    )

    return radial_bins(
        max_radius,
        mode=radial_cfg.get(
            "binning",
            "fixed",
        ),
        dr=float(
            radial_cfg.get(
                "dr",
                1.0,
            )
        ),
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
    """Render galaxy components while excluding additive sky terms."""
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
        )

        model_factory = _make_model_factory(
            model_name,
            psf,
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

        best_sky = float(
            fit.best_parameters.get(
                "sky",
                0.0,
            )
        )

        sky_subtracted_data = (
            np.asarray(
                image_products.prepared_data,
                dtype=float,
            )
            - best_sky
        )

        data_statistics = radial_profile_statistics_from_image(
            sky_subtracted_data,
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
                model_name=model_name,
                psf=psf,
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
            legend_title=_diagnostic_title(
                config,
                model_name,
            ),
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

    context = _build_multiband_context(
        config,
        base_dir=base_dir,
    )

    problem = context.problem

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

            best_sky = float(
                observation_fit.parameters.get(
                    "sky",
                    0.0,
                )
            )

            sky_subtracted_data = (
                np.asarray(
                    image_products.prepared_data,
                    dtype=float,
                )
                - best_sky
            )

            data_statistics = radial_profile_statistics_from_image(
                sky_subtracted_data,
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
