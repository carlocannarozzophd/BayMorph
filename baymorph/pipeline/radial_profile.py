"""Radial-profile diagnostic workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baymorph.analysis import (
    compute_posterior_profile_bands,
    radial_bins,
    radial_profile_from_image,
)
from baymorph.pipeline.config import (
    load_resolved_config,
    resolve_config,
)
from baymorph.pipeline.image_preparation import (
    load_image_preparation_products,
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
from baymorph.plotting.radial import plot_radial_profile_diagnostics
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

        return GaussianPSF(fwhm=float(gaussian_cfg["fwhm"]))

    if psf_type == "empirical":
        try:
            return load_empirical_psf(
                config,
            )
        except FileNotFoundError:
            return None

    raise ValueError(f"Unsupported PSF type: {psf_type}")


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
            name: float(value)
            for name, value in zip(
                parameter_names,
                sample,
            )
        }

        return model_factory(
            **parameters,
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
    mask,
):
    """Compute model-agnostic posterior radial-profile bands."""
    if fit.posterior is None:
        return None

    parameter_names = list(fit.posterior.parameter_names)

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
        mask=mask,
    )


def run_radial_profile_diagnostics(
    config,
    fit_results,
):
    """Generate and save radial-profile diagnostics for fitted models.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Pipeline configuration dictionary or path to a resolved YAML
        configuration file.
    fit_results : RealFitProducts or dict
        Single fit result or mapping from model names to fit results.

    Returns
    -------
    PosteriorProfileResult, dict, or None
        Posterior-profile result for a single fitted model, or a mapping from
        model names to posterior-profile results when multiple models are
        supplied.
    """
    single_input = not isinstance(
        fit_results,
        dict,
    )

    if isinstance(
        config,
        (str, Path),
    ):
        config = load_resolved_config(
            config,
        )
    else:
        config = resolve_config(
            config,
        )

    image_products = load_image_preparation_products(
        config,
    )

    paths = pipeline_paths(
        config,
    )["fits"]

    diagnostics_cfg = config.get(
        "diagnostics",
        {},
    )

    radial_cfg = diagnostics_cfg.get(
        "radial_profile",
        {},
    )

    if radial_cfg is True:
        radial_cfg = {}
    elif radial_cfg in (False, None):
        return {}
    elif not isinstance(radial_cfg, dict):
        raise TypeError("diagnostics.radial_profile must be a boolean or mapping.")

    psf = _load_psf_from_config(
        config,
    )

    if single_input:
        fit_results = {
            fit_results.model_name: fit_results,
        }

    profile_results = {}

    for model_name, fit in fit_results.items():
        center_x = float(fit.best_parameters["x0"])
        center_y = float(fit.best_parameters["y0"])

        max_radius = 0.5 * min(image_products.prepared_data.shape)

        radii = radial_bins(
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
        )

        radius = 0.5 * (radii[:-1] + radii[1:])

        model_factory = _make_model_factory(
            model_name,
            psf,
        )

        best_model = model_factory(
            **fit.best_parameters,
        )

        components = best_model.render_components(
            image_products.prepared_data.shape,
        )

        if not components:
            raise RuntimeError(f"Model '{model_name}' returned no rendered components.")

        component_images = {
            name: np.asarray(
                image,
                dtype=float,
            )
            for name, image in components.items()
        }

        galaxy_model_image = np.sum(
            np.stack(
                list(component_images.values()),
                axis=0,
            ),
            axis=0,
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

        profile_data = radial_profile_from_image(
            sky_subtracted_data,
            center_x=center_x,
            center_y=center_y,
            radii=radii,
            mask=image_products.fitting_mask,
        )

        profile_model = radial_profile_from_image(
            galaxy_model_image,
            center_x=center_x,
            center_y=center_y,
            radii=radii,
            mask=image_products.fitting_mask,
        )

        component_profiles = None

        if len(component_images) > 1:
            component_profiles = {
                name: radial_profile_from_image(
                    image,
                    center_x=center_x,
                    center_y=center_y,
                    radii=radii,
                    mask=image_products.fitting_mask,
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
                mask=image_products.fitting_mask,
            )

        profile_results[model_name] = profile_bands

        fig, _ = plot_radial_profile_diagnostics(
            radius=radius,
            profile_data=profile_data,
            profile_model=profile_model,
            posterior_profiles=profile_bands,
            component_profiles=component_profiles,
            legend_title=_diagnostic_title(
                config,
                model_name,
            ),
        )

        fig.savefig(
            paths[model_name]["radial_profile"],
            bbox_inches="tight",
        )

        plt.close(
            fig,
        )

    if single_input:
        return next(iter(profile_results.values()))

    return profile_results
