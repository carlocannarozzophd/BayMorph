"""Radial-profile diagnostic workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baymorph.analysis import (
    compute_posterior_profile_bands,
    radial_profile_from_image,
)
from baymorph.plotting.radial import plot_radial_profile_diagnostics

from .config import (
    load_resolved_config,
    resolve_config,
)
from .image_preparation import (
    load_image_preparation_products,
)
from .outputs import (
    pipeline_paths,
)
from .psf_construction import (
    load_empirical_psf,
)
from .real_fit import (
    _diagnostic_title,
    _make_model_factory,
)


def _load_psf_from_config(config):
    """Load the empirical PSF for radial-profile posterior diagnostics."""
    try:
        return load_empirical_psf(config)
    except FileNotFoundError:
        return None


def _posterior_samples(posterior):
    """Return posterior samples from a posterior result."""
    if hasattr(posterior, "samples"):
        return posterior.samples

    if hasattr(posterior, "posterior_samples"):
        return posterior.posterior_samples

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
        posterior=_posterior_samples(fit.posterior),
        model_factory=sample_model_factory,
        image_shape=image_shape,
        radii=radii,
        center_x=center_x,
        center_y=center_y,
        mask=mask,
    )


def run_radial_profile_diagnostics(config, fit_results):
    """Generate and save radial-profile diagnostics for fitted models."""
    single_input = not isinstance(fit_results, dict)

    if isinstance(config, (str, Path)):
        config = load_resolved_config(config)
    else:
        config = resolve_config(config)

    image_products = load_image_preparation_products(config)
    paths = pipeline_paths(config)["fits"]
    diagnostics_cfg = config.get("diagnostics", {})
    psf = _load_psf_from_config(config)

    if single_input:
        fit_results = {fit_results.model_name: fit_results}

    profile_results = {}

    for model_name, fit in fit_results.items():
        center_x = fit.best_parameters["x0"]
        center_y = fit.best_parameters["y0"]

        max_radius = 0.5 * min(image_products.prepared_data.shape)

        radii = np.arange(
            0.0,
            max_radius + 1.0,
            1.0,
        )

        radius = 0.5 * (radii[:-1] + radii[1:])

        profile_data = radial_profile_from_image(
            image_products.prepared_data,
            center_x=center_x,
            center_y=center_y,
            radii=radii,
            mask=image_products.contaminant_mask,
        )

        profile_model = radial_profile_from_image(
            fit.best_model_image,
            center_x=center_x,
            center_y=center_y,
            radii=radii,
            mask=image_products.contaminant_mask,
        )

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
                mask=image_products.contaminant_mask,
            )

        profile_results[model_name] = profile_bands

        fig, axes = plot_radial_profile_diagnostics(
            radius=radius,
            profile_data=profile_data,
            profile_model=profile_model,
            posterior_profiles=profile_bands,
        )

        axes[0].set_title(
            _diagnostic_title(config, model_name),
            fontsize=24,
        )

        fig.savefig(
            paths[model_name]["radial_profile"],
            bbox_inches="tight",
        )

        plt.close(fig)

    if single_input:
        return next(iter(profile_results.values()))

    return profile_results