"""Radial-profile diagnostic workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baymorph.analysis import (
    compute_posterior_profile_bands,
    radial_profile_from_image,
)

from .config import (
    load_resolved_config,
    resolve_config,
)
from .image_preparation import load_image_preparation_products
from .outputs import pipeline_paths
from .psf_construction import load_empirical_psf
from .real_fit import _diagnostic_title


def _load_psf_from_config(
    config,
):
    """Load the empirical PSF for radial-profile posterior diagnostics."""
    return load_empirical_psf(config)


def run_radial_profile_diagnostics(
    config,
    fit_results,
):
    """Generate and save radial-profile diagnostics for fitted models.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Pipeline configuration dictionary or path to a YAML configuration file.
    fit_results : object or dict
        Single fit result or mapping from model name to fit result. Each fit
        result must expose model metadata, best-fit parameters, a best-fit
        model image, and optionally posterior samples.

    Returns
    -------
    None
        Diagnostic figures are written to the configured fitting output paths.
    """
    if isinstance(config, (str, Path)):
        config = load_resolved_config(config)
    else:
        config = resolve_config(config)

    image_products = load_image_preparation_products(
        config,
    )

    paths = pipeline_paths(config)["fits"]

    diagnostics_cfg = config.get(
        "diagnostics",
        {},
    )

    if not isinstance(fit_results, dict):
        fit_results = {
            fit_results.model_name: fit_results,
        }

    for model_name, fit in fit_results.items():
        max_radius = 0.5 * min(
            image_products.prepared_data.shape
        )

        radii = np.arange(
            0.0,
            max_radius + 1.0,
            1.0,
        )

        radius = 0.5 * (
            radii[:-1]
            + radii[1:]
        )

        profile_data = radial_profile_from_image(
            image_products.prepared_data,
            center_x=fit.best_parameters["x0"],
            center_y=fit.best_parameters["y0"],
            radii=radii,
            mask=image_products.contaminant_mask,
        )

        profile_model = radial_profile_from_image(
            fit.best_model_image,
            center_x=fit.best_parameters["x0"],
            center_y=fit.best_parameters["y0"],
            radii=radii,
            mask=image_products.contaminant_mask,
        )

        profile_noise = radial_profile_from_image(
            image_products.noise,
            center_x=fit.best_parameters["x0"],
            center_y=fit.best_parameters["y0"],
            radii=radii,
            mask=image_products.contaminant_mask,
        )

        fit.radius = radius
        fit.profile_data = profile_data
        fit.profile_model = profile_model
        fit.profile_noise = profile_noise

        residual_sigma_profile = (
            profile_data
            - profile_model
        ) / profile_noise

        valid = (
            np.isfinite(profile_data)
            & np.isfinite(profile_model)
            & (profile_data > 0)
            & (profile_model > 0)
        )

        log_data = np.full_like(
            profile_data,
            np.nan,
        )

        log_model = np.full_like(
            profile_model,
            np.nan,
        )

        log_data[valid] = np.log10(
            profile_data[valid]
        )

        log_model[valid] = np.log10(
            profile_model[valid]
        )

        profile_bands = None

        if (
            diagnostics_cfg.get(
                "posterior_profile_bands",
                True,
            )
            and model_name == "double_sersic"
            and fit.posterior is not None
        ):
            profile_bands = compute_posterior_profile_bands(
                fit.posterior.posterior_samples,
                shape=image_products.prepared_data.shape,
                radii=radii,
                mask=image_products.contaminant_mask,
                psf=_load_psf_from_config(
                    config,
                ),
                include_sky=True,
                profile_center=(
                    fit.best_parameters["x0"],
                    fit.best_parameters["y0"],
                ),
            )

        fig = plt.figure(
            figsize=(15, 15),
        )

        gs = fig.add_gridspec(
            5,
            1,
            hspace=0.0,
        )

        ax_main = fig.add_subplot(
            gs[:4, 0],
        )

        ax_res = fig.add_subplot(
            gs[4, 0],
            sharex=ax_main,
        )

        ax_main.plot(
            radius,
            log_data,
            "o",
            ms=4,
            label="Data",
        )

        if profile_bands is not None:
            total_band = profile_bands["total"]

            band_valid = (
                np.isfinite(total_band["p16"])
                & np.isfinite(total_band["p50"])
                & np.isfinite(total_band["p84"])
                & (total_band["p16"] > 0)
                & (total_band["p50"] > 0)
                & (total_band["p84"] > 0)
            )

            log_p16 = np.full_like(
                total_band["p16"],
                np.nan,
            )

            log_p50 = np.full_like(
                total_band["p50"],
                np.nan,
            )

            log_p84 = np.full_like(
                total_band["p84"],
                np.nan,
            )

            log_p16[band_valid] = np.log10(
                total_band["p16"][band_valid]
            )

            log_p50[band_valid] = np.log10(
                total_band["p50"][band_valid]
            )

            log_p84[band_valid] = np.log10(
                total_band["p84"][band_valid]
            )

            ax_main.fill_between(
                radius,
                log_p16,
                log_p84,
                alpha=0.25,
                label="68% posterior band",
            )

            ax_main.plot(
                radius,
                log_p50,
                lw=2,
                label="Posterior median",
            )

        ax_main.plot(
            radius,
            log_model,
            lw=2,
            label="Best model",
        )

        ax_main.set_ylabel(
            r"$\log_{10}(I)$",
        )

        ax_main.set_title(
            _diagnostic_title(
                config,
                model_name,
            ),
        )

        ax_main.legend(
            frameon=False,
        )

        ax_res.axhline(
            0.0,
            ls="--",
            lw=1,
        )

        for sigma in [1.0, 2.0, 3.0]:
            ax_res.axhline(
                sigma,
                ls=":",
                lw=0.8,
            )

            ax_res.axhline(
                -sigma,
                ls=":",
                lw=0.8,
            )

        ax_res.plot(
            radius,
            residual_sigma_profile,
            lw=1.5,
        )

        ax_res.set_ylim(
            -5.0,
            5.0,
        )

        ax_res.set_xlabel(
            "Radius [pix]",
        )

        ax_res.set_ylabel(
            "Residual [$\\sigma$]",
        )

        fig.savefig(
            paths[model_name]["radial_profile"],
            bbox_inches="tight",
        )

        plt.close(fig)