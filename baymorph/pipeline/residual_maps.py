"""Residual-map diagnostic workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from baymorph.plotting.images import (
    plot_pipeline_fit_maps,
)

from baymorph.simulation.psf import GaussianPSF

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
from .real_fit import (
    _diagnostic_title,
)

from .psf_construction import (
    load_empirical_psf,
)


def _psf_to_image(psf, config=None):
    """Return a displayable PSF image from a PSF object, array, or config."""
    if psf is not None:
        kernel = getattr(psf, "kernel", None)

        if callable(kernel):
            return kernel()

        if kernel is not None:
            return kernel

        return psf

    if config is None:
        return None

    psf_cfg = config.get("psf", {})

    if not psf_cfg:
        psf_cfg = config.get("simulation", {}).get("psf", {})

    psf_type = psf_cfg.get("type")

    if psf_type == "gaussian":
        gaussian_cfg = psf_cfg.get("gaussian", {})
        fwhm = gaussian_cfg.get(
            "fwhm",
            psf_cfg.get("fwhm"),
        )

        if fwhm is None:
            return None

        return GaussianPSF(
            fwhm=float(fwhm),
        ).kernel()

    return None


def run_residual_map_diagnostics(
    config,
    fit_results,
):
    """Generate and save pipeline residual-map diagnostics.

    For each fitted model, this routine creates the standard BayMorph
    residual-map summary, including the original image, original and
    prepared cutouts, best-fit model, combined masks, PSF, residual map,
    and noise-normalized residual map.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Pipeline configuration dictionary or path to a YAML configuration
        file.

    fit_results : RealFitProducts or dict
        Single fit result or mapping from model name to fit result. Each
        fit result must expose at least

        - ``model_name``
        - ``best_model_image``

    Returns
    -------
    None
        Residual-map figures are written to the configured fitting-output
        directories.
    """
    if isinstance(config, (str, Path)):
        config = load_resolved_config(config)
    else:
        config = resolve_config(config)

    image_products = load_image_preparation_products(
        config,
    )

    paths = pipeline_paths(
        config,
    )["fits"]

    try:
        psf = load_empirical_psf(
            config,
        )
    except FileNotFoundError:
        psf = None

    if not isinstance(
        fit_results,
        dict,
    ):
        fit_results = {
            fit_results.model_name: fit_results,
        }

    for model_name, fit in fit_results.items():
        fig, _ = plot_pipeline_fit_maps(
            original_cutout=image_products.cutout,
            prepared_cutout=image_products.prepared_data,
            model=fit.best_model_image,
            noise=image_products.noise,
            segmentation=image_products.segmentation,
            psf=_psf_to_image(psf, config),
            contaminant_mask=image_products.contaminant_mask,
            target_mask=image_products.target_segment,
            original_image=image_products.original_image,
            cutout_bounds=image_products.cutout_bounds,
            title=_diagnostic_title(
                config,
                model_name,
            ),
            show=False,
        )

        fig.savefig(
            paths[model_name]["residual"],
            bbox_inches="tight",
        )

        plt.close(
            fig,
        )
