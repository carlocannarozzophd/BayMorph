"""Residual-map diagnostic workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from baymorph.plotting.images import plot_log_data_model_residual

from .config import (
    load_resolved_config,
    resolve_config,
)
from .image_preparation import load_image_preparation_products
from .outputs import pipeline_paths
from .real_fit import _diagnostic_title


def run_residual_map_diagnostics(
    config,
    fit_results,
):
    """Generate and save residual-map diagnostics for fitted models.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Pipeline configuration dictionary or path to a YAML configuration file.
    fit_results : object or dict
        Single fit result or mapping from model name to fit result. Each fit
        result must expose ``model_name`` and ``best_model_image`` attributes.

    Returns
    -------
    None
        Residual-map figures are written to the configured fitting output paths.
    """
    if isinstance(config, (str, Path)):
        config = load_resolved_config(config)
    else:
        config = resolve_config(config)

    image_products = load_image_preparation_products(
        config,
    )

    paths = pipeline_paths(config)["fits"]

    if not isinstance(fit_results, dict):
        fit_results = {
            fit_results.model_name: fit_results,
        }

    for model_name, fit in fit_results.items():
        fig, _ = plot_log_data_model_residual(
            image_products.prepared_data,
            fit.best_model_image,
            noise=image_products.noise,
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

        plt.close(fig)