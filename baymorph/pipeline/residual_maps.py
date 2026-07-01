from __future__ import annotations

import matplotlib.pyplot as plt

from .image_preparation import (
    load_image_preparation_products,
)

from .outputs import (
    pipeline_paths,
)

from .real_fit import (
    _diagnostic_title,
)

from baymorph.plotting.images import (
    plot_log_data_model_residual,
)


def run_residual_map_diagnostics(
    config,
    fit_results,
):
    """
    Generate and save residual maps for one or more fitted models.
    """

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