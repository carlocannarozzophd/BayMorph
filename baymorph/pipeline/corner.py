"""Corner-plot workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from baymorph.models import (
    parameter_labels,
)

from baymorph.pipeline.outputs import (
    pipeline_paths,
)

from baymorph.plotting import (
    corner_plot,
)

from .config import (
    load_resolved_config,
    resolve_config,
)


def run_corner_diagnostics(
    config,
    fit_results,
):
    """
    Generate and save corner plots for one or more fitted models.
    """

    if isinstance(config, (str, Path)):
        config = load_resolved_config(config)
    else:
        config = resolve_config(config)

    paths = pipeline_paths(config)["fits"]

    if not isinstance(fit_results, dict):
        fit_results = {
            fit_results.model_name: fit_results,
        }

    for model_name, fit in fit_results.items():

        if fit.posterior is None:
            continue

        fig, _, _ = corner_plot(
            fit.posterior.posterior_samples.T,
            parameter_labels=parameter_labels(
                model_name,
            ),
        )

        fig.savefig(
            paths[model_name]["corner"],
            bbox_inches="tight",
        )

        plt.close(fig)