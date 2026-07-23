"""Corner-plot diagnostic workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from baymorph.pipeline.config import (
    load_resolved_config,
    resolve_config,
)
from baymorph.pipeline.outputs import pipeline_paths
from baymorph.plotting import corner_plot


def _posterior_samples(posterior):
    """Return equal-weight posterior samples."""
    if hasattr(posterior, "posterior_samples"):
        return posterior.posterior_samples

    if hasattr(posterior, "samples"):
        return posterior.samples

    return posterior


def run_corner_diagnostics(
    config,
    fit_results,
):
    """Generate and save corner plots for fitted BayMorph models."""
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

        samples = _posterior_samples(
            fit.posterior,
        )

        fig, _, _ = corner_plot(
            parameters=samples.T,
            parameter_labels=fit.posterior.parameter_names,
            sigma=[1, 2, 3],
        )

        fig.savefig(
            paths[model_name]["corner"],
            bbox_inches="tight",
        )

        plt.close(fig)
