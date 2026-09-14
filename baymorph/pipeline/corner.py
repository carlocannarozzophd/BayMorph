"""Corner-plot diagnostic workflow."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from baymorph.pipeline.config import (
    load_config,
    load_resolved_config,
    resolve_config,
)
from baymorph.pipeline.multi_observation_outputs import (
    multi_observation_paths,
)
from baymorph.pipeline.outputs import pipeline_paths
from baymorph.pipeline.real_fit import (
    _fit_diagnostic_title,
)
from baymorph.plotting import corner_plot


def _posterior_samples(posterior):
    """Return equal-weight posterior samples when available."""
    nested_posterior = getattr(
        posterior,
        "posterior",
        None,
    )

    if (
        nested_posterior is not None
        and hasattr(
            nested_posterior,
            "equal_weight_samples",
        )
    ):
        return nested_posterior.equal_weight_samples

    if hasattr(
        posterior,
        "equal_weight_samples",
    ):
        return posterior.equal_weight_samples

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

def _resolve_corner_config(
    config: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Resolve a diagnostic configuration and identify its workflow mode."""
    if isinstance(
        config,
        (str, Path),
    ):
        raw_config = load_config(
            config
        )

        if "observations" in raw_config:
            return (
                raw_config,
                True,
            )

        return (
            load_resolved_config(
                config
            ),
            False,
        )

    raw_config = dict(
        config
    )

    if "observations" in raw_config:
        return (
            raw_config,
            True,
        )

    return (
        resolve_config(
            raw_config
        ),
        False,
    )


def run_corner_diagnostics(
    config,
    fit_results,
):
    """Generate and save corner plots for fitted BayMorph models.

    Single-observation fits use the established pipeline output tree.
    Multi-observation fits write one corner plot for the single joint
    posterior into the joint analysis output tree.
    """
    resolved_config, is_multi_observation = (
        _resolve_corner_config(
            config
        )
    )

    if is_multi_observation:
        paths = multi_observation_paths(
            resolved_config
        )["fits"]
    else:
        paths = pipeline_paths(
            resolved_config
        )["fits"]

    if not isinstance(
        fit_results,
        dict,
    ):
        fit_results = {
            fit_results.model_name: fit_results,
        }

    for model_name, fit in fit_results.items():
        if fit.posterior is None:
            continue

        samples = _posterior_samples(
            fit.posterior
        )

        fit_paths = paths[
            model_name
        ]

        fit_paths[
            "dir"
        ].mkdir(
            parents=True,
            exist_ok=True,
        )

        corner_kwargs = {
            "parameters": samples.T,
            "parameter_labels": fit.posterior.parameter_names,
            "sigma": [
                1,
                2,
                3,
            ],
        }

        corner_kwargs[
            "title"
        ] = _fit_diagnostic_title(
            resolved_config,
            model_name,
        )

        fig, _, _ = corner_plot(
            **corner_kwargs
        )

        fig.savefig(
            fit_paths[
                "corner"
            ],
            bbox_inches="tight",
        )

        plt.close(
            fig
        )
