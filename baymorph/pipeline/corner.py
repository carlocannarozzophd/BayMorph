"""Corner-plot diagnostic workflow."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from baymorph.pipeline.image_preparation import (
    load_image_preparation_products,
)
from baymorph.pipeline.multi_observation_outputs import (
    multi_observation_paths,
)
from baymorph.pipeline.outputs import (
    pipeline_paths,
    simulation_truth_path,
)
from baymorph.pipeline.diagnostics_common import (
    fit_diagnostic_title as _fit_diagnostic_title,
    posterior_samples,
    resolve_diagnostic_config,
)
from baymorph.plotting import corner_plot


_posterior_samples = posterior_samples


def _simulation_truth_vector(
    config: Mapping[str, Any],
    parameter_names,
) -> list[float | None] | None:
    """Return simulation truth aligned to the sampled parameter order.

    The persisted ``truth.json`` already stores values in fitted parameter
    names and units, so no unit conversion happens here. Two quantities are
    deliberately not mapped onto sampled parameters:

    ``sky``
        The simulated sky plateau is removed during image preparation. The
        sampled ``background_level`` is the residual background of the
        prepared image, not the simulated sky, so marking it with the sky
        truth would be scientifically wrong.

    absent names
        Any sampled parameter without a truth entry stays ``None`` and is
        left unmarked.

    Truth ``x0``/``y0`` are full-image coordinates while sampled positions are
    cutout coordinates, so the prepared cutout origin is subtracted.

    Returns
    -------
    list or None
        Truth values ordered as ``parameter_names``, or None when the run has
        no persisted simulation truth or none of its values are usable.
    """
    truth_path = simulation_truth_path(
        config
    )

    if not truth_path.is_file():
        return None

    with truth_path.open() as stream:
        payload = json.load(
            stream
        )

    truth = payload.get(
        "truth",
        {},
    )

    if not isinstance(
        truth,
        Mapping,
    ):
        raise TypeError(
            f"Simulation truth in {truth_path} must be a mapping."
        )

    truth = {
        key: value
        for key, value in truth.items()
        if key != "sky"
    }

    image_products = load_image_preparation_products(
        config
    )
    cutout_bounds = image_products.cutout_bounds

    if cutout_bounds is None:
        raise ValueError(
            "Prepared products are missing cutout_bounds, so simulation "
            "truth cannot be transformed into fitting coordinates."
        )

    origin = {
        "x0": float(
            cutout_bounds["xmin"]
        ),
        "y0": float(
            cutout_bounds["ymin"]
        ),
    }

    values: list[float | None] = []

    for name in parameter_names:
        if name not in truth:
            values.append(
                None
            )
            continue

        values.append(
            float(
                truth[name]
            )
            - origin.get(
                name,
                0.0,
            )
        )

    if all(
        value is None
        for value in values
    ):
        return None

    return values


def _resolve_corner_config(
    config: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Resolve a diagnostic configuration and identify its workflow mode."""
    resolved_config, is_multi_observation, _ = resolve_diagnostic_config(
        config
    )

    return (
        resolved_config,
        is_multi_observation,
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

        if not is_multi_observation:
            # Multi-observation runs have one cutout origin per observation
            # and no single simulated truth, so they stay unmarked.
            truths = _simulation_truth_vector(
                resolved_config,
                fit.posterior.parameter_names,
            )

            if truths is not None:
                corner_kwargs[
                    "truths"
                ] = truths

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
