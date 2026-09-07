"""Persistence for multi-observation fit products."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from baymorph.pipeline.multi_observation_outputs import (
    make_multi_observation_output_tree,
    multi_observation_paths,
)
from baymorph.results import MultiObservationFitProducts
from baymorph.results.inference import json_compatible_metadata


def save_multi_observation_fit_products(
    config: Mapping[str, Any],
    products: MultiObservationFitProducts,
) -> None:
    """Persist one joint multi-observation fit.

    Parameters
    ----------
    config : mapping
        Joint multi-observation configuration containing the output namespace.
    products : MultiObservationFitProducts
        Joint posterior and observation-specific pixel-space products.
    """
    make_multi_observation_output_tree(
        config
    )

    paths = multi_observation_paths(
        config
    )["fits"][products.model_name]

    products.posterior.save(
        paths["posterior"]
    )

    with paths["best_parameters"].open(
        "w"
    ) as stream:
        json.dump(
            {
                name: float(value)
                for name, value in products.best_parameters.items()
            },
            stream,
            indent=2,
        )

    with paths["fit_summary"].open(
        "w"
    ) as stream:
        json.dump(
            _joint_summary(
                products
            ),
            stream,
            indent=2,
        )

    for (
        observation_name,
        observation_products,
    ) in products.observations.items():
        observation_paths = paths[
            "observations"
        ][observation_name]

        np.save(
            observation_paths[
                "best_model_image"
            ],
            observation_products.best_model_image,
        )

        np.save(
            observation_paths[
                "residual"
            ],
            observation_products.residual,
        )

        np.save(
            observation_paths[
                "norm_residual"
            ],
            observation_products.norm_residual,
        )

        with observation_paths[
            "summary"
        ].open(
            "w"
        ) as stream:
            json.dump(
                {
                    "parameters": {
                        name: float(value)
                        for name, value
                        in observation_products.parameters.items()
                    },
                    "chi2": float(
                        observation_products.chi2
                    ),
                    "n_valid_pixels": int(
                        observation_products.n_valid_pixels
                    ),
                    "likelihood": (
                        None
                        if observation_products.likelihood_metadata is None
                        else json_compatible_metadata(
                            observation_products.likelihood_metadata
                        )
                    ),
                },
                stream,
                indent=2,
            )


def _joint_summary(
    products: MultiObservationFitProducts,
) -> dict[str, Any]:
    """Return a JSON-compatible joint fit summary."""
    return {
        "model": products.model_name,
        "chi2": float(
            products.chi2
        ),
        "ndof": int(
            products.ndof
        ),
        "reduced_chi2": _finite_or_none(
            products.reduced_chi2
        ),
        "logz": _optional_float(
            products.logz
        ),
        "logzerr": _optional_float(
            products.logz_error
        ),
        "observations": {
            name: {
                "chi2": float(
                    observation_products.chi2
                ),
                "n_valid_pixels": int(
                    observation_products.n_valid_pixels
                ),
                "likelihood": (
                    None
                    if observation_products.likelihood_metadata is None
                    else json_compatible_metadata(
                        observation_products.likelihood_metadata
                    )
                ),
            }
            for name, observation_products
            in products.observations.items()
        },
    }


def _optional_float(
    value: Any,
) -> float | None:
    """Convert an optional scalar to a finite JSON float or None."""
    if value is None:
        return None

    return _finite_or_none(
        value
    )


def _finite_or_none(
    value: Any,
) -> float | None:
    """Convert a scalar to a finite JSON float or None."""
    scalar = float(
        value
    )

    if not np.isfinite(
        scalar
    ):
        return None

    return scalar
