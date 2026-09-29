"""Synthetic-observation pipeline workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from baymorph.pipeline.config import (
    load_resolved_config,
    resolve_config,
)
from baymorph.pipeline.outputs import (
    make_output_tree,
    save_resolved_config,
    simulation_truth_path,
)
from baymorph.pipeline.psf_construction import (
    save_simulated_psf_products,
)
from baymorph.simulation import (
    SyntheticObservation,
    make_observation_from_config,
)


def _simulation_psf_config(
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the configured simulated PSF definition.

    Parameters
    ----------
    config : dict
        Resolved BayMorph configuration.

    Returns
    -------
    dict or None
        Simulated PSF configuration, or None when no PSF is configured.
    """
    simulation_cfg = config.get(
        "simulation",
        {},
    )

    psf_cfg = simulation_cfg.get(
        "psf",
    )

    return psf_cfg


def _simulation_noise_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return normalized simulation-noise configuration.

    Parameters
    ----------
    config : dict
        Resolved BayMorph configuration.

    Returns
    -------
    dict
        Noise configuration using the keys expected by the simulation
        backend.
    """
    simulation_cfg = config.get(
        "simulation",
        {},
    )

    nested_noise = simulation_cfg.get(
        "noise",
        {},
    )

    return {
        "sky": float(
            nested_noise.get(
                "sky",
                0.0,
            )
        ),
        "poisson_noise": bool(
            nested_noise.get(
                "poisson_noise",
                True,
            )
        ),
        "read_noise_sigma": nested_noise.get(
            "read_noise",
        ),
    }


def _normalized_simulation_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return a simulation configuration accepted by the backend.

    The nested ``simulation.noise`` and ``simulation.psf`` sections are the
    only supported configuration surface; the flat keys consumed by the
    simulation backend are derived from them.

    Parameters
    ----------
    config : dict
        Resolved BayMorph configuration.

    Returns
    -------
    dict
        Shallow copy of the configuration with normalized simulation keys.
    """
    normalized = dict(
        config
    )

    simulation_cfg = dict(
        config["simulation"]
    )

    noise_cfg = _simulation_noise_config(
        config
    )

    psf_cfg = _simulation_psf_config(
        config
    )

    simulation_cfg["sky"] = noise_cfg["sky"]
    simulation_cfg["poisson_noise"] = noise_cfg["poisson_noise"]
    simulation_cfg["read_noise_sigma"] = noise_cfg[
        "read_noise_sigma"
    ]

    simulation_cfg["psf_fwhm"] = (
        None
        if psf_cfg is None
        else float(
            psf_cfg["fwhm"]
        )
    )

    normalized["simulation"] = simulation_cfg

    return normalized


def _simulation_truth_payload(
    observation: SyntheticObservation,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSON-serializable simulation truth payload.

    Parameters
    ----------
    observation : SyntheticObservation
        Generated synthetic observation.
    config : dict
        Resolved BayMorph configuration.

    Returns
    -------
    dict
        Simulation truth and generation metadata.
    """
    simulation_cfg = config["simulation"]
    psf_cfg = _simulation_psf_config(
        config
    )
    noise_cfg = _simulation_noise_config(
        config
    )

    return {
        "shape": [
            int(value)
            for value in observation.shape
        ],
        "seed": simulation_cfg.get(
            "seed"
        ),
        "truth": {
            key: float(value)
            for key, value in observation.truth.items()
        },
        "psf": psf_cfg,
        "noise": noise_cfg,
    }


def save_simulation_products(
    config: dict[str, Any],
    observation: SyntheticObservation,
) -> None:
    """Save a synthetic observation and its associated products.

    Parameters
    ----------
    config : dict
        Resolved BayMorph configuration.
    observation : SyntheticObservation
        Generated synthetic observation.

    Returns
    -------
    None
        Products are written to the configured output tree.
    """
    make_output_tree(
        config
    )

    overwrite = bool(
        config["outputs"].get(
            "overwrite",
            False,
        )
    )

    image_path = Path(
        config["data"]["image_path"]
    )

    image_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    observation.to_fits(
        image_path,
        overwrite=overwrite,
    )

    truth_path = simulation_truth_path(
        config
    )

    truth_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    truth_payload = _simulation_truth_payload(
        observation,
        config,
    )

    with truth_path.open(
        "w",
    ) as stream:
        json.dump(
            truth_payload,
            stream,
            indent=2,
        )

    psf_cfg = _simulation_psf_config(
        config
    )

    if (
        observation.has_psf
        and psf_cfg is not None
    ):
        psf_type = psf_cfg.get(
            "type",
            "gaussian",
        )

        if psf_type != "gaussian":
            raise ValueError(
                "Synthetic PSF product saving currently supports only "
                f"Gaussian PSFs, got '{psf_type}'."
            )

        kernel_size = int(
            psf_cfg.get(
                "size",
                observation.psf_kernel.shape[0],
            )
        )

        save_simulated_psf_products(
            config,
            fwhm=float(
                psf_cfg["fwhm"]
            ),
            size=kernel_size,
        )

    save_resolved_config(
        config
    )


def run_simulation_from_config(
    config: str | Path | dict[str, Any],
) -> SyntheticObservation:
    """Generate and save a synthetic BayMorph observation.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        User configuration dictionary or path to a YAML configuration file.

    Returns
    -------
    SyntheticObservation
        Generated synthetic observation, including intrinsic, convolved,
        sky-added, and final noisy images.
    """
    if isinstance(
        config,
        (str, Path),
    ):
        resolved = load_resolved_config(
            config
        )
    else:
        resolved = resolve_config(
            config
        )

    normalized = _normalized_simulation_config(
        resolved
    )

    observation = make_observation_from_config(
        normalized
    )

    save_simulation_products(
        resolved,
        observation,
    )

    return observation