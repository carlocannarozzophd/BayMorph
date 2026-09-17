"""Residual-map diagnostic workflow."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from baymorph.pipeline.config import (
    load_config,
    load_resolved_config,
    resolve_config,
)
from baymorph.pipeline.image_preparation import (
    load_image_preparation_products,
)
from baymorph.pipeline.multi_observation_config import (
    resolve_multi_observation_config,
)
from baymorph.pipeline.multi_observation_outputs import (
    multi_observation_paths,
)
from baymorph.pipeline.outputs import (
    pipeline_paths,
)
from baymorph.pipeline.psf_construction import (
    load_empirical_psf,
)
from baymorph.pipeline.real_fit import (
    _fit_diagnostic_title as _diagnostic_title,
)
from baymorph.plotting.images import (
    plot_pipeline_fit_maps,
)
from baymorph.simulation.psf import GaussianPSF


def _psf_to_image(
    psf,
    config=None,
):
    """Return a displayable PSF image from a PSF object, array, or config."""
    if psf is not None:
        kernel = getattr(
            psf,
            "kernel",
            None,
        )

        if callable(
            kernel
        ):
            return kernel()

        if kernel is not None:
            return kernel

        return psf

    if config is None:
        return None

    psf_cfg = config.get(
        "psf",
        {},
    )

    if not psf_cfg:
        psf_cfg = (
            config
            .get(
                "simulation",
                {},
            )
            .get(
                "psf",
                {},
            )
        )

    psf_type = psf_cfg.get(
        "type"
    )

    if psf_type == "gaussian":
        gaussian_cfg = psf_cfg.get(
            "gaussian",
            {},
        )
        fwhm = gaussian_cfg.get(
            "fwhm",
            psf_cfg.get(
                "fwhm"
            ),
        )

        if fwhm is None:
            return None

        return GaussianPSF(
            fwhm=float(
                fwhm
            ),
        ).kernel()

    return None


def _load_display_psf(
    config,
):
    """Return the saved empirical PSF for display, if the fit used one.

    Analytic PSFs are rebuilt from the configuration by ``_psf_to_image``.
    """
    psf_type = config.get(
        "psf",
        {},
    ).get(
        "type",
        "empirical",
    )

    if psf_type != "empirical":
        return None

    try:
        return load_empirical_psf(
            config,
        )
    except FileNotFoundError:
        return None


def _resolve_residual_config(
    config: str | Path | Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    bool,
    Path,
]:
    """Resolve a diagnostic configuration and identify its workflow mode."""
    if isinstance(
        config,
        (str, Path),
    ):
        path = Path(
            config
        )

        raw_config = load_config(
            path
        )

        if "observations" in raw_config:
            return (
                raw_config,
                True,
                path.parent,
            )

        return (
            load_resolved_config(
                path
            ),
            False,
            path.parent,
        )

    if isinstance(
        config,
        Mapping,
    ):
        raw_config = dict(
            config
        )

        if "observations" in raw_config:
            return (
                raw_config,
                True,
                Path.cwd(),
            )

        return (
            resolve_config(
                raw_config
            ),
            False,
            Path.cwd(),
        )

    raise TypeError(
        "config must be a path or a mapping."
    )


def _normalize_fit_results(
    fit_results,
) -> dict[str, Any]:
    """Return fit products as a model-name mapping."""
    if isinstance(
        fit_results,
        dict,
    ):
        return fit_results

    return {
        fit_results.model_name: fit_results,
    }


def _multi_observation_title(
    observation_config: Mapping[str, Any],
    model_name: str,
    observation_name: str,
) -> str:
    """Build a residual diagnostic title for one observation."""
    base_title = _diagnostic_title(
        observation_config,
        model_name,
    )

    return (
        f"{base_title} | observation: "
        f"{observation_name}"
    )


def _single_observation_plot_products(
    image_products,
    fit,
) -> dict[str, Any]:
    """Return aligned products on the prepared observation footprint."""
    original_cutout = np.asarray(
        image_products.cutout
    )

    prepared_cutout = np.asarray(
        image_products.prepared_data
    )

    segmentation = np.asarray(
        image_products.segmentation
    )

    target_mask = np.asarray(
        image_products.target_segment
    )

    fitting_mask = np.asarray(
        image_products.fitting_mask,
        dtype=bool,
    )

    noise = getattr(
        fit,
        "noise",
        None,
    )

    if noise is None:
        noise = image_products.noise

    noise = np.asarray(
        noise,
        dtype=float,
    )

    expected_shape = prepared_cutout.shape

    for name, array in (
        (
            "best_model_image",
            fit.best_model_image,
        ),
        (
            "noise",
            noise,
        ),
        (
            "fitting_mask",
            fitting_mask,
        ),
        (
            "segmentation",
            segmentation,
        ),
        (
            "target_mask",
            target_mask,
        ),
        (
            "original_cutout",
            original_cutout,
        ),
    ):
        if np.asarray(
            array
        ).shape != expected_shape:
            raise ValueError(
                f"{name} shape does not match the prepared footprint."
            )

    return {
        "original_cutout": original_cutout,
        "prepared_cutout": prepared_cutout,
        "noise": noise,
        "segmentation": segmentation,
        "contaminant_mask": fitting_mask,
        "target_mask": target_mask,
        "cutout_bounds": image_products.cutout_bounds,
    }

def _run_single_observation_residual_maps(
    config: Mapping[str, Any],
    fit_results,
) -> None:
    """Generate residual-map diagnostics for a single observation."""
    image_products = load_image_preparation_products(
        config,
    )

    paths = pipeline_paths(
        config,
    )["fits"]

    psf = _load_display_psf(
        config,
    )

    fits = _normalize_fit_results(
        fit_results
    )

    for model_name, fit in fits.items():
        plot_products = _single_observation_plot_products(
            image_products,
            fit,
        )

        fig, _ = plot_pipeline_fit_maps(
            original_cutout=plot_products[
                "original_cutout"
            ],
            prepared_cutout=plot_products[
                "prepared_cutout"
            ],
            model=fit.best_model_image,
            noise=plot_products[
                "noise"
            ],
            segmentation=plot_products[
                "segmentation"
            ],
            psf=_psf_to_image(
                psf,
                config,
            ),
            contaminant_mask=plot_products[
                "contaminant_mask"
            ],
            target_mask=plot_products[
                "target_mask"
            ],
            original_image=image_products.original_image,
            cutout_bounds=plot_products[
                "cutout_bounds"
            ],
            title=_diagnostic_title(
                config,
                model_name,
            ),
            show=False,
        )

        output_path = paths[
            model_name
        ][
            "residual"
        ]

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            bbox_inches="tight",
        )

        plt.close(
            fig,
        )


def _run_multi_observation_residual_maps(
    config: Mapping[str, Any],
    fit_results,
    *,
    base_dir: Path,
) -> None:
    """Generate one residual-map diagnostic for each observation."""
    observation_configs, _ = resolve_multi_observation_config(
        config,
        base_dir=base_dir,
    )

    paths = multi_observation_paths(
        config
    )[
        "fits"
    ]

    fits = _normalize_fit_results(
        fit_results
    )

    for model_name, fit in fits.items():
        if not hasattr(
            fit,
            "observations",
        ):
            raise TypeError(
                "Multi-observation residual diagnostics require fit "
                "products exposing an 'observations' mapping."
            )

        fit_observations = fit.observations

        if not isinstance(
            fit_observations,
            Mapping,
        ):
            raise TypeError(
                "Multi-observation fit products must expose observations "
                "as a mapping."
            )

        missing_observations = [
            observation_name
            for observation_name in observation_configs
            if observation_name not in fit_observations
        ]

        if missing_observations:
            missing_text = ", ".join(
                repr(
                    observation_name
                )
                for observation_name in missing_observations
            )

            raise ValueError(
                "Fit products are missing observation products for "
                f"{missing_text}."
            )

        observation_paths = paths[
            model_name
        ][
            "observations"
        ]

        for (
            observation_name,
            observation_config,
        ) in observation_configs.items():
            image_products = load_image_preparation_products(
                observation_config
            )

            observation_fit = fit_observations[
                observation_name
            ]

            psf = _load_display_psf(
                observation_config
            )

            fig, _ = plot_pipeline_fit_maps(
                original_cutout=image_products.cutout,
                prepared_cutout=image_products.prepared_data,
                model=observation_fit.best_model_image,
                noise=observation_fit.noise,
                segmentation=image_products.segmentation,
                psf=_psf_to_image(
                    psf,
                    observation_config,
                ),
                contaminant_mask=image_products.fitting_mask,
                target_mask=image_products.target_segment,
                original_image=image_products.original_image,
                cutout_bounds=image_products.cutout_bounds,
                title=_multi_observation_title(
                    observation_config,
                    model_name,
                    observation_name,
                ),
                show=False,
            )

            output_path = observation_paths[
                observation_name
            ][
                "residual_plot"
            ]

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            fig.savefig(
                output_path,
                bbox_inches="tight",
            )

            plt.close(
                fig,
            )


def run_residual_map_diagnostics(
    config,
    fit_results,
):
    """Generate and save pipeline residual-map diagnostics.

    Single-observation analyses use exactly the image footprint employed by
    the likelihood. Multi-observation analyses generate the established
    diagnostic independently on every native observation grid while using
    model products from the single joint posterior fit.

    Parameters
    ----------
    config : dict, str, or pathlib.Path
        Pipeline or joint multi-observation configuration.
    fit_results : RealFitProducts, MultiObservationFitProducts, or dict
        Fit products associated with the configuration.

    Returns
    -------
    None
        Residual-map figures are written to the configured fitting-output
        directories.
    """
    (
        resolved_config,
        is_multi_observation,
        base_dir,
    ) = _resolve_residual_config(
        config
    )

    if is_multi_observation:
        _run_multi_observation_residual_maps(
            resolved_config,
            fit_results,
            base_dir=base_dir,
        )
        return

    _run_single_observation_residual_maps(
        resolved_config,
        fit_results,
    )
