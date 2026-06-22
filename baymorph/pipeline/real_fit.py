from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np

from .config import load_resolved_config
from .outputs import make_output_tree, save_resolved_config, pipeline_paths

from .psf_construction import (
    load_empirical_psf,
)

from baymorph.fitting.factories import (
    make_single_sersic_psf_sky_model,
    make_double_sersic_psf_sky_model,
)

from .image_preparation import (
    load_image_preparation_products,
)

from baymorph.fitting.parameters import (
    Parameter,
    ParameterSet,
)

from baymorph.fitting import DynestyFitter

@dataclass
class RealFitProducts:
    posterior: object | None

    best_parameters: dict[str, float]

    best_model_image: np.ndarray

    residual: np.ndarray

    norm_residual: np.ndarray

    chi2: float

    ndof: int

    reduced_chi2: float

    
def run_real_fit_from_config(
    config: str | Path | dict[str, Any],
):
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = config

    make_output_tree(resolved)
    save_resolved_config(resolved)

    try:
        image_products = load_image_preparation_products(
            resolved,
        )

        psf = load_empirical_psf(
            resolved,
        )

    except FileNotFoundError:
        return resolved

    fit_cfg = resolved["fit"]

    sampler_cfg = fit_cfg["sampler"]

    all_products = {}

    for model_name in resolved["fit"]["models"]:

        parameter_names, parameters, _, _ = (
            _build_parameter_set(
                model_name,
                data_fit=image_products.prepared_data,
                noise_fit=image_products.noise,
                mask_fit=image_products.contaminant_mask,
                target_x=image_products.target_x,
                target_y=image_products.target_y,
            )
        )

        model_factory = _make_model_factory(
            model_name,
            psf,
        )

        result = _run_dynesty(
            data_fit=image_products.prepared_data,
            noise_fit=image_products.noise,
            mask_fit=image_products.contaminant_mask,
            parameters=parameters,
            model_factory=model_factory,
            nlive=sampler_cfg["nlive"],
            sample=sampler_cfg.get(
                "sample",
                "auto",
            ),
        )

        best_model_image, best_parameters = _compute_best_model(
            result=result,
            parameter_names=parameter_names,
            model_factory=model_factory,
            image_shape=image_products.prepared_data.shape,
        )

        stats = _compute_fit_statistics(
            data=image_products.prepared_data,
            model_image=best_model_image,
            noise=image_products.noise,
            mask=image_products.contaminant_mask,
            n_parameters=len(parameter_names),
        )

        products = RealFitProducts(
            posterior=result,
            best_parameters=best_parameters,
            best_model_image=best_model_image,
            residual=stats["residual"],
            norm_residual=stats["norm_residual"],
            chi2=stats["chi2"],
            ndof=stats["ndof"],
            reduced_chi2=stats["reduced_chi2"],
        )

        save_real_fit_products(
            resolved,
            model_name,
            products,
        )

        all_products[model_name] = products

    if len(all_products) > 1:
        comparison = _compute_bayesian_model_comparison(
            all_products,
        )

        save_bayesian_model_comparison(
            resolved,
            comparison,
    )

    return all_products

def save_real_fit_products(
    config: dict[str, Any],
    model_name: str,
    products: RealFitProducts,
) -> None:
    paths = pipeline_paths(config)["fits"][model_name]

    paths["dir"].mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(paths["dir"] / "best_model_image.npy", products.best_model_image)
    np.save(paths["dir"] / "residual.npy", products.residual)
    np.save(paths["dir"] / "norm_residual.npy", products.norm_residual)

    with open(paths["dir"] / "best_parameters.json", "w") as f:
        json.dump(
            {
                key: float(value)
                for key, value in products.best_parameters.items()
            },
            f,
            indent=2,
        )

    summary = {
        "chi2": float(products.chi2),
        "ndof": int(products.ndof),
        "reduced_chi2": float(products.reduced_chi2),
        "logz": (
            None
            if products.posterior is None
            else float(products.posterior.logz[-1])
        ),
    }

    with open(paths["dir"] / "fit_summary.json", "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )


def save_bayesian_model_comparison(
    config: dict[str, Any],
    comparison: dict[str, Any],
) -> None:
    paths = pipeline_paths(config)["model_comparison"]

    paths["dir"].mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(paths["bayes_factor"], "w") as f:
        json.dump(
            comparison,
            f,
            indent=2,
        )


def load_real_fit_products(
    config: str | Path | dict[str, Any],
    model_name: str,
) -> RealFitProducts:
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = config

    paths = pipeline_paths(resolved)["fits"][model_name]

    best_model_image = np.load(paths["dir"] / "best_model_image.npy")
    residual = np.load(paths["dir"] / "residual.npy")
    norm_residual = np.load(paths["dir"] / "norm_residual.npy")

    with open(paths["dir"] / "best_parameters.json", "r") as f:
        best_parameters = json.load(f)

    with open(paths["dir"] / "fit_summary.json", "r") as f:
        summary = json.load(f)

    return RealFitProducts(
        posterior=None,
        best_parameters={
            key: float(value)
            for key, value in best_parameters.items()
        },
        best_model_image=best_model_image,
        residual=residual,
        norm_residual=norm_residual,
        chi2=float(summary["chi2"]),
        ndof=int(summary["ndof"]),
        reduced_chi2=float(summary["reduced_chi2"]),
    )


def _build_parameter_set(
    model_name: str,
    data_fit: np.ndarray,
    noise_fit: np.ndarray,
    mask_fit: np.ndarray,
    target_x: float,
    target_y: float,
):
    valid_for_flux = (
        (~mask_fit)
        & np.isfinite(data_fit)
        & np.isfinite(noise_fit)
        & (noise_fit > 0)
    )

    flux_total = np.nansum(
        np.clip(
            data_fit[valid_for_flux],
            0.0,
            None,
        )
    )

    sky_guess = np.nanmedian(
        data_fit[valid_for_flux],
    )

    if model_name == "single_sersic":
        parameter_names = [
            "flux",
            "re",
            "n",
            "q",
            "theta",
            "x0",
            "y0",
            "sky",
        ]

        parameters = ParameterSet(
            [
                Parameter("flux", 0.0, 2.0 * flux_total),
                Parameter("re", 1.0, 120.0),
                Parameter("n", 0.5, 6.0),
                Parameter("q", 0.4, 1.0),
                Parameter("theta", -np.pi / 2, np.pi / 2),
                Parameter("x0", target_x - 8, target_x + 8),
                Parameter("y0", target_y - 8, target_y + 8),
                Parameter("sky", -50.0, 50.0),
            ]
        )

    elif model_name == "double_sersic":

        parameter_names = [
            "flux_bulge",
            "re_bulge",
            "n_bulge",
            "flux_disk",
            "re_disk",
            "n_disk",
            "q",
            "theta",
            "x0",
            "y0",
            "sky",
        ]

        parameters = ParameterSet(
            [
                Parameter(
                    "flux_bulge",
                    0.0,
                    1.0 * flux_total,
                ),
                Parameter(
                    "re_bulge",
                    1.0,
                    30.0,
                ),
                Parameter(
                    "n_bulge",
                    1.5,
                    6.0,
                ),
                Parameter(
                    "flux_disk",
                    0.0,
                    2.0 * flux_total,
                ),
                Parameter(
                    "re_disk",
                    10.0,
                    120.0,
                ),
                Parameter(
                    "n_disk",
                    0.5,
                    3.0,
                ),
                Parameter(
                    "q",
                    0.4,
                    1.0,
                ),
                Parameter(
                    "theta",
                    -np.pi / 2,
                    np.pi / 2,
                ),
                Parameter(
                    "x0",
                    target_x - 8,
                    target_x + 8,
                ),
                Parameter(
                    "y0",
                    target_y - 8,
                    target_y + 8,
                ),
                Parameter(
                    "sky",
                    -50.0,
                    50.0,
                ),
            ]
        )

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    return (
        parameter_names,
        parameters,
        flux_total,
        sky_guess,
    )


def _make_model_factory(
    model_name,
    psf,
):
    if model_name == "single_sersic":

        def model_factory(
            flux,
            re,
            n,
            q,
            theta,
            x0,
            y0,
            sky,
        ):
            return make_single_sersic_psf_sky_model(
                flux,
                re,
                n,
                q,
                theta,
                x0,
                y0,
                sky,
                psf=psf,
            )

    elif model_name == "double_sersic":

        def model_factory(
            flux_bulge,
            re_bulge,
            n_bulge,
            flux_disk,
            re_disk,
            n_disk,
            q,
            theta,
            x0,
            y0,
            sky,
        ):
            return make_double_sersic_psf_sky_model(
                flux_bulge,
                re_bulge,
                n_bulge,
                flux_disk,
                re_disk,
                n_disk,
                q,
                theta,
                x0,
                y0,
                sky,
                psf=psf,
            )

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    return model_factory

def _compute_best_model(
    result,
    parameter_names,
    model_factory,
    image_shape,
):
    best_parameters = {
        name: result.best_parameters[name]
        for name in parameter_names
    }

    best_values = [
        best_parameters[name]
        for name in parameter_names
    ]

    best_model = model_factory(*best_values)

    best_model_image = best_model.render(image_shape)

    return best_model_image, best_parameters


def _compute_fit_statistics(
    data,
    model_image,
    noise,
    mask,
    n_parameters: int,
):
    residual = data - model_image

    valid = (
        (~mask)
        & np.isfinite(data)
        & np.isfinite(model_image)
        & np.isfinite(noise)
        & (noise > 0)
    )

    norm_residual = np.full_like(
        residual,
        np.nan,
        dtype=float,
    )

    norm_residual[valid] = (
        residual[valid]
        / noise[valid]
    )

    chi2 = float(
        np.sum(
            norm_residual[valid] ** 2
        )
    )

    ndof = int(
        np.sum(valid)
        - n_parameters
    )

    reduced_chi2 = (
        float(chi2 / ndof)
        if ndof > 0
        else np.nan
    )

    return {
        "residual": residual,
        "norm_residual": norm_residual,
        "chi2": chi2,
        "ndof": ndof,
        "reduced_chi2": reduced_chi2,
    }


def _compute_bayesian_model_comparison(
    products_by_model,
):
    logz = {
        model_name: float(
            products.posterior.logz[-1]
        )
        for model_name, products in products_by_model.items()
    }

    if (
        "single_sersic" not in logz
        or "double_sersic" not in logz
    ):
        return {
            "logz": logz,
            "delta_logz": {},
            "bayes_factor": {},
        }

    delta_logz = (
        logz["double_sersic"]
        - logz["single_sersic"]
    )

    bayes_factor = float(
        np.exp(delta_logz)
    )

    return {
        "logz": logz,
        "delta_logz": {
            "double_sersic_minus_single_sersic": delta_logz,
        },
        "bayes_factor": {
            "double_sersic_over_single_sersic": bayes_factor,
        },
    }


def _run_dynesty(
    data_fit,
    noise_fit,
    mask_fit,
    parameters,
    model_factory,
    *,
    nlive: int = 300,
    sample: str = "auto",
):
    fitter = DynestyFitter(
        model_factory=model_factory,
        data=data_fit,
        noise=noise_fit,
        parameters=parameters,
        mask=mask_fit,
    )

    return fitter.run(
        nlive=nlive,
        sample=sample,
        print_progress=True,
    )