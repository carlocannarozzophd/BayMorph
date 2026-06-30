from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import matplotlib.pyplot as plt

from .config import (
    load_resolved_config,
    resolve_config,
    DEFAULT_CONFIG,
)

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

from baymorph.simulation.psf import GaussianPSF

from baymorph.plotting import (
    corner_plot,
    plot_log_data_model_residual,
)

from baymorph.analysis import (
    radial_profile_from_image,
    compute_posterior_profile_bands,
)

from baymorph.models import (
    get_model,
    parameter_labels as model_parameter_labels,
)

from baymorph.results import RealFitProducts


def run_real_fit_from_config(
    config: str | Path | dict[str, Any],
):
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = resolve_config(config)

    make_output_tree(resolved)
    save_resolved_config(resolved)

    try:
        image_products = load_image_preparation_products(
            resolved,
        )

        psf = _load_psf_from_config(
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
                prior_cfg=resolved["fit"]["priors"],
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
            dlogz=sampler_cfg.get(
                "dlogz",
                None,
            ),
            n_processes=sampler_cfg.get(
                "n_processes",
                1,
            ),
            queue_size=sampler_cfg.get(
                "queue_size",
                None,
            ),
            seed=sampler_cfg.get(
                "seed",
                None,
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
            model_name=model_name,
            posterior=result,
            best_parameters=best_parameters,
            best_model_image=best_model_image,
            residual=stats["residual"],
            norm_residual=stats["norm_residual"],
            chi2=stats["chi2"],
            ndof=stats["ndof"],
            reduced_chi2=stats["reduced_chi2"],
            noise=image_products.noise,
        )

        save_real_fit_products(
            resolved,
            model_name,
            products,
        )

        all_products[model_name] = products

    if len(all_products) == 1:
        return next(iter(all_products.values()))

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

    diagnostics_cfg = config.get(
        "diagnostics",
        {},
    )

    image_products = None

    if diagnostics_cfg.get(
        "residual",
        True,
    ):
        image_products = load_image_preparation_products(
            config,
        )

        fig, _ = plot_log_data_model_residual(
            image_products.prepared_data,
            products.best_model_image,
            noise=image_products.noise,
            title=_diagnostic_title(
                config,
                model_name,
            ),
            show=False,
        )

        fig.savefig(
            paths["residual"],
            bbox_inches="tight",
        )

        plt.close(fig)

    if diagnostics_cfg.get(
        "radial_profile",
        True,
    ):
        if image_products is None:
            image_products = load_image_preparation_products(
                config,
            )

        max_radius = 0.5 * min(
            image_products.prepared_data.shape
        )

        radii = np.arange(
            0.0,
            max_radius + 1.0,
            1.0,
        )

        radius = 0.5 * (
            radii[:-1]
            + radii[1:]
        )

        profile_data = radial_profile_from_image(
            image_products.prepared_data,
            center_x=products.best_parameters["x0"],
            center_y=products.best_parameters["y0"],
            radii=radii,
            mask=image_products.contaminant_mask,
        )

        profile_model = radial_profile_from_image(
            products.best_model_image,
            center_x=products.best_parameters["x0"],
            center_y=products.best_parameters["y0"],
            radii=radii,
            mask=image_products.contaminant_mask,
        )

        profile_noise = radial_profile_from_image(
            image_products.noise,
            center_x=products.best_parameters["x0"],
            center_y=products.best_parameters["y0"],
            radii=radii,
            mask=image_products.contaminant_mask,
        )

        products.radius = radius
        products.profile_data = profile_data
        products.profile_model = profile_model
        products.profile_noise = profile_noise

        residual_sigma_profile = (
            profile_data
            - profile_model
        ) / profile_noise

        valid = (
            np.isfinite(profile_data)
            & np.isfinite(profile_model)
            & (profile_data > 0)
            & (profile_model > 0)
        )

        log_data = np.full_like(
            profile_data,
            np.nan,
        )

        log_model = np.full_like(
            profile_model,
            np.nan,
        )

        log_data[valid] = np.log10(
            profile_data[valid]
        )

        log_model[valid] = np.log10(
            profile_model[valid]
        )

        profile_bands = None

        if (
            diagnostics_cfg.get(
                "posterior_profile_bands",
                True,
            )
            and model_name == "double_sersic"
            and products.posterior is not None
        ):
            profile_bands = compute_posterior_profile_bands(
                products.posterior.posterior_samples,
                shape=image_products.prepared_data.shape,
                radii=radii,
                mask=image_products.contaminant_mask,
                psf=_load_psf_from_config(
                    config,
                ),
                include_sky=True,
                profile_center=(
                    products.best_parameters["x0"],
                    products.best_parameters["y0"],
                ),
            )

        fig = plt.figure(
            figsize=(15, 15),
        )

        gs = fig.add_gridspec(
            5,
            1,
            hspace=0.0,
        )

        ax_main = fig.add_subplot(
            gs[:4, 0],
        )

        ax_res = fig.add_subplot(
            gs[4, 0],
            sharex=ax_main,
        )

        ax_main.plot(
            radius,
            log_data,
            "o",
            ms=4,
            label="Data",
        )

        if profile_bands is not None:
            total_band = profile_bands["total"]

            band_valid = (
                np.isfinite(total_band["p16"])
                & np.isfinite(total_band["p50"])
                & np.isfinite(total_band["p84"])
                & (total_band["p16"] > 0)
                & (total_band["p50"] > 0)
                & (total_band["p84"] > 0)
            )

            log_p16 = np.full_like(
                total_band["p16"],
                np.nan,
            )

            log_p50 = np.full_like(
                total_band["p50"],
                np.nan,
            )

            log_p84 = np.full_like(
                total_band["p84"],
                np.nan,
            )

            log_p16[band_valid] = np.log10(
                total_band["p16"][band_valid]
            )

            log_p50[band_valid] = np.log10(
                total_band["p50"][band_valid]
            )

            log_p84[band_valid] = np.log10(
                total_band["p84"][band_valid]
            )

            ax_main.fill_between(
                radius,
                log_p16,
                log_p84,
                alpha=0.25,
                label="68% posterior band",
            )

            ax_main.plot(
                radius,
                log_p50,
                lw=2,
                label="Posterior median",
            )

        ax_main.plot(
            radius,
            log_model,
            lw=2,
            label="Best model",
        )

        ax_main.set_ylabel(
            r"$\log_{10}(I)$",
        )

        ax_main.set_title(
            _diagnostic_title(
                config,
                model_name,
            ),
        )

        ax_main.legend(
            frameon=False,
        )

        ax_res.axhline(
            0.0,
            ls="--",
            lw=1,
        )

        for sigma in [1.0, 2.0, 3.0]:
            ax_res.axhline(
                sigma,
                ls=":",
                lw=0.8,
            )

            ax_res.axhline(
                -sigma,
                ls=":",
                lw=0.8,
            )

        ax_res.plot(
            radius,
            residual_sigma_profile,
            lw=1.5,
        )

        ax_res.set_ylim(
            -5.0,
            5.0,
        )

        ax_res.set_xlabel(
            "Radius [pix]",
        )

        ax_res.set_ylabel(
            "Residual [$\\sigma$]",
        )

        fig.savefig(
            paths["radial_profile"],
            bbox_inches="tight",
        )

        plt.close(fig)

    if (
        diagnostics_cfg.get(
            "corner",
            True,
        )
        and products.posterior is not None
    ):
        fig, _, _ = corner_plot(
            products.posterior.posterior_samples.T,
            parameter_labels=model_parameter_labels(
                model_name,
            ),
        )

        fig.savefig(
            paths["corner"],
            bbox_inches="tight",
        )

        plt.close(fig)

    with open(paths["dir"] / "best_parameters.json", "w") as f:
        json.dump(
            {
                key: float(value)
                for key, value in products.best_parameters.items()
            },
            f,
            indent=2,
        )

    sampler_cfg = config["fit"]["sampler"]

    summary = {
        "chi2": float(products.chi2),
        "ndof": int(products.ndof),
        "reduced_chi2": float(products.reduced_chi2),
        "logz": (
            None
            if products.posterior is None
            else float(products.posterior.logz[-1])
        ),
        "logzerr": (
            None
            if (
                products.posterior is None
                or products.posterior.logzerr is None
            )
            else float(
                products.posterior.logzerr[-1]
            )
        ),
        "sampler": {
            "name": sampler_cfg.get(
                "name",
                "dynesty",
            ),
            "nlive": sampler_cfg.get(
                "nlive",
            ),
            "sample": sampler_cfg.get(
                "sample",
            ),
            "dlogz": sampler_cfg.get(
                "dlogz",
            ),
            "seed": sampler_cfg.get(
                "seed",
            ),
            "n_processes": sampler_cfg.get(
                "n_processes",
                1,
            ),
            "queue_size": sampler_cfg.get(
                "queue_size",
            ),
        },
    }

    with open(paths["dir"] / "fit_summary.json", "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    save_fit_summary_text(
        paths["dir"] / "fit_summary.txt",
        model_title=_diagnostic_title(
            config,
            model_name,
        ),
        products=products,
    )


def save_fit_summary_text(
    path: Path,
    *,
    model_title: str,
    products: RealFitProducts,
) -> None:
    logz = (
        None
        if products.posterior is None
        else float(products.posterior.logz[-1])
    )

    logzerr = (
        None
        if (
            products.posterior is None
            or products.posterior.logzerr is None
        )
        else float(products.posterior.logzerr[-1])
    )

    lines = [
        f"Model: {model_title}",
        "",
        "Fit statistics",
        f"chi2 = {products.chi2:.8g}",
        f"ndof = {products.ndof:d}",
        f"reduced_chi2 = {products.reduced_chi2:.8g}",
        f"logZ = {logz:.8g}" if logz is not None else "logZ = None",
        f"logZerr = {logzerr:.8g}" if logzerr is not None else "logZerr = None",
        "",
        "Best-fit parameters",
    ]

    for key, value in products.best_parameters.items():
        lines.append(
            f"{key} = {float(value):.8g}"
        )

    path.write_text(
        "\n".join(lines) + "\n"
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
        resolved = resolve_config(config)

    paths = pipeline_paths(resolved)["fits"][model_name]

    best_model_image = np.load(paths["dir"] / "best_model_image.npy")
    residual = np.load(paths["dir"] / "residual.npy")
    norm_residual = np.load(paths["dir"] / "norm_residual.npy")

    with open(paths["dir"] / "best_parameters.json", "r") as f:
        best_parameters = json.load(f)

    with open(paths["dir"] / "fit_summary.json", "r") as f:
        summary = json.load(f)

    image_products = load_image_preparation_products(resolved)

    return RealFitProducts(
        model_name=model_name,
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
        noise=image_products.noise,
    )


def _load_psf_from_config(
    config: dict[str, Any],
):
    psf_cfg = config.get(
        "psf",
        {},
    )

    psf_type = psf_cfg.get(
        "type",
        "empirical",
    )

    if psf_type == "empirical":
        return load_empirical_psf(
            config,
        )

    if psf_type == "gaussian":
        gaussian_cfg = psf_cfg.get(
            "gaussian",
            {},
        )

        return GaussianPSF(
            fwhm=float(
                gaussian_cfg["fwhm"]
            )
        )

    raise ValueError(
        f"Unsupported PSF type: {psf_type}"
    )


def _parse_prior_value(value):
    if isinstance(value, str):
        if value == "pi/2":
            return float(np.pi / 2)

        if value == "-pi/2":
            return float(-np.pi / 2)

    return float(value)


def _bounds_from_prior(
    name: str,
    cfg: dict,
    *,
    flux_total: float,
    target_x: float,
    target_y: float,
):
    if "min_factor" in cfg or "max_factor" in cfg:
        return (
            float(cfg["min_factor"]) * flux_total,
            float(cfg["max_factor"]) * flux_total,
        )

    if "offset" in cfg:
        offset = float(cfg["offset"])

        if name == "x0":
            return (
                float(target_x - offset),
                float(target_x + offset),
            )

        if name == "y0":
            return (
                float(target_y - offset),
                float(target_y + offset),
            )

        raise ValueError(
            f"Offset prior is only supported for x0/y0, got {name}."
        )

    return (
        _parse_prior_value(cfg["min"]),
        _parse_prior_value(cfg["max"]),
    )


def _build_parameter_list(
    prior_cfg: dict,
    *,
    flux_total: float,
    target_x: float,
    target_y: float,
) -> ParameterSet:
    return ParameterSet(
        [
            Parameter(
                name,
                *_bounds_from_prior(
                    name,
                    cfg,
                    flux_total=flux_total,
                    target_x=target_x,
                    target_y=target_y,
                ),
            )
            for name, cfg in prior_cfg.items()
        ]
    )


def _build_parameter_set(
    model_name: str,
    data_fit: np.ndarray,
    noise_fit: np.ndarray,
    mask_fit: np.ndarray,
    target_x: float,
    target_y: float,
    prior_cfg=None,
):
    
    if prior_cfg is None:
        prior_cfg = DEFAULT_CONFIG["fit"]["priors"]

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

    model_prior_cfg = prior_cfg[model_name]

    model_definition = get_model(model_name)
    parameter_names = model_definition.parameter_names

    parameters = _build_parameter_list(
    model_prior_cfg,
    flux_total=flux_total,
    target_x=target_x,
    target_y=target_y,
    )

    return (
        parameter_names,
        parameters,
        flux_total,
        sky_guess,
    )


class SingleSersicModelFactory:
    def __init__(self, psf):
        self.psf = psf

    def __call__(
        self,
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
            psf=self.psf,
        )


class DoubleSersicModelFactory:
    def __init__(self, psf):
        self.psf = psf

    def __call__(
        self,
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
            psf=self.psf,
        )


def _make_model_factory(
    model_name,
    psf,
):
    model_definition = get_model(model_name)

    factories = {
        "SingleSersicModelFactory": SingleSersicModelFactory,
        "DoubleSersicModelFactory": DoubleSersicModelFactory,
    }

    try:
        factory_class = factories[model_definition.factory_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model factory: {model_definition.factory_name}"
        ) from exc

    return factory_class(psf)

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
            "log_bayes_factor": {},
            "bayes_factor": {},
        }

    delta_logz = (
        logz["double_sersic"]
        - logz["single_sersic"]
    )

    log_bayes_factor = float(delta_logz)

    if log_bayes_factor < 700:
        bayes_factor = float(
            np.exp(log_bayes_factor)
        )
    else:
        bayes_factor = None

    return {
        "logz": logz,
        "delta_logz": {
            "double_sersic_minus_single_sersic": delta_logz,
        },
        "log_bayes_factor": {
            "double_sersic_over_single_sersic": log_bayes_factor,
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
    dlogz: float | None = None,
    n_processes: int = 1,
    queue_size: int | None = None,
    seed: int | None = None,
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
        dlogz=dlogz,
        print_progress=True,
        n_processes=n_processes,
        queue_size=queue_size,
        seed=seed,
    )


def _diagnostic_title(
    config: dict[str, Any],
    model_name: str,
) -> str:
    diagnostics_cfg = config.get(
        "diagnostics",
        {},
    )

    titles = diagnostics_cfg.get(
        "titles",
        {},
    )

    if model_name in titles:
        return str(
            titles[model_name]
        )

    return (
        model_name
        .replace("_", " ")
        .title()
        .replace("Sersic", "Sersic")
    )
