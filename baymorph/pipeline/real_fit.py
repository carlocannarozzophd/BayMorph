"""Real-galaxy fitting workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from baymorph.core.observation import Observation
from baymorph.fitting import make_sampler
from baymorph.fitting.model_factories import (
    make_bulge_disk_psf_sky_model,
    make_double_sersic_psf_sky_model,
    make_sersic_point_source_psf_sky_model,
    make_single_sersic_psf_sky_model,
)
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.fitting.parameters import Parameter, ParameterSet
from baymorph.likelihood import make_likelihood
from baymorph.models import get_model
from baymorph.observation.background import (
    BackgroundAppliedModel,
    make_background_effect,
)
from baymorph.pipeline.config import (
    DEFAULT_CONFIG,
    load_resolved_config,
    resolve_config,
)
from baymorph.pipeline.image_preparation import load_image_preparation_products
from baymorph.pipeline.observation_metadata import observation_metadata_dict
from baymorph.pipeline.outputs import (
    make_output_tree,
    pipeline_paths,
    save_resolved_config,
)
from baymorph.pipeline.psf_construction import load_empirical_psf
from baymorph.results.inference import json_compatible_metadata
from baymorph.results import InferenceResult, RealFitProducts
from baymorph.simulation.psf import GaussianPSF


def run_real_fit_from_config(
    config: str | Path | dict[str, Any],
):
    """Run real-galaxy model fitting from a pipeline configuration."""
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
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Image-preparation products are required before fitting."
        ) from exc

    try:
        psf = _load_psf_from_config(
            resolved,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "PSF products are required before fitting."
        ) from exc

    try:
        observation = Observation.from_fits(
            resolved["data"]["image_path"]
        )
    except FileNotFoundError:
        observation = None

    likelihood_cfg = resolved["fit"].get(
        "likelihood",
        None,
    )
    sampler_cfg = resolved["fit"]["sampler"]

    all_products = {}

    for model_name in resolved["fit"]["models"]:
        products = _execute_single_model_fit(
            model_name=model_name,
            image_products=image_products,
            psf=psf,
            prior_cfg=resolved["fit"]["priors"],
            background_cfg=resolved["fit"].get(
                "background",
                DEFAULT_CONFIG["fit"]["background"],
            ),
            likelihood_cfg=likelihood_cfg,
            sampler_cfg=sampler_cfg,
        )

        save_real_fit_products(
            resolved,
            model_name,
            products,
            observation=observation,
        )

        all_products[model_name] = products

    if len(all_products) == 1:
        return next(iter(all_products.values()))

    comparison_cfg = resolved["fit"]["comparison"]

    if comparison_cfg["enabled"]:
        metric = comparison_cfg["metric"]

        if metric != "bayes_factor":
            raise ValueError(
                "Unsupported model-comparison metric "
                f"{metric!r}. Supported metric: 'bayes_factor'."
            )

        comparison = _compute_bayesian_model_comparison(
            all_products,
        )

        save_bayesian_model_comparison(
            resolved,
            comparison,
        )

    return all_products


def _execute_single_model_fit(
    *,
    model_name,
    image_products,
    psf,
    prior_cfg,
    background_cfg,
    likelihood_cfg,
    sampler_cfg,
):
    """Execute one complete Bayesian fit for one morphology model."""

    parameter_names, parameters, _, _ = _build_parameter_set(
        model_name,
        data_fit=image_products.prepared_data,
        noise_fit=image_products.noise,
        mask_fit=image_products.fitting_mask,
        target_x=image_products.target_x,
        target_y=image_products.target_y,
        prior_cfg=prior_cfg,
        background_cfg=background_cfg,
    )

    model_factory = _make_model_factory(
        model_name,
        psf,
        background_cfg=background_cfg,
    )

    problem = _build_image_inference_problem(
        model_factory=model_factory,
        parameters=parameters,
        likelihood_cfg=likelihood_cfg,
        data_fit=image_products.prepared_data,
        noise_fit=image_products.noise,
        mask_fit=image_products.fitting_mask,
    )

    backend = make_sampler(
        sampler_cfg,
        problem=problem,
    )

    result = backend.fit(
        progress_label=model_name,
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
        mask=image_products.fitting_mask,
        n_parameters=len(parameter_names),
    )

    return RealFitProducts(
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



def save_real_fit_products(
    config: dict[str, Any],
    model_name: str,
    products: RealFitProducts,
    *,
    observation: Observation | None = None,
) -> None:
    """Save real-galaxy fit products to disk."""
    paths = pipeline_paths(config)["fits"][model_name]

    paths["dir"].mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(paths["dir"] / "best_model_image.npy", products.best_model_image)
    np.save(paths["dir"] / "residual.npy", products.residual)
    np.save(paths["dir"] / "norm_residual.npy", products.norm_residual)

    if products.posterior is not None:
        products.posterior.save(paths["posterior"])

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
        "observation": (
            observation_metadata_dict(observation)
            if observation is not None
            else None
        ),
        "chi2": float(products.chi2),
        "ndof": int(products.ndof),
        "reduced_chi2": float(products.reduced_chi2),
        "logz": (
            None
            if products.posterior is None
            else products.posterior.logz
        ),
        "logzerr": (
            None
            if (
                products.posterior is None
                or products.posterior.logz_error is None
            )
            else products.posterior.logz_error
        ),
        "sampler": json_compatible_metadata(
            sampler_cfg
        ),
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
    """Save a human-readable text summary of a fit."""
    logz = (
        None
        if products.posterior is None
        else products.posterior.logz
    )

    logzerr = (
        None
        if (
            products.posterior is None
            or products.posterior.logz_error is None
        )
        else products.posterior.logz_error
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
    """Save Bayesian model-comparison results."""
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
    """Load saved real-galaxy fit products from disk."""
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

    posterior = None

    if paths["posterior"].exists():
        posterior = InferenceResult.load(
            paths["posterior"]
        )

    return RealFitProducts(
        model_name=model_name,
        posterior=posterior,
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
    """Load the configured PSF model."""
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
    """Parse numeric prior values, including symbolic pi/2 strings."""
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
    """Resolve parameter bounds from a prior configuration entry."""
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


def _prior_entry_for_parameter(
    name: str,
    prior_cfg: dict,
) -> dict:
    """Return prior configuration for a parameter, including legacy aliases."""
    if name in prior_cfg:
        return prior_cfg[name]

    aliases = {
        "q_bulge": "q",
        "q_disk": "q",
        "theta_bulge": "theta",
        "theta_disk": "theta",
    }

    alias = aliases.get(name)

    if alias is not None and alias in prior_cfg:
        return prior_cfg[alias]

    raise ValueError(
        f"Missing prior configuration for parameter: {name}"
    )


def _build_parameter_list(
    parameter_names: list[str],
    prior_cfg: dict,
    *,
    flux_total: float,
    target_x: float,
    target_y: float,
) -> ParameterSet:
    """Build a ParameterSet from ordered model parameter names."""
    parameters = []

    for name in parameter_names:
        cfg = _prior_entry_for_parameter(
            name,
            prior_cfg,
        )

        parameters.append(
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
        )

    return ParameterSet(parameters)


def _background_parameter_names(
    background_cfg: dict,
) -> list[str]:
    """Return the sampled parameter contract for one background model."""
    model = str(
        background_cfg["model"]
    ).strip().lower()

    if model == "constant":
        return [
            "background_level",
        ]

    if model == "plane":
        return [
            "background_level",
            "background_gradient_x",
            "background_gradient_y",
        ]

    raise ValueError(
        f"Unsupported background model {model!r}. "
        "Available models: 'constant', 'plane'."
    )


def _build_parameter_set(
    model_name: str,
    data_fit: np.ndarray,
    noise_fit: np.ndarray,
    mask_fit: np.ndarray,
    target_x: float,
    target_y: float,
    prior_cfg=None,
    background_cfg=None,
):
    """Build morphology and residual-background fit parameters."""
    if prior_cfg is None:
        prior_cfg = DEFAULT_CONFIG["fit"]["priors"]

    if background_cfg is None:
        background_cfg = DEFAULT_CONFIG["fit"]["background"]

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

    background_guess = np.nanmedian(
        data_fit[valid_for_flux],
    )

    model_prior_cfg = prior_cfg[model_name]

    model_definition = get_model(
        model_name
    )

    morphology_parameter_names = [
        name
        for name in model_definition.parameter_names
        if name != "sky"
    ]

    morphology_parameters = _build_parameter_list(
        morphology_parameter_names,
        model_prior_cfg,
        flux_total=flux_total,
        target_x=target_x,
        target_y=target_y,
    )

    background_parameter_names = _background_parameter_names(
        background_cfg
    )

    background_parameters = _build_parameter_list(
        background_parameter_names,
        background_cfg["priors"],
        flux_total=flux_total,
        target_x=target_x,
        target_y=target_y,
    )

    parameter_names = (
        morphology_parameter_names
        + background_parameter_names
    )

    parameters = ParameterSet(
        morphology_parameters.all_parameters
        + background_parameters.all_parameters
    )

    return (
        parameter_names,
        parameters,
        flux_total,
        background_guess,
    )



def _resolve_model_factory(
    model_name,
):
    """Return the production factory associated with a registered model."""
    model_definition = get_model(
        model_name
    )

    factories = {
        "SingleSersicModelFactory": make_single_sersic_psf_sky_model,
        "DoubleSersicModelFactory": make_double_sersic_psf_sky_model,
        "BulgeDiskModelFactory": make_bulge_disk_psf_sky_model,
        "SersicPointSourceModelFactory": (
            make_sersic_point_source_psf_sky_model
        ),
    }

    try:
        return factories[
            model_definition.factory_name
        ]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model factory: "
            f"{model_definition.factory_name}"
        ) from exc


class _ObservedModelFactory:
    """Pickle-safe observation-specific source-plus-background factory."""

    def __init__(
        self,
        model_name,
        psf,
        *,
        background_cfg,
    ) -> None:
        self.model_name = str(
            model_name
        )
        self.psf = psf

        self.background_model = str(
            background_cfg["model"]
        ).strip().lower()

        model_definition = get_model(
            self.model_name
        )

        self.morphology_parameter_names = tuple(
            name
            for name in model_definition.parameter_names
            if name != "sky"
        )

    def __call__(
        self,
        **parameters,
    ):
        """Build one observed source-plus-background model."""
        source_parameters = {
            name: parameters[name]
            for name in self.morphology_parameter_names
        }

        source_model = _resolve_model_factory(
            self.model_name
        )(
            **source_parameters,
            psf=self.psf,
            sky=0.0,
        )

        background = make_background_effect(
            self.background_model,
            level=parameters[
                "background_level"
            ],
            gradient_x=parameters.get(
                "background_gradient_x",
                0.0,
            ),
            gradient_y=parameters.get(
                "background_gradient_y",
                0.0,
            ),
        )

        return BackgroundAppliedModel(
            model=source_model,
            background=background,
        )


def _make_model_factory(
    model_name,
    psf,
    *,
    background_cfg=None,
):
    """Build a pickle-safe observation-specific production model factory."""
    if background_cfg is None:
        background_cfg = DEFAULT_CONFIG["fit"]["background"]

    return _ObservedModelFactory(
        model_name,
        psf,
        background_cfg=background_cfg,
    )


def _compute_best_model(
    result,
    parameter_names,
    model_factory,
    image_shape,
):
    """Render the best-fit model image from a posterior result."""
    best_parameters = {
        name: result.best_parameters[name]
        for name in parameter_names
    }

    best_model = model_factory(
        **best_parameters,
    )

    best_model_image = best_model.render(
        image_shape,
    )

    return best_model_image, best_parameters


def _compute_fit_statistics(
    data,
    model_image,
    noise,
    mask,
    n_parameters: int,
):
    """Compute residuals and chi-squared fit statistics."""
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
    """Compute Bayesian model-comparison quantities from model evidences."""
    logz = {
        model_name: float(
            products.posterior.logz
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


def _build_image_inference_problem(
    *,
    model_factory,
    parameters,
    likelihood_cfg,
    data_fit,
    noise_fit,
    mask_fit,
):
    """Build one backend-neutral image inference problem."""
    likelihood = make_likelihood(
        likelihood_cfg,
        data=data_fit,
        noise=noise_fit,
        mask=mask_fit,
    )

    return ImageInferenceProblem(
        model_factory=model_factory,
        parameters=parameters,
        likelihood=likelihood,
    )



def _run_sampler(
    data_fit,
    noise_fit,
    mask_fit,
    parameters,
    model_factory,
    likelihood_cfg,
    sampler_cfg,
    *,
    progress_label: str | None = None,
):
    """Build an inference problem and run one configured backend."""
    problem = _build_image_inference_problem(
        model_factory=model_factory,
        parameters=parameters,
        likelihood_cfg=likelihood_cfg,
        data_fit=data_fit,
        noise_fit=noise_fit,
        mask_fit=mask_fit,
    )

    backend = make_sampler(
        sampler_cfg,
        problem=problem,
    )

    return backend.fit(
        progress_label=progress_label,
    )


def _diagnostic_title(
    config: dict[str, Any],
    model_name: str,
) -> str:
    """Return a human-readable diagnostic title for a model."""
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
        .replace("Sersic", "Sérsic")
    )
