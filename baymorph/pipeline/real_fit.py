"""Real-galaxy fitting workflow."""

from __future__ import annotations

import json
import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from baymorph.pipeline.diagnostics_common import (
    diagnostic_title,
    fit_diagnostic_title,
    object_label,
)
from baymorph.core.observation import Observation
from baymorph.fitting import make_sampler
from baymorph.fitting.image_problem import ImageInferenceProblem
from baymorph.fitting.model_factories import ObservedModelFactory
from baymorph.fitting.parameters import Parameter, ParameterSet
from baymorph.likelihood import (
    CorrelatedGaussianImageLikelihood,
    make_likelihood,
)
from baymorph.models import get_model
from baymorph.observation.background import background_parameter_names
from baymorph.pipeline.checkpoint_signing import (
    baymorph_source_signature as _baymorph_source_signature,
    canonical_json_sha256 as _canonical_json_sha256,
    checkpoint_array_signature as _checkpoint_array_signature,
    checkpoint_runtime_versions as _checkpoint_runtime_versions,
    checkpoint_signature_value as _checkpoint_signature_value,
    prior_checkpoint_signature as _prior_checkpoint_signature,
    prepare_signed_sampler_checkpoint,
    qualified_class_name as _qualified_class_name,
    runtime_psf_checkpoint_signature as _runtime_psf_checkpoint_signature,
    write_json_atomic as _write_json_atomic,
)
from baymorph.pipeline.config import (
    DEFAULT_CONFIG,
    load_resolved_config,
    resolve_config,
    validate_fit_priors,
)
from baymorph.pipeline.image_preparation import load_image_preparation_products
from baymorph.pipeline.likelihood_state import (
    RealFitLikelihoodState,
    prepare_real_fit_likelihood_state,
    require_complete_likelihood_normalization,
)
from baymorph.pipeline.observation_metadata import observation_metadata_dict
from baymorph.pipeline.outputs import (
    make_output_tree,
    pipeline_paths,
    save_fit_resolved_config,
)
from baymorph.pipeline.prior_config import (
    parse_prior_config,
    prior_config_for_parameter,
)
from baymorph.pipeline.psf_construction import load_empirical_psf
from baymorph.results import InferenceResult, RealFitProducts
from baymorph.results.inference import json_compatible_metadata
from baymorph.simulation.psf import GaussianPSF


def run_real_fit_from_config(
    config: str | Path | dict[str, Any],
):
    """Run real-galaxy model fitting from a pipeline configuration."""
    if isinstance(config, (str, Path)):
        resolved = load_resolved_config(config)
    else:
        resolved = resolve_config(config)

    validate_fit_priors(
        resolved
    )

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

    preparation_references = _target_reference_coordinates(
        target_position=resolved["data"].get(
            "target_position"
        ),
        cutout_bounds=image_products.cutout_bounds,
        observation=observation,
    )

    prior_references = dict(
        preparation_references
    )

    likelihood_cfg = resolved["fit"].get(
        "likelihood",
        None,
    )
    likelihood_state = prepare_real_fit_likelihood_state(
        resolved,
        image_products,
    )
    require_complete_likelihood_normalization(
        likelihood_state
    )

    # Accepted preparation and PSF products are validated before fit output
    # directories or inference configuration snapshots are written.
    make_output_tree(resolved)

    all_products = {}

    for model_name in resolved["fit"]["models"]:
        model_config = _single_model_fit_config(
            resolved,
            model_name,
        )
        model_paths = pipeline_paths(
            model_config
        )["fits"][model_name]
        completion_path = model_paths[
            "complete"
        ]

        if (
            completion_path.exists()
            and not bool(
                model_config["outputs"].get(
                    "overwrite",
                    False,
                )
            )
        ):
            raise FileExistsError(
                "A complete fit already exists for model "
                f"{model_name!r}: {completion_path}. Use a new output "
                "namespace, or set outputs.overwrite=true explicitly."
            )

        if completion_path.exists():
            completion_path.unlink()

        save_fit_resolved_config(
            model_config,
            model_name,
        )

        products = _execute_single_model_fit(
            model_name=model_name,
            image_products=image_products,
            psf=psf,
            prior_cfg=model_config["fit"]["priors"],
            background_cfg=model_config["fit"].get(
                "background",
                DEFAULT_CONFIG["fit"]["background"],
            ),
            references=prior_references,
            likelihood_cfg=likelihood_cfg,
            likelihood_state=likelihood_state,
            sampler_cfg=model_config["fit"]["sampler"],
            checkpoint_directory=model_paths["dir"],
        )

        save_real_fit_products(
            model_config,
            model_name,
            products,
            observation=observation,
            boundary_priors=_resolved_boundary_priors(
                model_name=model_name,
                prior_cfg=model_config["fit"]["priors"],
                background_cfg=model_config["fit"].get(
                    "background",
                    DEFAULT_CONFIG["fit"]["background"],
                ),
                references=prior_references,
            ),
        )

        _write_fit_completion_marker(
            completion_path,
            model_name=model_name,
            config=model_config,
            products=products,
        )

        all_products[model_name] = products

    if len(all_products) == 1:
        return next(
            iter(
                all_products.values()
            )
        )

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


def _single_model_fit_config(
    config: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Return one model-specific execution configuration."""
    runtime = deepcopy(config)
    runtime["fit"]["models"] = [
        model_name
    ]
    runtime["fit"]["comparison"] = {
        "enabled": False,
        "metric": "bayes_factor",
    }

    sampler = runtime["fit"]["sampler"]
    sampler_name = str(
        sampler.get(
            "name",
            "",
        )
    ).strip().lower()
    init_config = dict(
        sampler.get(
            "init",
            {},
        )
    )
    sampler["init"] = init_config
    fit_directory = pipeline_paths(
        config
    )["fits"][model_name]["dir"]

    if (
        sampler_name == "nautilus"
        and str(
            init_config.get(
                "filepath",
                "",
            )
        ).strip().lower()
        == "auto"
    ):
        init_config["filepath"] = str(
            (
                fit_directory
                / "nautilus_checkpoint.hdf5"
            ).expanduser().resolve()
        )

    return runtime


def _resolved_boundary_priors(
    *,
    model_name: str,
    prior_cfg: dict[str, Any],
    background_cfg: dict[str, Any],
    references: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """Resolve every sampled prior into absolute boundary coordinates."""
    model_priors = prior_cfg[
        model_name
    ]
    resolved_priors: dict[
        str,
        dict[str, Any],
    ] = {}

    for parameter_name in get_model(
        model_name
    ).parameter_names:
        prior = dict(
            prior_config_for_parameter(
                parameter_name,
                model_priors,
            )
        )
        reference_name = prior.pop(
            "relative_to",
            None,
        )

        if reference_name is not None:
            reference = float(
                references[
                    str(
                        reference_name
                    )
                ]
            )
            prior["lower"] = float(
                prior["lower"]
            ) + reference
            prior["upper"] = float(
                prior["upper"]
            ) + reference

        resolved_priors[
            parameter_name
        ] = prior

    background_priors = background_cfg.get(
        "priors",
        {},
    )
    background_names = background_parameter_names(
        background_cfg.get(
            "model",
            "constant",
        )
    )

    for parameter_name in background_names:
        resolved_priors[
            parameter_name
        ] = dict(
            background_priors[
                parameter_name
            ]
        )

    return resolved_priors


def _write_fit_completion_marker(
    path: Path,
    *,
    model_name: str,
    config: dict[str, Any],
    products: RealFitProducts,
) -> None:
    """Mark a fully persisted model fit complete."""
    posterior = products.posterior
    evidence = (
        None
        if posterior is None
        else posterior.evidence
    )
    payload = {
        "schema_version": 1,
        "model": model_name,
        "config_sha256": _canonical_json_sha256(
            _checkpoint_signature_value(
                config
            )
        ),
        "logz": (
            None
            if posterior is None
            else posterior.logz
        ),
        "logz_error": (
            None
            if posterior is None
            else posterior.logz_error
        ),
        "evidence_method": (
            None
            if evidence is None
            else evidence.method
        ),
    }

    _write_json_atomic(
        path,
        payload,
    )


def _real_fit_problem_signature(
    *,
    model_name,
    problem,
    likelihood_cfg,
    data_fit,
    noise_fit,
    mask_fit,
    correlation_kernel_fit,
) -> str:
    """Return a strict signature of one real-fit posterior target."""
    parameter_state = []

    for parameter in problem.parameters.all_parameters:
        parameter_state.append(
            {
                "name": parameter.name,
                "sampled": not parameter.is_fixed,
                "prior": _prior_checkpoint_signature(
                    parameter.prior
                ),
            }
        )

    model_factory = problem.model_factory

    model_state = {
        "class": _qualified_class_name(
            model_factory
        ),
        "model_name": str(
            model_factory.model_name
        ),
        "background_model": str(
            model_factory.background_model
        ),
        "morphology_parameter_names": list(
            model_factory.morphology_parameter_names
        ),
        "psf": _runtime_psf_checkpoint_signature(
            model_factory.psf
        ),
    }

    payload = {
        "schema": "baymorph_real_fit_problem_v1",
        "problem_class": _qualified_class_name(
            problem
        ),
        "model": {
            "requested_name": str(
                model_name
            ),
            "runtime": model_state,
        },
        "parameters": parameter_state,
        "likelihood": {
            "class": _qualified_class_name(
                problem.likelihood
            ),
            "configuration": _checkpoint_signature_value(
                (
                    {}
                    if likelihood_cfg is None
                    else dict(
                        likelihood_cfg
                    )
                )
            ),
            "data": _checkpoint_array_signature(
                data_fit
            ),
            "noise": _checkpoint_array_signature(
                noise_fit
            ),
            "mask": _checkpoint_array_signature(
                mask_fit
            ),
            "correlation_kernel": (
                None
                if correlation_kernel_fit is None
                else _checkpoint_array_signature(
                    correlation_kernel_fit
                )
            ),
            "effective_gain": getattr(
                problem.likelihood,
                "effective_gain",
                None,
            ),
        },
        "baymorph_source": _baymorph_source_signature(),
        "runtime_versions": _checkpoint_runtime_versions(),
    }

    return (
        "sha256:"
        + _canonical_json_sha256(
            payload
        )
    )


_CHECKPOINT_SCHEMAS = {
    "nautilus": "baymorph_nautilus_checkpoint_target_v1",
    "pocomc": "baymorph_pocomc_checkpoint_target_v1",
}


def _prepare_sampler_checkpoint(
    sampler_cfg,
    *,
    model_name,
    problem,
    likelihood_cfg,
    data_fit,
    noise_fit,
    mask_fit,
    correlation_kernel_fit,
    checkpoint_directory,
):
    """Resolve and authenticate the native checkpoint of one real fit."""
    return prepare_signed_sampler_checkpoint(
        sampler_cfg,
        schemas=_CHECKPOINT_SCHEMAS,
        checkpoint_directory=checkpoint_directory,
        problem_signature=partial(
            _real_fit_problem_signature,
            model_name=model_name,
            problem=problem,
            likelihood_cfg=likelihood_cfg,
            data_fit=data_fit,
            noise_fit=noise_fit,
            mask_fit=mask_fit,
            correlation_kernel_fit=correlation_kernel_fit,
        ),
    )


def _execute_single_model_fit(
    *,
    model_name,
    image_products,
    psf,
    prior_cfg,
    background_cfg,
    references,
    likelihood_cfg,
    likelihood_state: RealFitLikelihoodState,
    sampler_cfg,
    checkpoint_directory=None,
):
    """Execute one complete Bayesian fit for one morphology model."""
    parameter_names, parameters = _build_parameter_set(
        model_name,
        prior_cfg=prior_cfg,
        background_cfg=background_cfg,
        references=references,
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
        noise_fit=likelihood_state.noise,
        mask_fit=image_products.fitting_mask,
        correlation_kernel_fit=likelihood_state.correlation_kernel,
        effective_gain=likelihood_state.effective_gain,
    )

    runtime_sampler_cfg = (
        _prepare_sampler_checkpoint(
            sampler_cfg,
            model_name=model_name,
            problem=problem,
            likelihood_cfg=likelihood_cfg,
            data_fit=image_products.prepared_data,
            noise_fit=likelihood_state.noise,
            mask_fit=image_products.fitting_mask,
            correlation_kernel_fit=(
                likelihood_state.correlation_kernel
            ),
            checkpoint_directory=checkpoint_directory,
        )
    )

    backend = make_sampler(
        runtime_sampler_cfg,
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
        noise=likelihood_state.noise,
        mask=image_products.fitting_mask,
        n_parameters=len(
            parameter_names
        ),
        likelihood=problem.likelihood,
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
        noise=likelihood_state.noise,
        likelihood_metadata=(
            None
            if likelihood_state.metadata is None
            else dict(likelihood_state.metadata)
        ),
    )


def save_real_fit_products(
    config: dict[str, Any],
    model_name: str,
    products: RealFitProducts,
    *,
    observation: Observation | None = None,
    boundary_priors: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Save real-galaxy fit products to disk."""
    paths = pipeline_paths(
        config
    )["fits"][model_name]

    paths["dir"].mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        paths["dir"] / "best_model_image.npy",
        products.best_model_image,
    )
    np.save(
        paths["dir"] / "residual.npy",
        products.residual,
    )
    np.save(
        paths["dir"] / "norm_residual.npy",
        products.norm_residual,
    )

    noise_path = (
        paths["dir"]
        / "noise.npy"
    )

    if products.noise is not None:
        np.save(
            noise_path,
            np.asarray(
                products.noise,
                dtype=float,
            ),
        )
    elif noise_path.exists():
        noise_path.unlink()

    prior_boundary_diagnostics = None

    if products.posterior is not None:
        from baymorph.results.prior_boundaries import (
            audit_posterior_prior_boundaries,
        )

        model_priors = (
            boundary_priors
            if boundary_priors is not None
            else (
                config.get(
                    "fit",
                    {}
                )
                .get(
                    "priors",
                    {}
                )
                .get(
                    model_name,
                    {}
                )
            )
        )

        prior_boundary_diagnostics = (
            audit_posterior_prior_boundaries(
                products.posterior.posterior,
                model_priors,
            )
        )

        products.posterior.diagnostics[
            "prior_boundary_diagnostics"
        ] = prior_boundary_diagnostics

        if (
            prior_boundary_diagnostics[
                "status"
            ]
            == "boundary_contact"
        ):
            flagged = ", ".join(
                prior_boundary_diagnostics[
                    "flagged_parameters"
                ]
            )

            warnings.warn(
                "Posterior contact with a declared prior boundary "
                f"was detected for: {flagged}. Posterior summaries "
                "and Bayesian-evidence interpretation may be "
                "sensitive to the declared prior support.",
                RuntimeWarning,
                stacklevel=2,
            )

        products.posterior.save(
            paths["posterior"]
        )

    with open(
        paths["dir"] / "best_parameters.json",
        "w",
    ) as f:
        json.dump(
            {
                key: float(
                    value
                )
                for key, value in products.best_parameters.items()
            },
            f,
            indent=2,
        )

    sampler_cfg = config["fit"]["sampler"]

    if products.posterior is not None:
        runtime_sampler_cfg = (
            products.posterior.provenance.get(
                "sampler"
            )
        )

        if isinstance(
            runtime_sampler_cfg,
            dict,
        ):
            sampler_cfg = runtime_sampler_cfg

    summary = {
        "observation": (
            observation_metadata_dict(
                observation
            )
            if observation is not None
            else None
        ),
        "chi2": float(
            products.chi2
        ),
        "ndof": int(
            products.ndof
        ),
        "reduced_chi2": float(
            products.reduced_chi2
        ),
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
        "evidence_method": (
            None
            if products.posterior is None
            else getattr(
                products.posterior,
                "evidence_method",
                None,
            )
        ),
        "evidence_uncertainty_kind": (
            None
            if (
                products.posterior is None
                or getattr(
                    products.posterior,
                    "evidence",
                    None,
                )
                is None
            )
            else products.posterior.evidence.uncertainty_kind
        ),
        "sampler": json_compatible_metadata(
            sampler_cfg
        ),
        "likelihood": (
            None
            if products.likelihood_metadata is None
            else json_compatible_metadata(
                products.likelihood_metadata
            )
        ),
        "prior_boundary_diagnostics": (
            None
            if prior_boundary_diagnostics is None
            else json_compatible_metadata(
                prior_boundary_diagnostics
            )
        ),
    }

    with open(
        paths["fit_summary"],
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    save_fit_summary_text(
        paths["fit_summary_text"],
        model_title=_diagnostic_title(
            config,
            model_name,
        ),
        products=products,
        likelihood_metadata=products.likelihood_metadata,
    )


def save_fit_summary_text(
    path: Path,
    *,
    model_title: str,
    products: RealFitProducts,
    likelihood_metadata: dict[str, Any] | None = None,
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
    ]

    if likelihood_metadata is not None:
        normalization_complete = bool(
            likelihood_metadata.get(
                "normalization_complete",
                True,
            )
        )

        lines.extend(
            [
                (
                    "Likelihood: "
                    f"{likelihood_metadata.get('type', 'unknown')}"
                ),
                (
                    "Likelihood normalization complete: "
                    f"{normalization_complete}"
                ),
            ]
        )

        if not normalization_complete:
            lines.append(
                "logZ interpretation: relative; a fixed likelihood "
                "normalization constant is omitted"
            )

    lines.extend(
        [
        "",
        "Fit statistics",
        f"chi2 = {products.chi2:.8g}",
        f"ndof = {products.ndof:d}",
        f"reduced_chi2 = {products.reduced_chi2:.8g}",
        (
            f"logZ = {logz:.8g}"
            if logz is not None
            else "logZ = None"
        ),
        (
            f"logZerr = {logzerr:.8g}"
            if logzerr is not None
            else "logZerr = None"
        ),
        "",
        "Best-fit parameters",
        ]
    )

    for key, value in products.best_parameters.items():
        lines.append(
            f"{key} = {float(value):.8g}"
        )

    path.write_text(
        "\n".join(
            lines
        )
        + "\n"
    )


def save_bayesian_model_comparison(
    config: dict[str, Any],
    comparison: dict[str, Any],
) -> None:
    """Save Bayesian model-comparison results."""
    paths = pipeline_paths(
        config
    )["model_comparison"]

    paths["dir"].mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        paths["bayes_factor"],
        "w",
    ) as f:
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

    paths = pipeline_paths(
        resolved
    )["fits"][model_name]

    best_model_image = np.load(
        paths["dir"]
        / "best_model_image.npy"
    )
    residual = np.load(
        paths["dir"]
        / "residual.npy"
    )
    norm_residual = np.load(
        paths["dir"]
        / "norm_residual.npy"
    )

    with open(
        paths["dir"]
        / "best_parameters.json",
        "r",
    ) as f:
        best_parameters = json.load(
            f
        )

    with open(
        paths["dir"]
        / "fit_summary.json",
        "r",
    ) as f:
        summary = json.load(
            f
        )

    image_products = load_image_preparation_products(
        resolved
    )

    posterior = None

    if paths["posterior"].exists():
        posterior = InferenceResult.load(
            paths["posterior"]
        )

    noise_path = (
        paths["dir"]
        / "noise.npy"
    )

    if noise_path.exists():
        noise = np.asarray(
            np.load(
                noise_path
            ),
            dtype=float,
        )
    else:
        noise = np.asarray(
            image_products.noise,
            dtype=float,
        )

    if noise.shape != best_model_image.shape:
        raise ValueError(
            "Loaded inference noise shape does not match "
            "the saved best-model image."
        )

    return RealFitProducts(
        model_name=model_name,
        posterior=posterior,
        best_parameters={
            key: float(
                value
            )
            for key, value in best_parameters.items()
        },
        best_model_image=best_model_image,
        residual=residual,
        norm_residual=norm_residual,
        chi2=float(
            summary["chi2"]
        ),
        ndof=int(
            summary["ndof"]
        ),
        reduced_chi2=float(
            summary["reduced_chi2"]
        ),
        noise=noise,
        likelihood_metadata=(
            summary.get(
                "likelihood"
            )
        ),
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


def _target_reference_coordinates(
    *,
    target_position,
    cutout_bounds,
    observation,
) -> dict[str, float]:
    """Transform an explicit target seed into cutout coordinates.

    The transformation depends only on the caller-declared detector position
    and the deterministic cutout geometry. Image-derived morphology estimates
    are intentionally excluded from prior construction.
    """
    if target_position == "center":
        if observation is None:
            raise ValueError(
                "Target position 'center' cannot be resolved without the "
                "original observation metadata."
            )

        full_x = float(
            observation.metadata.center_x
        )
        full_y = float(
            observation.metadata.center_y
        )

    elif isinstance(
        target_position,
        dict,
    ):
        try:
            full_x = float(
                target_position["x"]
            )
            full_y = float(
                target_position["y"]
            )
        except KeyError as exc:
            raise ValueError(
                "Target-position mapping must define 'x' and 'y'."
            ) from exc

    else:
        raise ValueError(
            "Fitting requires an explicit target position or 'center'."
        )

    if cutout_bounds is None:
        raise ValueError(
            "Prepared observation is missing cutout_bounds. "
            "Explicit cutout geometry is required to transform the target "
            "seed into fitting coordinates."
        )

    try:
        xmin = float(
            cutout_bounds["xmin"]
        )
        ymin = float(
            cutout_bounds["ymin"]
        )
    except (
        KeyError,
        TypeError,
    ) as exc:
        raise ValueError(
            "cutout_bounds must define 'xmin' and 'ymin'."
        ) from exc

    return {
        "target_x": full_x - xmin,
        "target_y": full_y - ymin,
    }


def _build_parameter_list(
    parameter_names: list[str],
    prior_cfg: dict,
    *,
    references: dict[str, float],
) -> ParameterSet:
    """Build parameters exclusively from typed scientific priors."""
    parameters = []

    for name in parameter_names:
        cfg = prior_config_for_parameter(
            name,
            prior_cfg,
        )

        parameters.append(
            Parameter(
                name,
                prior=parse_prior_config(
                    cfg,
                    references=references,
                ),
            )
        )

    return ParameterSet(
        parameters
    )


def _build_parameter_set(
    model_name: str,
    *,
    prior_cfg: dict,
    background_cfg: dict,
    references: dict[str, float],
) -> tuple[
    list[str],
    ParameterSet,
]:
    """Build morphology and background parameters from typed priors only."""
    if model_name not in prior_cfg:
        raise ValueError(
            f"Missing prior configuration for model: {model_name}"
        )

    model_prior_cfg = prior_cfg[
        model_name
    ]

    model_definition = get_model(
        model_name
    )

    morphology_parameter_names = list(
        model_definition.parameter_names
    )

    morphology_parameters = _build_parameter_list(
        morphology_parameter_names,
        model_prior_cfg,
        references=references,
    )

    background_names = list(
        background_parameter_names(
            background_cfg["model"]
        )
    )

    background_parameters = _build_parameter_list(
        background_names,
        background_cfg["priors"],
        references=references,
    )

    parameter_names = (
        morphology_parameter_names
        + background_names
    )

    parameters = ParameterSet(
        morphology_parameters.all_parameters
        + background_parameters.all_parameters
    )

    return (
        parameter_names,
        parameters,
    )


def _make_model_factory(
    model_name,
    psf,
    *,
    background_cfg=None,
):
    """Build the pickle-safe production model factory of one observation.

    The factory composes the registered morphology with the observation PSF
    and applies the declared residual background (AD-031). The same class
    serves joint multi-observation inference, where each observation injects
    its own PSF at call time instead of binding it here.
    """
    if background_cfg is None:
        background_cfg = DEFAULT_CONFIG[
            "fit"
        ][
            "background"
        ]

    return ObservedModelFactory(
        model_name,
        background_model=background_cfg[
            "model"
        ],
        psf=psf,
    )


def _compute_best_model(
    result,
    parameter_names,
    model_factory,
    image_shape,
):
    """Render the best-fit model image from a posterior result."""
    best_parameters = {
        name: result.best_parameters[
            name
        ]
        for name in parameter_names
    }

    best_model = model_factory(
        **best_parameters,
    )

    best_model_image = best_model.render(
        image_shape,
    )

    return (
        best_model_image,
        best_parameters,
    )


def _compute_fit_statistics(
    data,
    model_image,
    noise,
    mask,
    n_parameters: int,
    likelihood=None,
):
    """Compute residuals and likelihood-consistent fit statistics."""
    residual = (
        data
        - model_image
    )

    if isinstance(
        likelihood,
        CorrelatedGaussianImageLikelihood,
    ):
        norm_residual = (
            likelihood.marginal_standardized_residual(
                model_image
            )
        )

        chi2 = float(
            likelihood.quadratic_form(
                model_image
            )
        )

        n_valid = int(
            likelihood.n_valid_pixels
        )
    else:
        valid = (
            (~mask)
            & np.isfinite(
                data
            )
            & np.isfinite(
                model_image
            )
            & np.isfinite(
                noise
            )
            & (
                noise > 0
            )
        )

        norm_residual = np.full_like(
            residual,
            np.nan,
            dtype=float,
        )

        norm_residual[
            valid
        ] = (
            residual[
                valid
            ]
            / noise[
                valid
            ]
        )

        chi2 = float(
            np.sum(
                norm_residual[
                    valid
                ]
                ** 2
            )
        )

        n_valid = int(
            np.sum(
                valid
            )
        )

    ndof = int(
        n_valid
        - n_parameters
    )

    reduced_chi2 = (
        float(
            chi2
            / ndof
        )
        if ndof > 0
        else np.nan
    )

    return {
        "residual": residual,
        "norm_residual": norm_residual,
        "chi2": chi2,
        "n_valid_pixels": n_valid,
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
        for (
            model_name,
            products,
        ) in products_by_model.items()
    }
    logz_error = {
        model_name: (
            None
            if getattr(
                products.posterior,
                "logz_error",
                None,
            )
            is None
            else float(
                products.posterior.logz_error
            )
        )
        for (
            model_name,
            products,
        ) in products_by_model.items()
    }
    evidence_method = {}
    uncertainty_kind = {}

    for model_name, products in products_by_model.items():
        evidence = getattr(
            products.posterior,
            "evidence",
            None,
        )
        evidence_method[
            model_name
        ] = (
            getattr(
                products.posterior,
                "evidence_method",
                None,
            )
            if evidence is None
            else evidence.method
        )
        uncertainty_kind[
            model_name
        ] = (
            None
            if evidence is None
            else evidence.uncertainty_kind
        )

    if (
        "single_sersic" not in logz
        or "double_sersic" not in logz
    ):
        return {
            "logz": logz,
            "logz_error": logz_error,
            "evidence_method": evidence_method,
            "uncertainty_kind": uncertainty_kind,
            "delta_logz": {},
            "delta_logz_error": {},
            "log_bayes_factor": {},
            "bayes_factor": {},
        }

    delta_logz = (
        logz[
            "double_sersic"
        ]
        - logz[
            "single_sersic"
        ]
    )

    log_bayes_factor = float(
        delta_logz
    )
    single_error = logz_error[
        "single_sersic"
    ]
    double_error = logz_error[
        "double_sersic"
    ]
    delta_logz_error = (
        None
        if (
            single_error is None
            or double_error is None
        )
        else float(
            np.hypot(
                single_error,
                double_error,
            )
        )
    )

    if log_bayes_factor < 700:
        bayes_factor = float(
            np.exp(
                log_bayes_factor
            )
        )
    else:
        bayes_factor = None

    return {
        "logz": logz,
        "logz_error": logz_error,
        "evidence_method": evidence_method,
        "uncertainty_kind": uncertainty_kind,
        "delta_logz": {
            "double_sersic_minus_single_sersic": delta_logz,
        },
        "delta_logz_error": {
            "double_sersic_minus_single_sersic": delta_logz_error,
            "combination": (
                "quadrature_assuming_independent_estimates"
                if delta_logz_error is not None
                else None
            ),
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
    correlation_kernel_fit=None,
    effective_gain=None,
):
    """Build one backend-neutral image inference problem."""
    likelihood = make_likelihood(
        likelihood_cfg,
        data=data_fit,
        noise=noise_fit,
        mask=mask_fit,
        correlation_kernel=correlation_kernel_fit,
        effective_gain=effective_gain,
    )

    return ImageInferenceProblem(
        model_factory=model_factory,
        parameters=parameters,
        likelihood=likelihood,
    )


_object_label = object_label


_fit_diagnostic_title = fit_diagnostic_title


_diagnostic_title = diagnostic_title
