"""Production workflow for joint multi-observation image inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from baymorph.core.observation import Observation
from baymorph.fitting import make_sampler
from baymorph.fitting.model_factories import ObservedModelFactory
from baymorph.fitting.multi_observation_problem import (
    MultiObservationInferenceProblem,
)
from baymorph.fitting.representations import (
    RepresentationFactory,
)
from baymorph.likelihood import (
    ImageLikelihood,
    make_likelihood,
)
from baymorph.likelihood.multi_observation import (
    MultiObservationLikelihood,
)
from baymorph.models.geometry import get_model_geometry
from baymorph.observation.geometry import (
    slice_wcs_to_cutout,
)
from baymorph.observation.model_parameter_transforms import (
    SkyModelParameterTransform,
)
from baymorph.pipeline.checkpoint_signing import (
    baymorph_source_signature as _baymorph_source_signature,
    canonical_json_sha256 as _canonical_json_sha256,
    checkpoint_array_signature as _checkpoint_array_signature,
    checkpoint_runtime_versions as _checkpoint_runtime_versions,
    checkpoint_signature_value as _checkpoint_signature_value,
    prepare_signed_sampler_checkpoint,
    prior_checkpoint_signature as _prior_checkpoint_signature,
    qualified_class_name as _qualified_class_name,
    runtime_psf_checkpoint_signature as _runtime_psf_checkpoint_signature,
    write_json_atomic as _write_json_atomic,
)
from baymorph.pipeline.image_preparation import (
    load_image_preparation_products,
)
from baymorph.pipeline.likelihood_state import (
    RealFitLikelihoodState,
    prepare_real_fit_likelihood_state,
    require_complete_likelihood_normalization,
)
from baymorph.pipeline.multi_observation_noise import (
    estimate_cross_observation_noise_diagnostics,
    print_cross_observation_noise_diagnostics,
)
from baymorph.pipeline.multi_observation_outputs import (
    make_multi_observation_output_tree,
    multi_observation_paths,
)
from baymorph.pipeline.multi_observation_persistence import (
    save_multi_observation_fit_products,
)
from baymorph.pipeline.multi_observation_config import (
    multi_observation_background_model,
    multi_observation_parameter_frame,
    resolve_multi_observation_config,
)
from baymorph.pipeline.real_fit import (
    _compute_fit_statistics,
    _load_psf_from_config,
)
from baymorph.results import (
    InferenceResult,
    MultiObservationFitProducts,
    ObservationFitProducts,
)


@dataclass(frozen=True)
class _MultiObservationBuildContext:
    """Internal products required by joint inference and fit diagnostics."""

    problem: MultiObservationInferenceProblem
    image_products: dict[str, Any]
    observation_configs: dict[str, dict[str, Any]]
    psfs: dict[str, Any]
    likelihoods: dict[str, ImageLikelihood]
    likelihood_states: dict[str, RealFitLikelihoodState]
    likelihood_configs: dict[str, dict[str, Any]]


def build_multi_observation_problem_from_config(
    config: str | Path | Mapping[str, Any],
) -> MultiObservationInferenceProblem:
    """Build a production multi-observation inference problem.

    Parameters
    ----------
    config : mapping, str, or pathlib.Path
        Joint multi-observation configuration or path to its YAML file.

    Returns
    -------
    MultiObservationInferenceProblem
        Fully constructed backend-neutral inference problem.

    Notes
    -----
    This function constructs the scientific inference problem but does not
    instantiate or execute an inference backend. It is therefore suitable
    for deterministic validation, likelihood inspection, and testing.
    """
    raw_config, base_dir = _load_multi_observation_config(
        config
    )

    return _build_multi_observation_problem(
        raw_config,
        base_dir=base_dir,
    )


def _multi_observation_checkpoint_directory(
    raw_config: Mapping[str, Any],
    *,
    model_name: str,
) -> Path | None:
    """Return the pipeline-managed checkpoint directory, if outputs are configured."""
    if "outputs" not in raw_config:
        return None

    return multi_observation_paths(
        raw_config
    )["fits"][model_name]["dir"]


def run_multi_observation_inference_from_config(
    config: str | Path | Mapping[str, Any],
) -> InferenceResult:
    """Run joint multi-observation inference.

    Pre-fit blank-sky cross-observation noise diagnostics are evaluated
    before the configured sampler is instantiated. These diagnostics are
    descriptive and never modify the likelihood automatically.
    """
    raw_config, base_dir = _load_multi_observation_config(
        config
    )

    fit_config = raw_config["fit"]
    model_name = fit_config["models"][0]

    context = _build_multi_observation_context(
        raw_config,
        base_dir=base_dir,
    )

    _require_evidence_safe_likelihoods(
        fit_config=fit_config,
        context=context,
    )

    noise_diagnostics = (
        _estimate_pre_fit_noise_diagnostics(
            context=context,
            observation_configs=context.observation_configs,
        )
    )

    result = _run_multi_observation_inference(
        fit_config=fit_config,
        model_name=model_name,
        context=context,
        checkpoint_directory=_multi_observation_checkpoint_directory(
            raw_config,
            model_name=model_name,
        ),
    )

    result.diagnostics[
        "pre_fit_cross_observation_noise"
    ] = noise_diagnostics

    return result


def run_multi_observation_fit_from_config(
    config: str | Path | Mapping[str, Any],
) -> MultiObservationFitProducts:
    """Run a joint multi-observation analysis and build fit products.

    Pre-fit blank-sky cross-observation noise diagnostics are evaluated
    before sampling and stored in the canonical inference diagnostics.
    """
    raw_config, base_dir = _load_multi_observation_config(
        config
    )

    fit_config = raw_config["fit"]
    model_name = fit_config["models"][0]

    context = _build_multi_observation_context(
        raw_config,
        base_dir=base_dir,
    )

    _require_evidence_safe_likelihoods(
        fit_config=fit_config,
        context=context,
    )

    noise_diagnostics = (
        _estimate_pre_fit_noise_diagnostics(
            context=context,
            observation_configs=context.observation_configs,
        )
    )

    outputs_configured = (
        "outputs" in raw_config
    )
    fit_paths = (
        multi_observation_paths(
            raw_config
        )["fits"][model_name]
        if outputs_configured
        else None
    )
    completion_path = (
        None
        if fit_paths is None
        else fit_paths["complete"]
    )

    if completion_path is not None:
        outputs_cfg = raw_config.get(
            "outputs",
            {},
        )

        if (
            completion_path.exists()
            and not bool(
                outputs_cfg.get(
                    "overwrite",
                    False,
                )
            )
        ):
            raise FileExistsError(
                "A complete multi-observation fit already exists for "
                f"model {model_name!r}: {completion_path}. Use a new "
                "output namespace, or set outputs.overwrite=true "
                "explicitly."
            )

        if completion_path.exists():
            completion_path.unlink()

        make_multi_observation_output_tree(
            raw_config
        )

    inference_result = _run_multi_observation_inference(
        fit_config=fit_config,
        model_name=model_name,
        context=context,
        checkpoint_directory=(
            None
            if fit_paths is None
            else fit_paths["dir"]
        ),
    )

    inference_result.diagnostics[
        "pre_fit_cross_observation_noise"
    ] = noise_diagnostics

    products = _build_multi_observation_fit_products(
        context=context,
        inference_result=inference_result,
        model_name=model_name,
    )

    if outputs_configured:
        save_multi_observation_fit_products(
            raw_config,
            products,
        )

        _write_multi_observation_completion_marker(
            completion_path,
            model_name=model_name,
            config=raw_config,
            products=products,
        )

    return products


def _write_multi_observation_completion_marker(
    path: Path,
    *,
    model_name: str,
    config: Mapping[str, Any],
    products: MultiObservationFitProducts,
) -> None:
    """Mark a fully persisted joint multi-observation fit complete."""
    evidence = products.posterior.evidence

    payload = {
        "schema_version": 1,
        "model": model_name,
        "config_sha256": _canonical_json_sha256(
            _checkpoint_signature_value(
                dict(
                    config
                )
            )
        ),
        "logz": products.logz,
        "logz_error": products.logz_error,
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


def _estimate_pre_fit_noise_diagnostics(
    *,
    context: _MultiObservationBuildContext,
    observation_configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Estimate blank-sky cross-observation noise coupling before sampling."""
    diagnostics = estimate_cross_observation_noise_diagnostics(
        image_products=context.image_products,
        observation_configs=observation_configs,
    )

    print_cross_observation_noise_diagnostics(
        diagnostics
    )

    return diagnostics


def _require_evidence_safe_likelihoods(
    *,
    fit_config: Mapping[str, Any],
    context: _MultiObservationBuildContext,
) -> None:
    """Require a completely normalized likelihood for every observation.

    The requirement applies whatever the backend: every built-in backend
    produces Bayesian evidence.
    """
    sampler_config = fit_config.get(
        "sampler",
        {},
    )

    if not isinstance(
        sampler_config,
        Mapping,
    ):
        raise ValueError(
            "Multi-observation fit configuration requires a "
            "'sampler' mapping."
        )

    require_complete_likelihood_normalization(
        context.likelihood_states
    )


def _multi_observation_problem_signature(
    *,
    model_name: str,
    context: _MultiObservationBuildContext,
) -> str:
    """Return a strict signature of one joint multi-observation target.

    Covers the shared global parameter state, the model factory used across
    observations, and each observation's own PSF, likelihood, data, noise,
    mask, and correlation kernel. One joint problem has one deterministic
    global parameter state; this signature is the multi-observation
    counterpart of ``_real_fit_problem_signature``.
    """
    parameter_state = []

    for parameter in (
        context.problem.parameter_layout.parameter_set.all_parameters
    ):
        parameter_state.append(
            {
                "name": parameter.name,
                "sampled": not parameter.is_fixed,
                "prior": _prior_checkpoint_signature(
                    parameter.prior
                ),
            }
        )

    representation_factory = (
        context.problem.representation_factory
    )

    model_state = {
        "factory_class": _qualified_class_name(
            representation_factory.factory
        ),
        "background_model": getattr(
            representation_factory.factory,
            "background_model",
            None,
        ),
        "requested_name": str(
            model_name
        ),
        "parameter_transforms": _runtime_psf_checkpoint_signature(
            representation_factory.parameter_transforms
        ),
    }

    observation_state = {}

    for observation_name in sorted(
        context.image_products
    ):
        products = context.image_products[
            observation_name
        ]
        likelihood_state = context.likelihood_states[
            observation_name
        ]

        observation_state[
            observation_name
        ] = {
            "psf": _runtime_psf_checkpoint_signature(
                context.psfs[
                    observation_name
                ]
            ),
            "likelihood": {
                "class": _qualified_class_name(
                    context.likelihoods[
                        observation_name
                    ]
                ),
                "configuration": _checkpoint_signature_value(
                    dict(
                        context.likelihood_configs[
                            observation_name
                        ]
                    )
                ),
                "data": _checkpoint_array_signature(
                    products.prepared_data
                ),
                "noise": _checkpoint_array_signature(
                    likelihood_state.noise
                ),
                "mask": _checkpoint_array_signature(
                    products.fitting_mask
                ),
                "correlation_kernel": (
                    None
                    if likelihood_state.correlation_kernel is None
                    else _checkpoint_array_signature(
                        likelihood_state.correlation_kernel
                    )
                ),
            },
        }

    payload = {
        "schema": "baymorph_multi_observation_problem_v1",
        "problem_class": _qualified_class_name(
            context.problem
        ),
        "model": model_state,
        "parameters": parameter_state,
        "observations": observation_state,
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
    "nautilus": "baymorph_multi_observation_nautilus_checkpoint_target_v1",
    "pocomc": "baymorph_multi_observation_pocomc_checkpoint_target_v1",
}


def _prepare_multi_observation_sampler_checkpoint(
    sampler_cfg,
    *,
    model_name: str,
    context: _MultiObservationBuildContext,
    checkpoint_directory: Path | None,
):
    """Resolve and authenticate the native checkpoint of one joint fit."""
    return prepare_signed_sampler_checkpoint(
        sampler_cfg,
        schemas=_CHECKPOINT_SCHEMAS,
        checkpoint_directory=checkpoint_directory,
        problem_signature=partial(
            _multi_observation_problem_signature,
            model_name=model_name,
            context=context,
        ),
    )


def _run_multi_observation_inference(
    *,
    fit_config: Mapping[str, Any],
    model_name: str,
    context: _MultiObservationBuildContext,
    checkpoint_directory: Path | None = None,
) -> InferenceResult:
    """Execute the configured inference backend for one joint problem."""
    sampler_config = fit_config.get(
        "sampler"
    )

    if not isinstance(
        sampler_config,
        Mapping,
    ):
        raise ValueError(
            "Multi-observation fit configuration requires a "
            "'sampler' mapping."
        )

    runtime_sampler_config = (
        _prepare_multi_observation_sampler_checkpoint(
            sampler_config,
            model_name=model_name,
            context=context,
            checkpoint_directory=checkpoint_directory,
        )
    )

    backend = make_sampler(
        runtime_sampler_config,
        problem=context.problem,
    )

    return backend.fit(
        progress_label=(
            f"multi_observation_{model_name}"
        ),
    )


def _build_multi_observation_problem(
    raw_config: Mapping[str, Any],
    *,
    base_dir: str | Path,
) -> MultiObservationInferenceProblem:
    """Build a multi-observation problem from an already loaded config."""
    return _build_multi_observation_context(
        raw_config,
        base_dir=base_dir,
    ).problem


def _build_multi_observation_context(
    raw_config: Mapping[str, Any],
    *,
    base_dir: str | Path,
) -> _MultiObservationBuildContext:
    """Build the inference problem and retain observation data products."""
    observation_configs, parameter_layout = (
        resolve_multi_observation_config(
            raw_config,
            base_dir=base_dir,
        )
    )

    fit_config = raw_config["fit"]

    models = fit_config["models"]
    model_name = models[0]

    parameter_frame = multi_observation_parameter_frame(
        raw_config
    )

    image_products = {}
    psfs = {}

    for observation_name, observation_config in observation_configs.items():
        try:
            image_products[
                observation_name
            ] = load_image_preparation_products(
                observation_config
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Image-preparation products are required for "
                f"observation '{observation_name}'."
            ) from exc

        try:
            psfs[
                observation_name
            ] = _load_psf_from_config(
                observation_config
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "PSF products are required for observation "
                f"'{observation_name}'."
            ) from exc

    likelihoods = {}
    likelihood_states = {}
    likelihood_configs = {}

    for observation_name, products in image_products.items():
        observation_config = observation_configs[
            observation_name
        ]

        global_likelihood_config = fit_config.get(
            "likelihood",
            {
                "type": "gaussian",
                "include_normalization": True,
            },
        )

        observation_fit_config = observation_config.get(
            "fit",
            {},
        )

        likelihood_config = observation_fit_config.get(
            "likelihood",
            global_likelihood_config,
        )

        effective_observation_config = dict(
            observation_config
        )
        effective_fit_config = dict(
            observation_fit_config
        )
        effective_fit_config[
            "likelihood"
        ] = likelihood_config
        effective_observation_config[
            "fit"
        ] = effective_fit_config

        likelihood_state = prepare_real_fit_likelihood_state(
            effective_observation_config,
            products,
        )

        likelihood_states[
            observation_name
        ] = likelihood_state
        likelihood_configs[
            observation_name
        ] = likelihood_config

        likelihoods[observation_name] = make_likelihood(
            likelihood_config,
            data=products.prepared_data,
            noise=likelihood_state.noise,
            mask=products.fitting_mask,
            correlation_kernel=likelihood_state.correlation_kernel,
            vecchia_factorization=likelihood_state.vecchia_factorization,
        )

    likelihood = MultiObservationLikelihood(
        likelihoods
    )

    parameter_transforms = None

    if parameter_frame == "sky":
        parameter_transforms = _build_sky_parameter_transforms(
            observation_configs=observation_configs,
            image_products=image_products,
            model_name=model_name,
        )

    representation_factory = RepresentationFactory(
        factory=ObservedModelFactory(
            model_name,
            background_model=multi_observation_background_model(
                raw_config
            ),
        ),
        fixed_kwargs={
            observation_name: {
                "psf": psfs[
                    observation_name
                ],
            }
            for observation_name in parameter_layout.observations
        },
        parameter_transforms=parameter_transforms,
    )

    problem = MultiObservationInferenceProblem(
        representation_factory=representation_factory,
        parameter_layout=parameter_layout,
        likelihood=likelihood,
    )

    return _MultiObservationBuildContext(
        problem=problem,
        image_products=image_products,
        observation_configs=observation_configs,
        psfs=psfs,
        likelihoods=likelihoods,
        likelihood_states=likelihood_states,
        likelihood_configs=likelihood_configs,
    )


def _build_multi_observation_fit_products(
    *,
    context: _MultiObservationBuildContext,
    inference_result: InferenceResult,
    model_name: str,
) -> MultiObservationFitProducts:
    """Build observation and joint diagnostics at the best posterior sample."""
    problem = context.problem

    best_sample = np.asarray(
        inference_result.best_sample,
        dtype=float,
    )

    model_images = problem.render_models(
        best_sample
    )

    routed_parameters = problem.routed_parameters(
        best_sample
    )

    observation_products = {}

    joint_chi2 = 0.0
    total_valid_pixels = 0

    for observation_name in problem.parameter_layout.observations:
        products = context.image_products[
            observation_name
        ]

        parameters = dict(
            routed_parameters[
                observation_name
            ]
        )

        transform = (
            problem
            .representation_factory
            .parameter_transforms
            .get(
                observation_name
            )
        )

        if transform is not None:
            parameters = dict(
                transform(
                    parameters
                )
            )

        likelihood_state = context.likelihood_states[
            observation_name
        ]

        diagnostics = _compute_fit_statistics(
            data=products.prepared_data,
            model_image=model_images[
                observation_name
            ],
            noise=likelihood_state.noise,
            mask=products.fitting_mask,
            n_parameters=0,
            likelihood=context.likelihoods[
                observation_name
            ],
        )

        joint_chi2 += diagnostics[
            "chi2"
        ]
        total_valid_pixels += diagnostics[
            "n_valid_pixels"
        ]

        observation_products[
            observation_name
        ] = ObservationFitProducts(
            parameters=parameters,
            best_model_image=model_images[
                observation_name
            ],
            residual=diagnostics[
                "residual"
            ],
            norm_residual=diagnostics[
                "norm_residual"
            ],
            chi2=diagnostics[
                "chi2"
            ],
            n_valid_pixels=diagnostics[
                "n_valid_pixels"
            ],
            noise=likelihood_state.noise,
            likelihood_metadata=(
                None
                if likelihood_state.metadata is None
                else dict(
                    likelihood_state.metadata
                )
            ),
        )

    ndof = (
        total_valid_pixels
        - problem.ndim
    )

    reduced_chi2 = (
        float(
            joint_chi2
            / ndof
        )
        if ndof > 0
        else np.nan
    )

    return MultiObservationFitProducts(
        model_name=model_name,
        posterior=inference_result,
        best_parameters=dict(
            inference_result.best_parameters
        ),
        observations=observation_products,
        chi2=joint_chi2,
        ndof=ndof,
        reduced_chi2=reduced_chi2,
    )


def _build_sky_parameter_transforms(
    *,
    observation_configs: Mapping[str, Mapping[str, Any]],
    image_products: Mapping[str, Any],
    model_name: str,
) -> dict[str, SkyModelParameterTransform]:
    """Build fitted-grid sky-to-detector transforms for all observations."""
    geometry = get_model_geometry(
        model_name
    )

    transforms = {}

    for observation_name, observation_config in observation_configs.items():
        image_path = observation_config.get(
            "data",
            {},
        ).get(
            "image_path"
        )

        if not image_path:
            raise ValueError(
                "Sky-frame inference requires data.image_path for "
                f"observation '{observation_name}'."
            )

        observation = Observation.from_fits(
            image_path
        )

        wcs = observation.metadata.wcs

        if wcs is None:
            raise ValueError(
                "Sky-frame inference requires a valid celestial WCS for "
                f"observation '{observation_name}'."
            )

        bounds = image_products[
            observation_name
        ].cutout_bounds

        if not isinstance(
            bounds,
            Mapping,
        ):
            raise ValueError(
                "Sky-frame inference requires saved cutout bounds for "
                f"observation '{observation_name}'."
            )

        required_bounds = {
            "xmin",
            "xmax",
            "ymin",
            "ymax",
        }

        if set(bounds) != required_bounds:
            raise ValueError(
                "Sky-frame inference requires cutout bounds containing "
                "exactly xmin, xmax, ymin, and ymax for observation "
                f"'{observation_name}'."
            )

        cutout_wcs = slice_wcs_to_cutout(
            wcs,
            xmin=bounds["xmin"],
            xmax=bounds["xmax"],
            ymin=bounds["ymin"],
            ymax=bounds["ymax"],
        )

        transforms[
            observation_name
        ] = SkyModelParameterTransform(
            wcs=cutout_wcs,
            geometry=geometry,
        )

    return transforms


def _load_multi_observation_config(
    config: str | Path | Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    Path,
]:
    """Load a raw joint configuration and its path-resolution directory."""
    if isinstance(
        config,
        (str, Path),
    ):
        path = Path(
            config
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Multi-observation config file not found: {path}"
            )

        with path.open(
            "r"
        ) as stream:
            loaded = yaml.safe_load(
                stream
            )

        if not isinstance(
            loaded,
            dict,
        ):
            raise TypeError(
                "Multi-observation YAML top level must be a mapping."
            )

        return (
            loaded,
            path.parent,
        )

    if isinstance(
        config,
        Mapping,
    ):
        return (
            dict(config),
            Path.cwd(),
        )

    raise TypeError(
        "config must be a path or a mapping."
    )
