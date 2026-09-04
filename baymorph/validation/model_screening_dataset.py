"""Balanced simulation datasets for model-screening validation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from baymorph.models import (
    parameter_names,
    registered_models,
)
from baymorph.pipeline.prior_config import parse_prior_config
from baymorph.validation.model_screening_simulation import (
    ModelScreeningSimulationCase,
    simulate_model_screening_case,
)


PriorConfig = Mapping[str, Any]
ModelPriorConfig = Mapping[
    str,
    Mapping[
        str,
        PriorConfig,
    ],
]


def _sample_prior(
    config: PriorConfig,
    *,
    rng: np.random.Generator,
    references: Mapping[str, float] | None = None,
) -> float:
    """Draw one value from a BayMorph prior configuration."""
    prior = parse_prior_config(
        config,
        references=references,
    )

    if prior.is_fixed:
        unit_value = 0.5
    else:
        unit_value = float(
            rng.random()
        )

    return float(
        prior.transform(
            unit_value
        )
    )


def _validate_model_priors(
    model_priors: ModelPriorConfig,
) -> None:
    """Require complete priors for every registered model."""
    expected_models = set(
        registered_models()
    )
    supplied_models = set(
        model_priors
    )

    if supplied_models != expected_models:
        missing = sorted(
            expected_models - supplied_models
        )
        extra = sorted(
            supplied_models - expected_models
        )

        raise ValueError(
            "Model-screening priors must cover every registered model. "
            f"Missing: {missing}. Extra: {extra}."
        )

    for model_name in registered_models():
        expected_parameters = set(
            parameter_names(
                model_name
            )
        )
        supplied_parameters = set(
            model_priors[
                model_name
            ]
        )

        if supplied_parameters != expected_parameters:
            missing = sorted(
                expected_parameters - supplied_parameters
            )
            extra = sorted(
                supplied_parameters - expected_parameters
            )

            raise ValueError(
                f"Simulation priors for {model_name!r} must cover "
                "its complete parameter contract. "
                f"Missing: {missing}. Extra: {extra}."
            )


def generate_model_screening_cases(
    *,
    model_priors: ModelPriorConfig,
    n_per_model: int,
    image_shape: tuple[int, int],
    psf_fwhm_prior: PriorConfig,
    noise_sigma_prior: PriorConfig,
    random_state: int = 0,
) -> Iterator[ModelScreeningSimulationCase]:
    """Generate balanced cases from all registered BayMorph models.

    Parameters
    ----------
    model_priors : mapping
        Complete parameter priors for every registered model. These
        distributions define the model-conditional training population.
    n_per_model : int
        Number of simulated observations generated for each model.
        Equal class counts imply a uniform training model prior.
    image_shape : tuple of int
        Common image shape as ``(ny, nx)``.
    psf_fwhm_prior : mapping
        Shared prior for the Gaussian PSF FWHM in pixels.
    noise_sigma_prior : mapping
        Shared prior for the independent Gaussian-noise standard deviation.
    random_state : int, default=0
        Root seed controlling parameters, nuisance quantities, and noise.

    Yields
    ------
    ModelScreeningSimulationCase
        Reproducible labelled simulations in deterministic model order.

    Notes
    -----
    The same PSF and noise prior families are used for every model class.
    This prevents those nuisance distributions from directly identifying
    the morphology label.

    This initial generator remains an idealized validation baseline.
    Empirical PSFs, correlated noise, contaminants, masks, saturation,
    and structured backgrounds require later controlled extensions.
    """
    _validate_model_priors(
        model_priors
    )

    n_per_model = int(
        n_per_model
    )

    if n_per_model <= 0:
        raise ValueError(
            "n_per_model must be positive."
        )

    if len(
        image_shape
    ) != 2:
        raise ValueError(
            "image_shape must contain exactly two dimensions."
        )

    normalized_shape = tuple(
        int(
            value
        )
        for value in image_shape
    )

    if any(
        value <= 0
        for value in normalized_shape
    ):
        raise ValueError(
            "image_shape dimensions must be positive."
        )

    random_state = int(
        random_state
    )

    if random_state < 0:
        raise ValueError(
            "random_state must be non-negative."
        )

    rng = np.random.default_rng(
        random_state
    )

    references = {
        "target_x": 0.5 * (
            normalized_shape[1]
            - 1
        ),
        "target_y": 0.5 * (
            normalized_shape[0]
            - 1
        ),
    }

    case_index = 0

    for model_name in registered_models():
        for _ in range(
            n_per_model
        ):
            parameters = {
                parameter_name: _sample_prior(
                    model_priors[
                        model_name
                    ][
                        parameter_name
                    ],
                    rng=rng,
                    references=references,
                )
                for parameter_name in parameter_names(
                    model_name
                )
            }

            psf_fwhm = _sample_prior(
                psf_fwhm_prior,
                rng=rng,
            )
            noise_sigma = _sample_prior(
                noise_sigma_prior,
                rng=rng,
            )

            case_seed = int(
                rng.integers(
                    0,
                    np.iinfo(
                        np.int32
                    ).max,
                )
            )

            yield simulate_model_screening_case(
                index=case_index,
                model_name=model_name,
                parameters=parameters,
                image_shape=normalized_shape,
                psf_fwhm=psf_fwhm,
                noise_sigma=noise_sigma,
                random_seed=case_seed,
            )

            case_index += 1


__all__ = [
    "ModelPriorConfig",
    "PriorConfig",
    "generate_model_screening_cases",
]
