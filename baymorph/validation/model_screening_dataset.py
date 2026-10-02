"""Balanced simulation datasets for model-screening validation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from baymorph.analysis.model_screening import SCREENING_MODELS
from baymorph.pipeline.prior_config import parse_prior_config
from baymorph.validation.model_screening_simulation import (
    ModelScreeningSimulationCase,
    screening_parameter_names,
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
    """Require complete priors for every screening model."""
    expected_models = set(
        SCREENING_MODELS
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
            "Model-screening priors must cover every screening model. "
            f"Missing: {missing}. Extra: {extra}."
        )

    for model_name in SCREENING_MODELS:
        expected_parameters = set(
            screening_parameter_names(
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
    """Generate balanced cases from all screening models.

    Parameters
    ----------
    model_priors : mapping
        Complete parameter priors for every screening model. These
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

    for model_name in SCREENING_MODELS:
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
                for parameter_name in screening_parameter_names(
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


CONTROLLED_POPULATION_PRIOR_NAMES = (
    "total_flux",
    "outer_re_pixels",
    "sersic_n",
    "inner_n",
    "outer_n",
    "outer_q",
    "inner_q",
    "position_angle_radians",
    "component_angle_offset_radians",
    "inner_flux_fraction",
    "inner_re_fraction",
    "point_source_fraction",
    "centroid_x_offset_pixels",
    "centroid_y_offset_pixels",
    "background_level",
    "psf_fwhm_pixels",
    "noise_sigma",
)

ControlledPopulationPriorConfig = Mapping[
    str,
    PriorConfig,
]


def _wrap_position_angle(
    angle: float,
) -> float:
    """Wrap a position angle into the BayMorph half-turn interval."""
    return float(
        (
            angle
            + 0.5 * np.pi
        )
        % np.pi
        - 0.5 * np.pi
    )


def _validate_controlled_population_priors(
    population_priors: ControlledPopulationPriorConfig,
) -> None:
    """Require the complete controlled latent-prior contract."""
    expected = set(
        CONTROLLED_POPULATION_PRIOR_NAMES
    )
    supplied = set(
        population_priors
    )

    if supplied != expected:
        missing = sorted(
            expected - supplied
        )
        extra = sorted(
            supplied - expected
        )

        raise ValueError(
            "Controlled population priors must cover the complete "
            f"latent contract. Missing: {missing}. Extra: {extra}."
        )


def controlled_parameters_from_latents(
    *,
    model_name: str,
    latents: Mapping[str, float],
    image_shape: tuple[int, int],
) -> dict[str, float]:
    """Map controlled population latents to one model parameter vector.

    The common ``total_flux`` latent is conserved exactly for every model.
    Two-component models split it through ``inner_flux_fraction``.
    The Sérsic-plus-point-source model splits it through
    ``point_source_fraction``.

    ``outer_re_pixels`` defines the Sérsic radius for one-component hosts
    and the outer-component radius for two-component models. The inner
    radius is obtained through ``inner_re_fraction``.
    """
    if model_name not in SCREENING_MODELS:
        raise ValueError(
            f"Unknown model name: {model_name!r}."
        )

    if len(image_shape) != 2:
        raise ValueError(
            "image_shape must contain exactly two dimensions."
        )

    normalized_shape = tuple(
        int(value)
        for value in image_shape
    )
    if any(
        value <= 0
        for value in normalized_shape
    ):
        raise ValueError(
            "image_shape dimensions must be positive."
        )

    expected_latents = set(
        CONTROLLED_POPULATION_PRIOR_NAMES
    )
    supplied_latents = set(
        latents
    )
    if supplied_latents != expected_latents:
        missing = sorted(
            expected_latents - supplied_latents
        )
        extra = sorted(
            supplied_latents - expected_latents
        )
        raise ValueError(
            "Controlled latent values must cover the complete contract. "
            f"Missing: {missing}. Extra: {extra}."
        )

    values = {
        name: float(
            latents[name]
        )
        for name in CONTROLLED_POPULATION_PRIOR_NAMES
    }

    if not all(
        np.isfinite(value)
        for value in values.values()
    ):
        raise ValueError(
            "Controlled latent values must all be finite."
        )

    if values["total_flux"] <= 0.0:
        raise ValueError(
            "total_flux must be positive."
        )

    if values["outer_re_pixels"] <= 0.0:
        raise ValueError(
            "outer_re_pixels must be positive."
        )

    for name in (
        "sersic_n",
        "inner_n",
        "outer_n",
    ):
        if values[name] <= 0.0:
            raise ValueError(
                f"{name} must be positive."
            )

    for name in (
        "outer_q",
        "inner_q",
    ):
        if not 0.0 < values[name] <= 1.0:
            raise ValueError(
                f"{name} must lie in (0, 1]."
            )

    for name in (
        "inner_flux_fraction",
        "point_source_fraction",
    ):
        if not 0.0 < values[name] < 1.0:
            raise ValueError(
                f"{name} must lie strictly in (0, 1)."
            )

    if not 0.0 < values["inner_re_fraction"] <= 1.0:
        raise ValueError(
            "inner_re_fraction must lie in (0, 1]."
        )

    if values["psf_fwhm_pixels"] <= 0.0:
        raise ValueError(
            "psf_fwhm_pixels must be positive."
        )

    if values["noise_sigma"] <= 0.0:
        raise ValueError(
            "noise_sigma must be positive."
        )

    x0 = (
        0.5
        * (
            normalized_shape[1]
            - 1
        )
        + values["centroid_x_offset_pixels"]
    )
    y0 = (
        0.5
        * (
            normalized_shape[0]
            - 1
        )
        + values["centroid_y_offset_pixels"]
    )

    if not 0.0 <= x0 < normalized_shape[1]:
        raise ValueError(
            "The controlled x centroid lies outside the image."
        )

    if not 0.0 <= y0 < normalized_shape[0]:
        raise ValueError(
            "The controlled y centroid lies outside the image."
        )

    total_flux = values["total_flux"]
    outer_re = values["outer_re_pixels"]
    outer_q = values["outer_q"]
    position_angle = _wrap_position_angle(
        values["position_angle_radians"]
    )
    background_level = values["background_level"]

    if model_name == "single_sersic":
        return {
            "flux": total_flux,
            "re": outer_re,
            "n": values["sersic_n"],
            "q": outer_q,
            "theta": position_angle,
            "x0": x0,
            "y0": y0,
            "background_level": background_level,
        }

    if model_name == "sersic_point_source":
        point_fraction = values[
            "point_source_fraction"
        ]
        return {
            "flux_sersic": total_flux
            * (
                1.0
                - point_fraction
            ),
            "re": outer_re,
            "n": values["sersic_n"],
            "q": outer_q,
            "theta": position_angle,
            "flux_point_source": total_flux
            * point_fraction,
            "x0": x0,
            "y0": y0,
            "background_level": background_level,
        }

    inner_fraction = values[
        "inner_flux_fraction"
    ]
    shared_parameters = {
        "flux_bulge": total_flux
        * inner_fraction,
        "re_bulge": outer_re
        * values[
            "inner_re_fraction"
        ],
        "n_bulge": values["inner_n"],
        "q_bulge": values["inner_q"],
        "theta_bulge": _wrap_position_angle(
            position_angle
            + values[
                "component_angle_offset_radians"
            ]
        ),
        "flux_disk": total_flux
        * (
            1.0
            - inner_fraction
        ),
        "re_disk": outer_re,
        "q_disk": outer_q,
        "theta_disk": position_angle,
        "x0": x0,
        "y0": y0,
        "background_level": background_level,
    }

    if model_name == "double_sersic":
        return {
            **shared_parameters,
            "n_disk": values["outer_n"],
        }

    return shared_parameters


def generate_controlled_model_screening_cases(
    *,
    population_priors: ControlledPopulationPriorConfig,
    n_per_model: int,
    image_shape: tuple[int, int],
    random_state: int = 0,
) -> Iterator[ModelScreeningSimulationCase]:
    """Generate a balanced, nuisance-controlled screening population.

    Every class draws the same latent total-flux, outer-size, centroid,
    background, PSF, and noise distributions. Composite-model fluxes are derived
    from a total flux and a component fraction, preventing total brightness
    from acting as an unintended class label.

    This remains an idealized baseline with Gaussian PSFs and independent
    Gaussian noise.
    """
    _validate_controlled_population_priors(
        population_priors
    )

    normalized_count = int(
        n_per_model
    )
    if normalized_count <= 0:
        raise ValueError(
            "n_per_model must be positive."
        )

    if len(image_shape) != 2:
        raise ValueError(
            "image_shape must contain exactly two dimensions."
        )

    normalized_shape = tuple(
        int(value)
        for value in image_shape
    )
    if any(
        value <= 0
        for value in normalized_shape
    ):
        raise ValueError(
            "image_shape dimensions must be positive."
        )

    normalized_seed = int(
        random_state
    )
    if normalized_seed < 0:
        raise ValueError(
            "random_state must be non-negative."
        )

    rng = np.random.default_rng(
        normalized_seed
    )
    case_index = 0

    for model_name in SCREENING_MODELS:
        for _ in range(
            normalized_count
        ):
            latents = {
                name: _sample_prior(
                    population_priors[name],
                    rng=rng,
                )
                for name in CONTROLLED_POPULATION_PRIOR_NAMES
            }

            parameters = controlled_parameters_from_latents(
                model_name=model_name,
                latents=latents,
                image_shape=normalized_shape,
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
                psf_fwhm=latents[
                    "psf_fwhm_pixels"
                ],
                noise_sigma=latents[
                    "noise_sigma"
                ],
                random_seed=case_seed,
            )

            case_index += 1


__all__ += [
    "CONTROLLED_POPULATION_PRIOR_NAMES",
    "ControlledPopulationPriorConfig",
    "controlled_parameters_from_latents",
    "generate_controlled_model_screening_cases",
]
