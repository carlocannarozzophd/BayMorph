"""Synthetic observations for model-screening validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from baymorph.analysis.model_screening import SCREENING_MODELS
from baymorph.fitting.model_factories import ObservedModelFactory
from baymorph.models import parameter_names
from baymorph.simulation.psf import GaussianPSF


FloatArray = NDArray[np.float64]


def screening_parameter_names(
    model_name: str,
) -> tuple[str, ...]:
    """Return the simulated parameter contract of one screening model.

    Every screening observation carries one constant residual background,
    ``background_level``, in addition to the registered morphology
    parameters (AD-031).
    """
    return (
        tuple(
            parameter_names(
                model_name
            )
        )
        + (
            "background_level",
        )
    )


@dataclass(frozen=True, slots=True)
class ModelScreeningSimulationCase:
    """One labelled synthetic observation for model screening."""

    index: int
    model_name: str

    image: FloatArray
    noiseless_image: FloatArray

    parameters: Mapping[str, float]

    psf_fwhm: float
    noise_sigma: float
    random_seed: int

    def __post_init__(self) -> None:
        """Validate and freeze the simulated case."""
        index = int(
            self.index
        )
        random_seed = int(
            self.random_seed
        )

        if index < 0:
            raise ValueError(
                "Simulation-case index must be non-negative."
            )

        if random_seed < 0:
            raise ValueError(
                "Simulation-case random seed must be non-negative."
            )

        if self.model_name not in SCREENING_MODELS:
            raise ValueError(
                f"Not a model-screening model: {self.model_name!r}."
            )

        image = np.asarray(
            self.image,
            dtype=float,
        )
        noiseless_image = np.asarray(
            self.noiseless_image,
            dtype=float,
        )

        if image.ndim != 2:
            raise ValueError(
                "A screening simulation image must be two-dimensional."
            )

        if noiseless_image.shape != image.shape:
            raise ValueError(
                "Noisy and noiseless screening images must have "
                "identical shapes."
            )

        if not np.all(
            np.isfinite(
                image
            )
        ):
            raise ValueError(
                "The screening simulation image must be finite."
            )

        if not np.all(
            np.isfinite(
                noiseless_image
            )
        ):
            raise ValueError(
                "The noiseless screening image must be finite."
            )

        expected_parameters = set(
            screening_parameter_names(
                self.model_name
            )
        )
        supplied_parameters = set(
            self.parameters
        )

        if supplied_parameters != expected_parameters:
            missing = sorted(
                expected_parameters - supplied_parameters
            )
            extra = sorted(
                supplied_parameters - expected_parameters
            )

            raise ValueError(
                "Simulation parameters must match the complete model "
                f"contract. Missing: {missing}. Extra: {extra}."
            )

        normalized_parameters = {
            name: float(
                value
            )
            for name, value in self.parameters.items()
        }

        if any(
            not np.isfinite(
                value
            )
            for value in normalized_parameters.values()
        ):
            raise ValueError(
                "Simulation parameters must be finite."
            )

        psf_fwhm = float(
            self.psf_fwhm
        )
        noise_sigma = float(
            self.noise_sigma
        )

        if not np.isfinite(
            psf_fwhm
        ) or psf_fwhm <= 0.0:
            raise ValueError(
                "Simulation PSF FWHM must be finite and positive."
            )

        if not np.isfinite(
            noise_sigma
        ) or noise_sigma < 0.0:
            raise ValueError(
                "Simulation noise sigma must be finite and non-negative."
            )

        image.setflags(
            write=False
        )
        noiseless_image.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "index",
            index,
        )
        object.__setattr__(
            self,
            "random_seed",
            random_seed,
        )
        object.__setattr__(
            self,
            "image",
            image,
        )
        object.__setattr__(
            self,
            "noiseless_image",
            noiseless_image,
        )
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(
                normalized_parameters
            ),
        )
        object.__setattr__(
            self,
            "psf_fwhm",
            psf_fwhm,
        )
        object.__setattr__(
            self,
            "noise_sigma",
            noise_sigma,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the simulated image shape."""
        return self.image.shape


def simulate_model_screening_case(
    *,
    model_name: str,
    parameters: Mapping[str, float],
    image_shape: tuple[int, int],
    psf_fwhm: float,
    noise_sigma: float,
    random_seed: int,
    index: int = 0,
) -> ModelScreeningSimulationCase:
    """Generate one labelled observation from a BayMorph forward model.

    This initial validation simulator uses a circular Gaussian PSF and
    independent Gaussian noise. It is not yet a production-realism claim.
    """
    if model_name not in SCREENING_MODELS:
        raise ValueError(
            f"Not a model-screening model: {model_name!r}."
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

    expected_parameters = set(
        screening_parameter_names(
            model_name
        )
    )
    supplied_parameters = set(
        parameters
    )

    if supplied_parameters != expected_parameters:
        missing = sorted(
            expected_parameters - supplied_parameters
        )
        extra = sorted(
            supplied_parameters - expected_parameters
        )

        raise ValueError(
            "Simulation parameters must match the complete model "
            f"contract. Missing: {missing}. Extra: {extra}."
        )

    normalized_parameters = {
        name: float(
            value
        )
        for name, value in parameters.items()
    }

    psf_fwhm = float(
        psf_fwhm
    )
    noise_sigma = float(
        noise_sigma
    )
    random_seed = int(
        random_seed
    )

    if random_seed < 0:
        raise ValueError(
            "random_seed must be non-negative."
        )

    psf = GaussianPSF(
        fwhm=psf_fwhm
    )

    model = ObservedModelFactory(
        model_name,
        background_model="constant",
        psf=psf,
    )(
        **normalized_parameters,
    )

    noiseless_image = np.asarray(
        model.render(
            normalized_shape
        ),
        dtype=float,
    )

    rng = np.random.default_rng(
        random_seed
    )

    noise = rng.normal(
        loc=0.0,
        scale=noise_sigma,
        size=normalized_shape,
    )

    image = noiseless_image + noise

    return ModelScreeningSimulationCase(
        index=index,
        model_name=model_name,
        image=image,
        noiseless_image=noiseless_image,
        parameters=normalized_parameters,
        psf_fwhm=psf_fwhm,
        noise_sigma=noise_sigma,
        random_seed=random_seed,
    )


__all__ = [
    "ModelScreeningSimulationCase",
    "simulate_model_screening_case",
]
