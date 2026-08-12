"""Result containers for joint multi-observation galaxy fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from baymorph.results.inference import InferenceResult


@dataclass
class ObservationFitProducts:
    """Pixel-space fit products for one observation.

    Parameters
    ----------
    parameters : dict of str to float
        Best-fitting parameters routed into the coordinate system of this
        observation.
    best_model_image : ndarray
        Best-fitting model rendered on the native observation grid.
    residual : ndarray
        Data-minus-model residual image.
    norm_residual : ndarray
        Residual image normalized by the noise map.
    chi2 : float
        Chi-squared contribution from this observation.
    n_valid_pixels : int
        Number of pixels contributing to the fit statistic.
    noise : ndarray
        Noise map used for this observation.
    """

    parameters: dict[str, float]
    best_model_image: np.ndarray
    residual: np.ndarray
    norm_residual: np.ndarray
    chi2: float
    n_valid_pixels: int
    noise: np.ndarray

    def __post_init__(self) -> None:
        """Normalize and validate observation-level fit products."""
        self.parameters = {
            str(name): float(value)
            for name, value in self.parameters.items()
        }

        self.best_model_image = np.asarray(
            self.best_model_image,
            dtype=float,
        )
        self.residual = np.asarray(
            self.residual,
            dtype=float,
        )
        self.norm_residual = np.asarray(
            self.norm_residual,
            dtype=float,
        )
        self.noise = np.asarray(
            self.noise,
            dtype=float,
        )

        shape = self.best_model_image.shape

        for name, array in (
            ("residual", self.residual),
            ("norm_residual", self.norm_residual),
            ("noise", self.noise),
        ):
            if array.shape != shape:
                raise ValueError(
                    f"{name} must match the best-model image shape."
                )

        self.chi2 = float(
            self.chi2
        )

        if (
            not np.isfinite(self.chi2)
            or self.chi2 < 0.0
        ):
            raise ValueError(
                "Observation chi-squared must be finite and non-negative."
            )

        self.n_valid_pixels = int(
            self.n_valid_pixels
        )

        if self.n_valid_pixels <= 0:
            raise ValueError(
                "Observation fit products require at least one valid pixel."
            )

    @property
    def statistics(self) -> dict[str, float | int]:
        """Return observation-level scalar fit diagnostics."""
        return {
            "chi2": self.chi2,
            "n_valid_pixels": self.n_valid_pixels,
        }


@dataclass
class MultiObservationFitProducts:
    """Products of one joint Bayesian multi-observation galaxy fit.

    The posterior is joint across all observations. Pixel-space products are
    stored separately for each observation because each dataset may have a
    different native grid, PSF, noise model, and coordinate transformation.

    Parameters
    ----------
    model_name : str
        Name of the fitted galaxy model.
    posterior : InferenceResult
        Single joint posterior and evidence result.
    best_parameters : dict of str to float
        Global maximum-likelihood sampled parameters.
    observations : dict of str to ObservationFitProducts
        Observation-indexed pixel-space fit products.
    chi2 : float
        Sum of observation-level chi-squared contributions.
    ndof : int
        Nominal joint degrees of freedom, defined as the total number of valid
        pixels minus the number of globally sampled parameters.
    reduced_chi2 : float
        Nominal joint reduced chi-squared.
    """

    model_name: str
    posterior: InferenceResult
    best_parameters: dict[str, float]
    observations: dict[str, ObservationFitProducts]
    chi2: float
    ndof: int
    reduced_chi2: float

    def __post_init__(self) -> None:
        """Normalize and validate joint fit products."""
        self.model_name = str(
            self.model_name
        ).strip()

        if not self.model_name:
            raise ValueError(
                "Multi-observation fit products require a model name."
            )

        if not isinstance(
            self.posterior,
            InferenceResult,
        ):
            raise TypeError(
                "posterior must be an InferenceResult."
            )

        self.best_parameters = {
            str(name): float(value)
            for name, value in self.best_parameters.items()
        }

        if not self.observations:
            raise ValueError(
                "Multi-observation fit products require observations."
            )

        self.observations = dict(
            self.observations
        )

        for name, products in self.observations.items():
            if not isinstance(name, str) or not name:
                raise TypeError(
                    "Observation names must be non-empty strings."
                )

            if not isinstance(
                products,
                ObservationFitProducts,
            ):
                raise TypeError(
                    "Observation products must be "
                    "ObservationFitProducts instances."
                )

        self.chi2 = float(
            self.chi2
        )
        self.ndof = int(
            self.ndof
        )
        self.reduced_chi2 = float(
            self.reduced_chi2
        )

        if (
            not np.isfinite(self.chi2)
            or self.chi2 < 0.0
        ):
            raise ValueError(
                "Joint chi-squared must be finite and non-negative."
            )

        if self.ndof > 0:
            if not np.isfinite(
                self.reduced_chi2
            ):
                raise ValueError(
                    "Reduced chi-squared must be finite for positive "
                    "degrees of freedom."
                )
        elif not np.isnan(
            self.reduced_chi2
        ):
            raise ValueError(
                "Reduced chi-squared must be NaN when joint degrees of "
                "freedom are non-positive."
            )

    @property
    def statistics(self) -> dict[str, float | int]:
        """Return joint scalar goodness-of-fit diagnostics."""
        return {
            "chi2": self.chi2,
            "ndof": self.ndof,
            "reduced_chi2": self.reduced_chi2,
        }

    @property
    def logz(self) -> float | None:
        """Return the joint logarithmic Bayesian evidence."""
        return self.posterior.logz

    @property
    def logz_error(self) -> float | None:
        """Return the reported uncertainty on the joint evidence."""
        return self.posterior.logz_error

    @property
    def parameter_names(self) -> list[str]:
        """Return sampled global parameter names."""
        return self.posterior.parameter_names

    def summary(self) -> dict[str, Any]:
        """Return a concise summary of the joint fit."""
        return {
            "model": self.model_name,
            "statistics": self.statistics,
            "logz": self.logz,
            "logzerr": self.logz_error,
            "best_parameters": dict(
                self.best_parameters
            ),
            "observations": {
                name: products.statistics
                for name, products in self.observations.items()
            },
        }
