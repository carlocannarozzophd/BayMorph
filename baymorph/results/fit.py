from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.axes import Axes

from baymorph.models import (
    parameter_labels as model_parameter_labels,
)

from baymorph.plotting import (
    corner_plot,
)


@dataclass
class RealFitProducts:
    """
    Container holding the products of a Bayesian galaxy fit.

    Besides storing the numerical results of the fit, this class
    provides convenience methods for generating diagnostic plots
    and summary reports.
    """
    posterior: Any | None
    best_parameters: dict[str, float]
    best_model_image: np.ndarray
    residual: np.ndarray
    norm_residual: np.ndarray
    chi2: float
    ndof: int
    reduced_chi2: float

    model_name: str = "unknown"
    noise: np.ndarray | None = None

    radius: np.ndarray | None = None
    profile_data: np.ndarray | None = None
    profile_model: np.ndarray | None = None
    profile_noise: np.ndarray | None = None

    @property
    def statistics(self) -> dict[str, float]:
        return {
            "chi2": self.chi2,
            "reduced_chi2": self.reduced_chi2,
            "ndof": self.ndof,
        }
    
    def corner(
        self,
        parameter_labels: list[str] | None = None,
        **kwargs,
    ):
        if parameter_labels is None:
            parameter_labels = model_parameter_labels(
                self.model_name,
            )

        return corner_plot(
            self.posterior.posterior_samples.T,
            parameter_labels=parameter_labels,
            **kwargs,
        )
    
    def radial_profile(
        self,
        ax: Axes | None = None,
    ):
        from baymorph.plotting import (
            plot_radial_profile_diagnostics,
        )

        return plot_radial_profile_diagnostics(
            self.radius,
            self.profile_data,
            self.profile_model,
            self.profile_noise,
            ax=ax,
        )

    def report(
        self,
        observation: Any,
        **kwargs,
    ):
        from baymorph.plotting import (
            plot_mock_fit_report,
        )

        return plot_mock_fit_report(
            observation=observation,
            fit=self,
            noise_map=self.noise,
            statistics=self.statistics,
            corner_plotter=lambda ax: self.corner(ax=ax),
            radial_profile_plotter=lambda ax: self.radial_profile(ax=ax),
            **kwargs,
        )