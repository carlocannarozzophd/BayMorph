from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.axes import Axes

from baymorph.models import parameter_labels as model_parameter_labels
from baymorph.plotting import corner_plot
from baymorph.results.inference import InferenceResult


@dataclass
class RealFitProducts:
    """Container for the products of a Bayesian galaxy fit.

    Parameters
    ----------
    posterior : InferenceResult or None
        Posterior result returned by the sampler.
    best_parameters : dict of str to float
        Best-fit parameter values.
    best_model_image : ndarray
        Rendered best-fit model image.
    residual : ndarray
        Data-minus-model residual image.
    norm_residual : ndarray
        Residual image normalized by the noise map.
    chi2 : float
        Chi-squared value of the best-fit model.
    ndof : int
        Number of degrees of freedom.
    reduced_chi2 : float
        Reduced chi-squared value.
    model_name : str, default="unknown"
        Name of the fitted model.
    noise : ndarray, optional
        Noise map used during fitting.
    radius : ndarray, optional
        Radial-bin centers used by radial-profile diagnostics.
    profile_data : ndarray, optional
        Observed radial profile.
    profile_model : ndarray, optional
        Model radial profile.
    profile_noise : ndarray, optional
        Radial noise profile.
    """

    posterior: InferenceResult | None
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
        """dict of str to float : Scalar goodness-of-fit statistics."""
        return {
            "chi2": self.chi2,
            "reduced_chi2": self.reduced_chi2,
            "ndof": self.ndof,
        }

    def summary(self) -> dict[str, Any]:
        """Return a concise summary of the fitted model.

        Returns
        -------
        dict
            Model name, fit statistics, Bayesian evidence, and best-fitting
            parameter values.
        """
        logz = None
        logzerr = None

        if self.posterior is not None:
            logz = self.posterior.logz
            logzerr = self.posterior.logz_error

        return {
            "model": self.model_name,
            "statistics": self.statistics,
            "logz": logz,
            "logzerr": logzerr,
            "best_parameters": dict(
                self.best_parameters
            ),
        }

    def corner(
        self,
        parameter_labels: list[str] | None = None,
        **kwargs,
    ):
        """Plot the posterior corner diagram.

        Parameters
        ----------
        parameter_labels : list of str, optional
            Labels used for the corner-plot axes. If None, labels are inferred
            from ``model_name``.
        **kwargs
            Additional keyword arguments passed to
            :func:`baymorph.plotting.corner_plot`.

        Returns
        -------
        tuple
            Figure, axes grid, and parameter array returned by
            :func:`baymorph.plotting.corner_plot`.
        """
        if parameter_labels is None:
            parameter_labels = model_parameter_labels(
                self.model_name,
            )

        return corner_plot(
            self.posterior.posterior.equal_weight_samples.T,
            parameter_labels=parameter_labels,
            **kwargs,
        )

    def radial_profile(
        self,
        ax: Axes | None = None,
    ):
        """Plot radial-profile diagnostics.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axis whose position is replaced by the diagnostic layout.

        Returns
        -------
        tuple
            Figure and radial-profile axes returned by
            :func:`baymorph.plotting.plot_radial_profile_diagnostics`.
        """
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
        """Plot a high-level mock-fit report.

        Parameters
        ----------
        observation : object
            Observation object passed to
            :func:`baymorph.plotting.plot_mock_fit_report`.
        **kwargs
            Additional keyword arguments passed to
            :func:`baymorph.plotting.plot_mock_fit_report`.

        Returns
        -------
        tuple
            Figure and axes returned by
            :func:`baymorph.plotting.plot_mock_fit_report`.
        """
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